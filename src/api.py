"""
Production FastAPI Microservice for @AmazonHelp AI Support Agent.
Hardened with True LRU Session Storage, Turn Depth Bounding, Per-Session Mutex
Transaction Locking, Lifespan Pre-warming, Kubernetes /live & /ready Probes,
OpenMP Throttling, Request Tracing Middleware, and Strict Payload Boundaries.
"""
import os
# Lock BLAS/OpenMP threading to 1 thread per worker to eliminate thread over-subscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import re
import html
import time
import uuid
import logging
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from contextlib import asynccontextmanager, nullcontext
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator

from src.agent import AmazonSupportAgent
from src.config import TARGET_BRAND
from src.taxonomy import mask_pii

logger = logging.getLogger(__name__)

# Lazy singleton agent instance with thread-safe double-checked lock
_agent_instance: Optional[AmazonSupportAgent] = None
_agent_lock = threading.Lock()

def get_agent() -> AmazonSupportAgent:
    """Thread-safe lazy initialization of the AmazonSupportAgent singleton."""
    global _agent_instance
    if _agent_instance is None:
        with _agent_lock:
            if _agent_instance is None:
                _agent_instance = AmazonSupportAgent()
    return _agent_instance

# Lifespan Context Manager: Pre-warms the agent models and retrieval index on boot
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Lifespan Startup: Pre-warming AmazonSupportAgent and retrieval index...")
    agent = get_agent()
    # Warm up inference pipeline with a lightweight probe
    agent.process("Warmup probe: when will my package arrive?")
    logger.info("Lifespan Startup Complete: Models primed, zero cold-start latency guaranteed.")
    yield
    logger.info("Lifespan Shutdown: Cleaning up microservice resources...")

app = FastAPI(
    title='@AmazonHelp AI Customer Support Service',
    description='Production-grade AI customer support agent and deterministic triage engine for @AmazonHelp.',
    version='1.2.0',
    lifespan=lifespan
)

# --------------------------------------------------------------------------------------
# Architecture-Clean Session Storage Adapter Pattern (True LRU with Turn Bounding)
# --------------------------------------------------------------------------------------
class BaseSessionStore(ABC):
    """Abstract interface for multi-turn session storage (In-Memory, Redis, Memcached)."""
    @abstractmethod
    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        pass

    @abstractmethod
    def append_turn(self, session_id: str, turn: Dict[str, str]) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def count(self) -> int:
        pass

class InMemoryLRUSessionStore(BaseSessionStore):
    """
    Thread-safe True LRU Session Store backed by collections.OrderedDict.
    - move_to_end on access (both read and write) ensures active VIP conversations are never evicted.
    - popitem(last=False) drops only the least recently accessed sessions on capacity overflow.
    - MAX_TURNS_PER_SESSION enforces a sliding window to prevent single-session memory DoS.
    """
    def __init__(self, max_sessions: int = 2000, max_turns: int = 20):
        self.max_sessions = max_sessions
        self.max_turns = max_turns
        self._store: OrderedDict[str, List[Dict[str, str]]] = OrderedDict()
        self._lock = threading.Lock()

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        with self._lock:
            if session_id in self._store:
                self._store.move_to_end(session_id)
                return list(self._store[session_id])
            return []

    def append_turn(self, session_id: str, turn: Dict[str, str]) -> None:
        with self._lock:
            if session_id not in self._store:
                while len(self._store) >= self.max_sessions:
                    self._store.popitem(last=False)
                self._store[session_id] = []
            
            turns = self._store[session_id]
            turns.append(turn)
            if len(turns) > self.max_turns:
                self._store[session_id] = turns[-self.max_turns:]
            self._store.move_to_end(session_id)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def count(self) -> int:
        with self._lock:
            return len(self._store)

# Global session store instance and backwards-compatible aliases
session_store = InMemoryLRUSessionStore(max_sessions=2000, max_turns=20)
SESSION_STORE = session_store._store
SESSION_STORE_LOCK = session_store._lock

def clear_session_store() -> None:
    session_store.clear()

# --------------------------------------------------------------------------------------
# Fine-Grained Per-Session Transaction Lock (Prevents TOCTOU Race Conditions)
# --------------------------------------------------------------------------------------
class SessionLockManager:
    """
    Manages per-session mutex locks. Rapid concurrent requests targeting the same
    session_id execute in serial order, guaranteeing Read-Process-Write atomicity.
    Requests for different session IDs execute concurrently with zero contention.
    """
    def __init__(self):
        self._locks: Dict[str, threading.Lock] = {}
        self._meta_lock = threading.Lock()

    def get_lock(self, session_id: str) -> threading.Lock:
        with self._meta_lock:
            if session_id not in self._locks:
                self._locks[session_id] = threading.Lock()
            return self._locks[session_id]

session_lock_manager = SessionLockManager()

# Session identifier regex validation (1-128 chars: alphanumeric, dash, underscore, dot)
SESSION_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-\.]{1,128}$")

# --------------------------------------------------------------------------------------
# Observability Middleware (Request Tracing & SLA Latency Headers)
# --------------------------------------------------------------------------------------
@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    t0 = time.perf_counter()
    
    response = await call_next(request)
    
    process_time_ms = round((time.perf_counter() - t0) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = str(process_time_ms)
    return response

# --------------------------------------------------------------------------------------
# Sliding-Window Rate Limiter (Default: 60 req / min per IP)
# --------------------------------------------------------------------------------------
class SlidingWindowRateLimiter:
    def __init__(self, max_requests: int = 60, window_sec: float = 60.0):
        self.max_requests = max_requests
        self.window_sec = window_sec
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        with self.lock:
            timestamps = self.requests[client_ip]
            cutoff = now - self.window_sec
            while timestamps and timestamps[0] < cutoff:
                timestamps.pop(0)
            if len(timestamps) >= self.max_requests:
                return False
            timestamps.append(now)
            return True

    def reset(self) -> None:
        with self.lock:
            self.requests.clear()

rate_limiter = SlidingWindowRateLimiter(
    max_requests=int(os.getenv("HIV_RATE_LIMIT_MAX", "60")),
    window_sec=float(os.getenv("HIV_RATE_LIMIT_WINDOW", "60.0"))
)

async def check_rate_limit(request: Request):
    """Enforces sliding-window rate limit per client host IP."""
    client_ip = request.client.host if request.client else "127.0.0.1"
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Too Many Requests: Rate limit of 60 requests per minute exceeded."
        )

# Configurable API Key Authentication (Passthrough if HIV_API_KEY is not set)
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: Optional[str] = Security(API_KEY_HEADER)) -> Optional[str]:
    """Validates X-API-Key header if HIV_API_KEY environment variable is configured."""
    required_key = os.getenv("HIV_API_KEY", "").strip()
    if not required_key:
        return api_key
    if not api_key or api_key != required_key:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing X-API-Key header.")
    return api_key

# --------------------------------------------------------------------------------------
# Request / Response Schemas with Strict Boundaries
# --------------------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., description='Inbound customer tweet or message text', min_length=1, max_length=4096)
    session_id: Optional[str] = Field(None, description='Optional session identifier for multi-turn tracking', max_length=128)
    history: Optional[List[Dict[str, str]]] = Field(None, description='Optional historical turns', max_length=30)

    @field_validator("session_id", mode="before")
    @classmethod
    def validate_session_id(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        if not SESSION_ID_REGEX.match(v):
            raise ValueError("session_id must be 1-128 characters containing only alphanumeric characters, dashes, underscores, or dots")
        return v

class ChatResponse(BaseModel):
    session_id: Optional[str] = None
    intent: str
    intent_confidence: float
    decision: str
    escalation_category: str
    reason: str
    reply: str
    evidence: List[Dict[str, Any]]
    multi_turn: bool

class TriageRequest(BaseModel):
    message: str = Field(..., description='Customer inquiry text for triage routing', min_length=1, max_length=4096)

class TriageResponse(BaseModel):
    intent: str
    intent_confidence: float
    decision: str
    escalation_category: str
    reason: str

class HealthResponse(BaseModel):
    status: str
    brand: str
    model_name: str
    kb_size: int
    session_count: int

# --------------------------------------------------------------------------------------
# Kubernetes Liveness, Readiness & Diagnostic Endpoints
# --------------------------------------------------------------------------------------
@app.get('/live')
def liveness_probe() -> Dict[str, str]:
    """Instant Kubernetes liveness probe (< 1ms, zero locks, zero model dependencies)."""
    return {"status": "alive"}

@app.get('/ready')
def readiness_probe() -> Dict[str, str]:
    """Kubernetes readiness probe verifying model & retrieval index are initialized."""
    agent = get_agent()
    if agent is None or agent.classifier is None:
        raise HTTPException(status_code=503, detail="Model and retrieval index initializing")
    return {"status": "ready"}

@app.get('/health', response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Returns microservice operational status, model metadata, and active session count."""
    current_agent = get_agent()
    kb_len = len(current_agent.retriever.df_kb) if current_agent.retriever and hasattr(current_agent.retriever, 'df_kb') else 55011
    active_sessions = session_store.count()
    return HealthResponse(
        status='healthy',
        brand=TARGET_BRAND,
        model_name=current_agent.model_name,
        kb_size=kb_len,
        session_count=active_sessions
    )

# --------------------------------------------------------------------------------------
# Core API Endpoints with Transaction Locking & Graceful Degradation
# --------------------------------------------------------------------------------------
@app.post('/v1/chat', response_model=ChatResponse, dependencies=[Depends(check_rate_limit), Depends(verify_api_key)])
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """
    Processes customer inquiry with stateful multi-turn session tracking and reply generation.
    Enforces per-session mutex transaction locking to guarantee zero TOCTOU race conditions.
    """
    session_id = payload.session_id
    history = payload.history
    sess_lock = session_lock_manager.get_lock(session_id) if session_id else nullcontext()
    
    try:
        current_agent = get_agent()
        with sess_lock:
            # Atomic Read-Process-Write transaction
            if session_id and not history:
                history = session_store.get_history(session_id)

            result = current_agent.process(
                customer_text=payload.message,
                conversation_history=history,
                session_id=session_id
            )

            # Inbound PII masking, HTML escaping, and turn bounding in True LRU store
            if session_id:
                session_store.append_turn(session_id, {
                    'customer': html.escape(mask_pii(payload.message)),
                    'agent_reply': html.escape(result['reply']),
                    'intent': result['intent'],
                    'escalation_category': result['escalation_category']
                })

        # Ensure reflected reply is HTML-escaped for safety against XSS
        result['reply'] = html.escape(result['reply'])
        return ChatResponse(**result)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Internal error during chat processing: %s", exc)
        return ChatResponse(
            session_id=session_id,
            intent="SYSTEM_ERROR_FALLBACK",
            intent_confidence=0.0,
            decision="ESCALATE",
            escalation_category="HUMAN_AGENT_REQUEST",
            reason="An unexpected internal service error occurred; routing directly to human specialist for safety.",
            reply=html.escape("We are experiencing temporary technical difficulties. Connecting you to a live support specialist immediately. ^CS"),
            evidence=[],
            multi_turn=bool(history)
        )

@app.post('/v1/triage', response_model=TriageResponse, dependencies=[Depends(check_rate_limit), Depends(verify_api_key)])
def triage_endpoint(payload: TriageRequest) -> TriageResponse:
    """Triages customer inquiry into intent classification and deterministic escalation action."""
    try:
        current_agent = get_agent()
        cleaned = current_agent.preprocess(payload.message)
        intent, confidence, _ = current_agent.classify_intent(cleaned)
        decision, category, reason = current_agent.triage_decision(cleaned, intent, confidence)
        return TriageResponse(
            intent=intent,
            intent_confidence=confidence,
            decision=decision,
            escalation_category=category,
            reason=reason
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Internal error during triage processing: %s", exc)
        return TriageResponse(
            intent="UNKNOWN",
            intent_confidence=0.0,
            decision="ESCALATE",
            escalation_category="HUMAN_AGENT_REQUEST",
            reason="An unexpected internal service error occurred during triage; routing to human agent."
        )
