"""
Production FastAPI Microservice for @AmazonHelp AI Support Agent.
Hardened with Inbound PII Masking, Stored XSS Prevention, Session Isolation,
Sliding-Window Rate Limiting, and Configurable API Key Authentication.
"""
import os
import re
import html
import time
import logging
import threading
from collections import defaultdict
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator

from src.agent import AmazonSupportAgent
from src.config import TARGET_BRAND
from src.taxonomy import mask_pii

logger = logging.getLogger(__name__)

app = FastAPI(
    title='@AmazonHelp AI Customer Support Service',
    description='Production-minded AI customer support agent and deterministic triage engine for @AmazonHelp.',
    version='1.1.0'
)

# Lazy singleton agent instance
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

# Session identifier regex validation (1-128 chars: alphanumeric, dash, underscore, dot)
SESSION_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-\.]{1,128}$")

# Thread-safe bounded in-memory session store for multi-turn thread tracking
SESSION_STORE: Dict[str, List[Dict[str, str]]] = {}
SESSION_STORE_LOCK = threading.Lock()
MAX_STORED_SESSIONS = 2000

def _get_session_history(session_id: str) -> List[Dict[str, str]]:
    """Retrieves session history thread-safely."""
    with SESSION_STORE_LOCK:
        return list(SESSION_STORE.get(session_id, []))

def _append_session_turn(session_id: str, turn: Dict[str, str]) -> None:
    """Appends a turn to session history thread-safely with FIFO eviction when full."""
    with SESSION_STORE_LOCK:
        if session_id not in SESSION_STORE:
            if len(SESSION_STORE) >= MAX_STORED_SESSIONS:
                oldest_key = next(iter(SESSION_STORE))
                del SESSION_STORE[oldest_key]
            SESSION_STORE[session_id] = []
        SESSION_STORE[session_id].append(turn)

def clear_session_store() -> None:
    """Clears session store (useful for testing isolation)."""
    with SESSION_STORE_LOCK:
        SESSION_STORE.clear()

# Configurable Sliding-Window Rate Limiter (Default: 60 req / min per IP)
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

class ChatRequest(BaseModel):
    message: str = Field(..., description='Inbound customer tweet or message text', min_length=1, max_length=4096)
    session_id: Optional[str] = Field(None, description='Optional session identifier for multi-turn tracking')
    history: Optional[List[Dict[str, str]]] = Field(None, description='Optional historical turns')

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

@app.get('/health', response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Returns microservice operational status, model metadata, and active session count."""
    current_agent = get_agent()
    kb_len = len(current_agent.retriever.df_kb) if current_agent.retriever and hasattr(current_agent.retriever, 'df_kb') else 55011
    with SESSION_STORE_LOCK:
        active_sessions = len(SESSION_STORE)
    return HealthResponse(
        status='healthy',
        brand=TARGET_BRAND,
        model_name=current_agent.model_name,
        kb_size=kb_len,
        session_count=active_sessions
    )

@app.post('/v1/chat', response_model=ChatResponse, dependencies=[Depends(check_rate_limit), Depends(verify_api_key)])
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """Processes customer inquiry with stateful multi-turn session tracking and reply generation."""
    session_id = payload.session_id
    history = payload.history
    
    try:
        current_agent = get_agent()
        if session_id and not history:
            history = _get_session_history(session_id)

        result = current_agent.process(
            customer_text=payload.message,
            conversation_history=history,
            session_id=session_id
        )

        # Inbound PII masking and stored XSS prevention in bounded session store
        if session_id:
            _append_session_turn(session_id, {
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
