"""
Production FastAPI Microservice for @AmazonHelp AI Support Agent.
"""
import threading
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent import AmazonSupportAgent
from src.config import TARGET_BRAND

app = FastAPI(
    title='@AmazonHelp AI Customer Support Service',
    description='Production-minded AI customer support agent and deterministic triage engine for @AmazonHelp.',
    version='1.0.0'
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

class ChatRequest(BaseModel):
    message: str = Field(..., description='Inbound customer tweet or message text', min_length=1)
    session_id: Optional[str] = Field(None, description='Optional session identifier for multi-turn tracking')
    history: Optional[List[Dict[str, str]]] = Field(None, description='Optional historical turns')

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
    message: str = Field(..., description='Customer inquiry text for triage routing', min_length=1)

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

@app.post('/v1/chat', response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """Processes customer inquiry with stateful multi-turn session tracking and reply generation."""
    current_agent = get_agent()
    session_id = payload.session_id
    history = payload.history
    
    if session_id and not history:
        history = _get_session_history(session_id)

    result = current_agent.process(
        customer_text=payload.message,
        conversation_history=history,
        session_id=session_id
    )

    if session_id:
        _append_session_turn(session_id, {
            'customer': payload.message,
            'agent_reply': result['reply'],
            'intent': result['intent'],
            'escalation_category': result['escalation_category']
        })

    return ChatResponse(**result)

@app.post('/v1/triage', response_model=TriageResponse)
def triage_endpoint(payload: TriageRequest) -> TriageResponse:
    """Triages customer inquiry into intent classification and deterministic escalation action."""
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
