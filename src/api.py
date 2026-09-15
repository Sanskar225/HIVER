"""
Production FastAPI Microservice for @AmazonHelp AI Support Agent.
"""
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

# Global singleton agent instance
agent = AmazonSupportAgent()

# In-memory session store for multi-turn thread tracking
SESSION_STORE: Dict[str, List[Dict[str, str]]] = {}

class ChatRequest(BaseModel):
    message: str = Field(..., description='Inbound customer tweet or message text', min_length=1)
    session_id: Optional[str] = Field(None, description='Optional session/conversation identifier for multi-turn tracking')
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
def health_check():
    return HealthResponse(
        status='healthy',
        brand=TARGET_BRAND,
        model_name=agent.model_name,
        kb_size=len(agent.retriever.df_kb) if agent.retriever and hasattr(agent.retriever, 'df_kb') else 55011,
        session_count=len(SESSION_STORE)
    )

@app.post('/v1/chat', response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest):
    session_id = payload.session_id
    history = payload.history
    
    if session_id and not history:
        history = SESSION_STORE.get(session_id, [])

    result = agent.process(
        customer_text=payload.message,
        conversation_history=history,
        session_id=session_id
    )

    if session_id:
        if session_id not in SESSION_STORE:
            SESSION_STORE[session_id] = []
        SESSION_STORE[session_id].append({
            'customer': payload.message,
            'agent_reply': result['reply'],
            'intent': result['intent'],
            'escalation_category': result['escalation_category']
        })

    return ChatResponse(**result)

@app.post('/v1/triage', response_model=TriageResponse)
def triage_endpoint(payload: TriageRequest):
    cleaned = agent.preprocess(payload.message)
    intent, confidence, _ = agent.classify_intent(cleaned)
    decision, category, reason = agent.triage_decision(cleaned, intent, confidence)
    return TriageResponse(
        intent=intent,
        intent_confidence=confidence,
        decision=decision,
        escalation_category=category,
        reason=reason
    )
