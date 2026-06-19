from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class Message(BaseModel):
    role: MessageRole
    content: str


class ChatRequest(BaseModel):
    query: str = Field(..., description="User question")
    session_id: str = Field(default="default", description="Session ID for multi-turn conversation")
    user_id: Optional[str] = Field(default=None, description="User ID for cross-session long-term memory")
    top_k: Optional[int] = Field(default=None, description="Number of documents to retrieve")
    enable_rewrite: bool = Field(default=True, description="Whether to enable query rewriting")
    stream: bool = Field(default=False, description="Whether to stream the response")

    @property
    def effective_user_id(self) -> str:
        """Falls back to session_id when user_id is not provided (session-level memory, backwards-compatible)."""
        return self.user_id or self.session_id


class SourceChunk(BaseModel):
    content: str
    source: str
    page: int
    score: float
    chunk_id: str


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    original_query: str
    rewritten_query: Optional[str] = None
    sources: List[SourceChunk] = []
    eval_scores: Optional[Dict[str, float]] = None
    latency_ms: Optional[float] = None


class UploadResponse(BaseModel):
    filename: str
    doc_id: str
    chunks_count: int
    message: str


class EvalRequest(BaseModel):
    query: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None


class EvalResponse(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_recall: Optional[float] = None
    context_precision: float
    overall_score: float


class SessionHistory(BaseModel):
    session_id: str
    messages: List[Message]
    total_turns: int


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    chunks_count: int
    upload_time: str


class ListDocumentsResponse(BaseModel):
    documents: List[DocumentInfo]
    total: int
