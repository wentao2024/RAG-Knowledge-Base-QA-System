"""
多轮对话会话管理器
- 内存存储（可替换为 Redis)
- 会话隔离、历史截断
"""
import time
from typing import Dict, List, Optional
from collections import OrderedDict
from app.models.schemas import Message, MessageRole


class SessionManager:
    """LRU 会话管理器"""

    def __init__(self, max_sessions: int = 1000, max_history: int = 20):
        self.max_sessions = max_sessions
        self.max_history = max_history
        # {session_id: {"messages": [...], "created_at": ts, "updated_at": ts}}
        self._sessions: OrderedDict = OrderedDict()

    def _evict_if_needed(self):
        while len(self._sessions) >= self.max_sessions:
            self._sessions.popitem(last=False)  # 移除最旧的

    def get_history(self, session_id: str) -> List[Message]:
        """获取会话历史"""
        session = self._sessions.get(session_id)
        if not session:
            return []
        return session["messages"]

    def add_turn(self, session_id: str, user_msg: str, assistant_msg: str):
        """添加一轮对话"""
        if session_id not in self._sessions:
            self._evict_if_needed()
            self._sessions[session_id] = {
                "messages": [],
                "created_at": time.time(),
                "updated_at": time.time(),
            }
        session = self._sessions[session_id]
        session["messages"].append(
            Message(role=MessageRole.user, content=user_msg)
        )
        session["messages"].append(
            Message(role=MessageRole.assistant, content=assistant_msg)
        )
        session["updated_at"] = time.time()

        # 截断历史，保留最近 N 条
        if len(session["messages"]) > self.max_history * 2:
            session["messages"] = session["messages"][-(self.max_history * 2):]

        # 移到末尾（LRU）
        self._sessions.move_to_end(session_id)

    def clear_session(self, session_id: str):
        self._sessions.pop(session_id, None)

    def get_session_info(self, session_id: str) -> Optional[dict]:
        return self._sessions.get(session_id)

    def list_sessions(self) -> List[str]:
        return list(self._sessions.keys())

    def total_turns(self, session_id: str) -> int:
        session = self._sessions.get(session_id)
        if not session:
            return 0
        return len(session["messages"]) // 2


# 全局单例
session_manager = SessionManager()
