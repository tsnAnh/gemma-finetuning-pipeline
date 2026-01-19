"""Chat session management for multi-turn conversations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from ..data.formatter import embed_system_prompt, format_gemma_chat


class InferenceEngine(Protocol):
    """Protocol for inference engines (Keras or TFLite)."""

    def generate(self, prompt: str, max_new_tokens: int = 256, **kwargs: Any) -> str: ...

    def generate_stream(
        self, prompt: str, callback: Callable[[str], None], max_new_tokens: int = 256, **kwargs: Any
    ) -> str: ...


@dataclass
class ChatMessage:
    """A single chat message."""

    role: str  # "user" or "assistant"
    content: str


class ChatSession:
    """Multi-turn chat session with conversation history."""

    def __init__(
        self,
        inference_engine: InferenceEngine,
        system_prompt: str | None = None,
        max_history: int = 10,
    ):
        """Initialize chat session.

        Args:
            inference_engine: Keras or TFLite inference engine.
            system_prompt: Optional system instructions.
            max_history: Maximum conversation turns to keep.
        """
        self.engine = inference_engine
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.history: list[ChatMessage] = []

    def chat(
        self,
        user_message: str,
        stream_callback: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> str:
        """Send message and get response.

        Args:
            user_message: User's message.
            stream_callback: Optional callback for streaming output.
            **kwargs: Additional generation parameters.

        Returns:
            Assistant's response.
        """
        # Add user message to history
        self.history.append(ChatMessage("user", user_message))

        # Build conversation for model
        conversations = self._build_conversations()

        # Format for Gemma
        formatted = format_gemma_chat(conversations, add_generation_prompt=True)

        # Generate response
        if stream_callback:
            response = self.engine.generate_stream(formatted, stream_callback, **kwargs)
        else:
            response = self.engine.generate(formatted, **kwargs)

        # Clean up response
        response = self._clean_response(response)

        # Add assistant response to history
        self.history.append(ChatMessage("assistant", response))

        return response

    def _build_conversations(self) -> list[dict[str, str]]:
        """Build conversation list for model."""
        conversations: list[dict[str, str]] = []

        # Get recent history
        recent = self.history[-self.max_history :]

        for msg in recent:
            role = "human" if msg.role == "user" else "gpt"
            conversations.append({"from": role, "value": msg.content})

        # Embed system prompt if present and first message
        if self.system_prompt and len(self.history) == 1:
            conversations = embed_system_prompt(conversations, self.system_prompt)

        return conversations

    def _clean_response(self, response: str) -> str:
        """Clean up model response."""
        # Remove any leftover formatting tokens
        response = response.replace("<end_of_turn>", "")
        response = response.replace("<start_of_turn>model", "")
        response = response.replace("<start_of_turn>user", "")
        return response.strip()

    def clear(self) -> None:
        """Clear conversation history."""
        self.history = []

    def get_history(self) -> list[dict[str, str]]:
        """Get history as list of dicts."""
        return [{"role": m.role, "content": m.content} for m in self.history]

    def set_system_prompt(self, prompt: str | None) -> None:
        """Update system prompt.

        Args:
            prompt: New system prompt or None to clear.
        """
        self.system_prompt = prompt

    def __len__(self) -> int:
        """Return number of messages in history."""
        return len(self.history)
