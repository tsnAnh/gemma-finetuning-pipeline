"""Tests for Gemma chat formatting utilities."""

from gemma_finetune.config import GEMMA_END, GEMMA_START
from gemma_finetune.data.formatter import (
    embed_system_prompt,
    format_gemma_chat,
    format_training_example,
    parse_gemma_response,
)


class TestFormatGemmaChat:
    """Tests for format_gemma_chat function."""

    def test_basic_conversation(self):
        """Format basic two-turn conversation."""
        convs = [
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi there!"},
        ]
        result = format_gemma_chat(convs)
        assert f"{GEMMA_START}user\nHello{GEMMA_END}" in result
        assert f"{GEMMA_START}model\nHi there!{GEMMA_END}" in result

    def test_role_mapping_human_gpt(self):
        """Map human->user and gpt->model."""
        convs = [
            {"from": "human", "value": "Q"},
            {"from": "gpt", "value": "A"},
        ]
        result = format_gemma_chat(convs)
        assert "user" in result
        assert "model" in result
        assert "human" not in result
        assert "gpt" not in result

    def test_role_mapping_user_assistant(self):
        """Map user and assistant roles correctly."""
        convs = [
            {"from": "user", "value": "Q"},
            {"from": "assistant", "value": "A"},
        ]
        result = format_gemma_chat(convs)
        assert "user" in result
        assert "model" in result
        assert "assistant" not in result

    def test_add_generation_prompt(self):
        """Should add model turn marker for inference."""
        convs = [{"from": "human", "value": "Hello"}]
        result = format_gemma_chat(convs, add_generation_prompt=True)
        assert result.endswith(f"{GEMMA_START}model\n")

    def test_no_generation_prompt_default(self):
        """Should not add generation prompt by default."""
        convs = [
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi!"},
        ]
        result = format_gemma_chat(convs)
        assert not result.endswith(f"{GEMMA_START}model\n")

    def test_multi_turn_conversation(self):
        """Format multi-turn conversation correctly."""
        convs = [
            {"from": "human", "value": "First"},
            {"from": "gpt", "value": "Reply1"},
            {"from": "human", "value": "Second"},
            {"from": "gpt", "value": "Reply2"},
        ]
        result = format_gemma_chat(convs)
        # All turns should be present
        assert "First" in result
        assert "Reply1" in result
        assert "Second" in result
        assert "Reply2" in result


class TestEmbedSystemPrompt:
    """Tests for embed_system_prompt function."""

    def test_embed_in_first_human_turn(self):
        """System prompt should be embedded in first human turn."""
        convs = [
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi!"},
        ]
        result = embed_system_prompt(convs, "You are helpful.")
        assert "You are helpful." in result[0]["value"]
        assert "Hello" in result[0]["value"]

    def test_empty_conversations(self):
        """Empty conversations should create human turn with system prompt."""
        result = embed_system_prompt([], "System message")
        assert len(result) == 1
        assert result[0]["from"] == "human"
        assert result[0]["value"] == "System message"

    def test_no_human_turn(self):
        """If no human turn, insert system as first message."""
        convs = [{"from": "gpt", "value": "Hello!"}]
        result = embed_system_prompt(convs, "System")
        assert result[0]["from"] == "human"
        assert result[0]["value"] == "System"

    def test_preserves_other_turns(self):
        """Other turns should be preserved."""
        convs = [
            {"from": "human", "value": "Q"},
            {"from": "gpt", "value": "A"},
        ]
        result = embed_system_prompt(convs, "Sys")
        assert len(result) == 2
        assert result[1]["value"] == "A"

    def test_deep_copy(self):
        """Original conversations should not be modified."""
        convs = [{"from": "human", "value": "Original"}]
        embed_system_prompt(convs, "System")
        assert convs[0]["value"] == "Original"


class TestParseGemmaResponse:
    """Tests for parse_gemma_response function."""

    def test_extract_model_response(self):
        """Extract response from formatted output."""
        output = f"{GEMMA_START}user\nHello{GEMMA_END}\n{GEMMA_START}model\nHi there!{GEMMA_END}"
        result = parse_gemma_response(output)
        assert result == "Hi there!"

    def test_last_model_turn(self):
        """Should extract last model turn in multi-turn output."""
        output = (
            f"{GEMMA_START}user\nQ1{GEMMA_END}\n"
            f"{GEMMA_START}model\nA1{GEMMA_END}\n"
            f"{GEMMA_START}user\nQ2{GEMMA_END}\n"
            f"{GEMMA_START}model\nA2{GEMMA_END}"
        )
        result = parse_gemma_response(output)
        assert result == "A2"

    def test_no_formatting(self):
        """Return as-is if no formatting found."""
        output = "Plain text response"
        result = parse_gemma_response(output)
        assert result == "Plain text response"

    def test_strips_whitespace(self):
        """Should strip whitespace from response."""
        output = f"{GEMMA_START}model\n  Response with spaces  {GEMMA_END}"
        result = parse_gemma_response(output)
        assert result == "Response with spaces"


class TestFormatTrainingExample:
    """Tests for format_training_example function."""

    def test_basic_formatting(self):
        """Format conversation as training example."""
        convs = [
            {"from": "human", "value": "Hi"},
            {"from": "gpt", "value": "Hello!"},
        ]
        result = format_training_example(convs)
        assert "text" in result
        assert "user" in result["text"]
        assert "model" in result["text"]

    def test_with_system_prompt(self):
        """Format with system prompt embedded."""
        convs = [
            {"from": "human", "value": "Hi"},
            {"from": "gpt", "value": "Hello!"},
        ]
        result = format_training_example(convs, system_prompt="Be helpful")
        assert "Be helpful" in result["text"]

    def test_without_system_prompt(self):
        """Format without system prompt."""
        convs = [
            {"from": "human", "value": "Hi"},
            {"from": "gpt", "value": "Hello!"},
        ]
        result = format_training_example(convs, system_prompt=None)
        assert "text" in result
