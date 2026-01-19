"""Tests for dataset format conversion."""

import pytest

from gemma_finetune.data.converter import (
    alpaca_to_sharegpt,
    chatml_to_sharegpt,
    normalize_roles,
    normalize_to_sharegpt,
)


class TestAlpacaToShareGPT:
    """Tests for Alpaca to ShareGPT conversion."""

    def test_basic_conversion(self):
        """Convert basic Alpaca format."""
        alpaca = {"instruction": "Say hello", "input": "", "output": "Hello!"}
        result = alpaca_to_sharegpt(alpaca)
        assert "conversations" in result
        assert len(result["conversations"]) == 2
        assert result["conversations"][0]["from"] == "human"
        assert result["conversations"][1]["from"] == "gpt"

    def test_with_input(self):
        """Input should be appended to instruction."""
        alpaca = {"instruction": "Add numbers", "input": "2+2", "output": "4"}
        result = alpaca_to_sharegpt(alpaca)
        human_msg = result["conversations"][0]["value"]
        assert "Add numbers" in human_msg
        assert "2+2" in human_msg

    def test_output_preserved(self):
        """Output should be preserved as gpt response."""
        alpaca = {"instruction": "Test", "input": "", "output": "Response here"}
        result = alpaca_to_sharegpt(alpaca)
        assert result["conversations"][1]["value"] == "Response here"


class TestChatMLToShareGPT:
    """Tests for ChatML to ShareGPT conversion."""

    def test_user_assistant_conversion(self):
        """Convert user/assistant to human/gpt."""
        chatml = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        }
        result = chatml_to_sharegpt(chatml)
        assert result["conversations"][0]["from"] == "human"
        assert result["conversations"][0]["value"] == "Hello"
        assert result["conversations"][1]["from"] == "gpt"
        assert result["conversations"][1]["value"] == "Hi!"

    def test_system_message_embedded(self):
        """System message should be embedded in first user turn."""
        chatml = {
            "messages": [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ]
        }
        result = chatml_to_sharegpt(chatml)
        # System should be embedded, not a separate turn
        first_human = result["conversations"][0]["value"]
        assert "Be helpful" in first_human
        assert "Question" in first_human

    def test_multi_turn(self):
        """Multi-turn conversation should be preserved."""
        chatml = {
            "messages": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
            ]
        }
        result = chatml_to_sharegpt(chatml)
        assert len(result["conversations"]) == 4


class TestNormalizeToShareGPT:
    """Tests for normalize_to_sharegpt function."""

    def test_normalize_alpaca(self):
        """Normalize Alpaca to ShareGPT."""
        item = {"instruction": "Test", "input": "", "output": "Out"}
        result = normalize_to_sharegpt(item, source_format="alpaca")
        assert "conversations" in result

    def test_normalize_chatml(self):
        """Normalize ChatML to ShareGPT."""
        item = {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}
        result = normalize_to_sharegpt(item, source_format="chatml")
        assert "conversations" in result

    def test_sharegpt_passthrough(self):
        """ShareGPT format should pass through unchanged."""
        item = {"conversations": [{"from": "human", "value": "Test"}, {"from": "gpt", "value": "OK"}]}
        result = normalize_to_sharegpt(item, source_format="sharegpt")
        assert result == item

    def test_unknown_format(self):
        """Unknown format should raise error."""
        with pytest.raises(ValueError, match="Unsupported format"):
            normalize_to_sharegpt({}, source_format="invalid")


class TestNormalizeRoles:
    """Tests for normalize_roles function."""

    def test_user_to_human(self):
        """user role should map to human."""
        convs = [{"from": "user", "value": "Hi"}]
        result = normalize_roles(convs)
        assert result[0]["from"] == "human"

    def test_assistant_to_gpt(self):
        """assistant role should map to gpt."""
        convs = [{"from": "assistant", "value": "Hello"}]
        result = normalize_roles(convs)
        assert result[0]["from"] == "gpt"

    def test_model_to_gpt(self):
        """model role should map to gpt."""
        convs = [{"from": "model", "value": "Response"}]
        result = normalize_roles(convs)
        assert result[0]["from"] == "gpt"

    def test_preserves_human_gpt(self):
        """human/gpt roles should remain unchanged."""
        convs = [
            {"from": "human", "value": "Q"},
            {"from": "gpt", "value": "A"},
        ]
        result = normalize_roles(convs)
        assert result[0]["from"] == "human"
        assert result[1]["from"] == "gpt"
