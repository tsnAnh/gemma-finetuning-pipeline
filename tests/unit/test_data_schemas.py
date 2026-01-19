"""Tests for data validation schemas."""

from gemma_finetune.data.schemas import (
    validate_alpaca,
    validate_chatml,
    validate_sharegpt,
)


class TestShareGPTValidation:
    """Tests for ShareGPT format validation."""

    def test_valid_sharegpt(self):
        """Valid ShareGPT data should pass."""
        data = [
            {
                "conversations": [
                    {"from": "human", "value": "Hi"},
                    {"from": "gpt", "value": "Hello!"},
                ]
            }
        ]
        valid, errors = validate_sharegpt(data)
        assert valid
        assert len(errors) == 0

    def test_missing_conversations_key(self):
        """Missing conversations key should fail."""
        data = [{"text": "some text"}]
        valid, errors = validate_sharegpt(data)
        assert not valid
        assert "missing 'conversations'" in errors[0]

    def test_empty_conversations(self):
        """Empty conversations should fail."""
        data = [{"conversations": []}]
        valid, errors = validate_sharegpt(data)
        assert not valid

    def test_single_turn(self):
        """Single turn should fail (needs at least 2)."""
        data = [{"conversations": [{"from": "human", "value": "Hi"}]}]
        valid, errors = validate_sharegpt(data)
        assert not valid
        assert "at least 2" in errors[0]

    def test_missing_from_field(self):
        """Missing 'from' field should fail."""
        data = [{"conversations": [{"value": "Hi"}, {"from": "gpt", "value": "Hello"}]}]
        valid, errors = validate_sharegpt(data)
        assert not valid
        assert "missing 'from'" in errors[0]

    def test_missing_value_field(self):
        """Missing 'value' field should fail."""
        data = [{"conversations": [{"from": "human"}, {"from": "gpt", "value": "Hello"}]}]
        valid, errors = validate_sharegpt(data)
        assert not valid
        assert "missing 'value'" in errors[0]

    def test_empty_value(self):
        """Empty value should fail."""
        data = [{"conversations": [{"from": "human", "value": ""}, {"from": "gpt", "value": "Hello"}]}]
        valid, errors = validate_sharegpt(data)
        assert not valid
        assert "cannot be empty" in errors[0]

    def test_invalid_role(self):
        """Invalid role should fail."""
        data = [{"conversations": [{"from": "invalid", "value": "Hi"}, {"from": "gpt", "value": "Hello"}]}]
        valid, errors = validate_sharegpt(data)
        assert not valid
        assert "invalid role" in errors[0]

    def test_valid_alternate_roles(self):
        """user/assistant roles should be accepted."""
        data = [
            {
                "conversations": [
                    {"from": "user", "value": "Hi"},
                    {"from": "assistant", "value": "Hello!"},
                ]
            }
        ]
        valid, errors = validate_sharegpt(data)
        assert valid


class TestAlpacaValidation:
    """Tests for Alpaca format validation."""

    def test_valid_alpaca(self):
        """Valid Alpaca data should pass."""
        data = [{"instruction": "Test", "input": "", "output": "Result"}]
        valid, errors = validate_alpaca(data)
        assert valid

    def test_missing_instruction(self):
        """Missing instruction should fail."""
        data = [{"input": "", "output": "Result"}]
        valid, errors = validate_alpaca(data)
        assert not valid
        assert "missing 'instruction'" in errors[0]

    def test_missing_output(self):
        """Missing output should fail."""
        data = [{"instruction": "Test", "input": ""}]
        valid, errors = validate_alpaca(data)
        assert not valid
        assert "missing 'output'" in errors[0]


class TestChatMLValidation:
    """Tests for ChatML format validation."""

    def test_valid_chatml(self):
        """Valid ChatML data should pass."""
        data = [
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                ]
            }
        ]
        valid, errors = validate_chatml(data)
        assert valid

    def test_missing_messages(self):
        """Missing messages key should fail."""
        data = [{"text": "test"}]
        valid, errors = validate_chatml(data)
        assert not valid
        assert "missing 'messages'" in errors[0]

    def test_invalid_role(self):
        """Invalid role should fail."""
        data = [
            {
                "messages": [
                    {"role": "invalid", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ]
            }
        ]
        valid, errors = validate_chatml(data)
        assert not valid
        assert "invalid role" in errors[0]
