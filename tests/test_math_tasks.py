"""
Test LILAMath and GSM8K tasks without running the LLM.
Example run:

python -m pytest tests/test_math_tasks.py -v
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from tasks.math_custom import LILAMath
from tasks.gsm8k import GSM8K


class TestLILAMath:
    """Test LILAMath task initialization, data loading, and evaluation."""

    def test_initialization(self):
        """Test that LILAMath can be initialized."""
        task = LILAMath(split="test", subset="iid")
        assert task.subset == "iid"
        assert task.eval_type == "generative"
        assert task.num_examples() > 0

    def test_get_example_structure(self):
        """Test that get_example returns the correct structure."""
        task = LILAMath(split="test", subset="iid")
        example = task.get_example(0)
        
        assert "messages" in example
        assert "valid_answers" in example
        assert len(example["messages"]) == 2
        assert example["messages"][0]["role"] == "user"
        assert example["messages"][1]["role"] == "assistant"
        assert isinstance(example["valid_answers"], list)
        assert len(example["valid_answers"]) > 0

    def test_evaluate_exact_match(self):
        """Test evaluation with exact match."""
        task = LILAMath(split="test", subset="iid")
        example = task.get_example(0)
        
        # Use the first valid answer as the response
        correct_answer = example["valid_answers"][0]
        result = task.evaluate(example, correct_answer)
        assert result == 1

    def test_evaluate_numeric_match(self):
        """Test evaluation with numeric comparison."""
        task = LILAMath(split="test", subset="iid")
        
        # Create a mock conversation with numeric answers
        conversation = {
            "messages": [
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "4"}
            ],
            "valid_answers": ["4", "4.0", "4.00"]
        }
        
        # Test exact match
        assert task.evaluate(conversation, "4") == 1
        # Test numeric match with different formatting
        assert task.evaluate(conversation, "4.0") == 1
        assert task.evaluate(conversation, "4.00") == 1
        # Test wrong answer
        assert task.evaluate(conversation, "5") == 0

    def test_evaluate_numeric_float_comparison(self):
        """Test evaluation with floating point comparison."""
        task = LILAMath(split="test", subset="iid")
        
        conversation = {
            "messages": [
                {"role": "user", "content": "What is 1/3?"},
                {"role": "assistant", "content": "0.333333"}
            ],
            "valid_answers": ["0.333333", "0.3333333"]
        }
        
        # Test with small floating point differences
        assert task.evaluate(conversation, "0.333333") == 1
        assert task.evaluate(conversation, "0.3333333") == 1
        # Test with value that's close but not exact
        assert task.evaluate(conversation, "0.333334") == 0  # Different enough

    def test_evaluate_wrong_answer(self):
        """Test evaluation with incorrect answer."""
        task = LILAMath(split="test", subset="iid")
        example = task.get_example(0)
        
        wrong_answer = "This is definitely wrong"
        result = task.evaluate(example, wrong_answer)
        assert result == 0

    def test_evaluate_normalization(self):
        """Test that evaluation handles whitespace normalization."""
        task = LILAMath(split="test", subset="iid")
        
        conversation = {
            "messages": [
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "4"}
            ],
            "valid_answers": ["4"]
        }
        
        # Test with whitespace
        assert task.evaluate(conversation, "  4  ") == 1
        assert task.evaluate(conversation, "\n4\n") == 1

    def test_slicing(self):
        """Test that task slicing works correctly."""
        task = LILAMath(split="test", subset="iid", start=0, stop=5)
        assert len(task) <= 5
        
        if len(task) > 0:
            example = task[0]
            assert "messages" in example


class TestGSM8K:
    """Test GSM8K task initialization, data loading, and evaluation."""

    def test_initialization_main(self):
        """Test that GSM8K can be initialized with main subset."""
        task = GSM8K(subset="main", split="test")
        assert task.eval_type == "generative"
        assert task.num_examples() > 0

    def test_initialization_socratic(self):
        """Test that GSM8K can be initialized with socratic subset."""
        task = GSM8K(subset="socratic", split="test")
        assert task.eval_type == "generative"
        assert task.num_examples() > 0

    def test_initialization_invalid_subset(self):
        """Test that invalid subset raises an error."""
        with pytest.raises(AssertionError):
            GSM8K(subset="invalid", split="test")

    def test_initialization_invalid_split(self):
        """Test that invalid split raises an error."""
        with pytest.raises(AssertionError):
            GSM8K(subset="main", split="invalid")

    def test_get_example_structure(self):
        """Test that get_example returns the correct structure."""
        task = GSM8K(subset="main", split="test")
        example = task.get_example(0)
        
        assert "messages" in example
        assert len(example["messages"]) == 2
        assert example["messages"][0]["role"] == "user"
        assert example["messages"][1]["role"] == "assistant"
        
        # Assistant message should be a list of parts
        assistant_content = example["messages"][1]["content"]
        assert isinstance(assistant_content, list)
        assert len(assistant_content) > 0
        
        # Check that parts have the correct structure
        for part in assistant_content:
            assert "type" in part
            assert "text" in part
            assert part["type"] in ["text", "python", "python_output"]

    def test_get_example_has_tool_calls(self):
        """Test that examples contain tool calls (python parts)."""
        task = GSM8K(subset="main", split="test")
        example = task.get_example(0)
        
        assistant_content = example["messages"][1]["content"]
        has_python = any(part["type"] == "python" for part in assistant_content)
        has_python_output = any(part["type"] == "python_output" for part in assistant_content)
        
        # GSM8K should have tool calls
        assert has_python or has_python_output, "GSM8K examples should contain tool calls"

    def test_evaluate_correct_answer(self):
        """Test evaluation with correct answer."""
        task = GSM8K(subset="main", split="test")
        example = task.get_example(0)
        
        # Extract the ground truth answer from the example
        assistant_message = example["messages"][1]
        last_text_part = assistant_message["content"][-1]["text"]
        
        # The last text part should contain #### answer
        # We'll use the actual answer from the example
        from tasks.gsm8k import extract_answer
        ref_num = extract_answer(last_text_part)
        
        if ref_num:
            # Test with the correct answer
            response = f"Some reasoning here. #### {ref_num}"
            result = task.evaluate(example, response)
            assert result == 1

    def test_evaluate_wrong_answer(self):
        """Test evaluation with incorrect answer."""
        task = GSM8K(subset="main", split="test")
        example = task.get_example(0)
        
        # Use a clearly wrong answer
        wrong_response = "Some reasoning here. #### 999999"
        result = task.evaluate(example, wrong_response)
        assert result == 0

    def test_extract_answer_function(self):
        """Test the extract_answer helper function."""
        from tasks.gsm8k import extract_answer
        
        # Test standard format
        assert extract_answer("The answer is #### 42") == "42"
        assert extract_answer("#### 123") == "123"
        assert extract_answer("#### -5") == "-5"
        assert extract_answer("#### 3.14") == "3.14"
        
        # Test with commas
        assert extract_answer("#### 1,000") == "1000"
        assert extract_answer("#### 1,234.56") == "1234.56"
        
        # Test no match
        assert extract_answer("No answer here") is None
        assert extract_answer("") is None

    def test_evaluate_missing_answer_marker(self):
        """Test evaluation when answer marker is missing."""
        task = GSM8K(subset="main", split="test")
        example = task.get_example(0)
        
        # Response without #### marker
        response = "Some reasoning but no answer marker"
        result = task.evaluate(example, response)
        assert result == 0

    def test_slicing(self):
        """Test that task slicing works correctly."""
        task = GSM8K(subset="main", split="test", start=0, stop=5)
        assert len(task) <= 5
        
        if len(task) > 0:
            example = task[0]
            assert "messages" in example

    def test_reward_function(self):
        """Test that reward function works correctly."""
        task = GSM8K(subset="main", split="test")
        example = task.get_example(0)
        
        # Extract ground truth
        assistant_message = example["messages"][1]
        last_text_part = assistant_message["content"][-1]["text"]
        from tasks.gsm8k import extract_answer
        ref_num = extract_answer(last_text_part)
        
        if ref_num:
            # Test correct answer gives reward 1.0
            correct_response = f"Reasoning. #### {ref_num}"
            reward = task.reward(example, correct_response)
            assert reward == 1.0
            
            # Test wrong answer gives reward 0.0
            wrong_response = "Reasoning. #### 999"
            reward = task.reward(example, wrong_response)
            assert reward == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

