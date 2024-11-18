import pytest
import pandas as pd
from unittest.mock import patch
from openai.error import OpenAIError
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score

from app.chatgpt_handler import (
    generate_python_code,
    extract_code_from_response,
    interpret_data,
)


@pytest.fixture
def sample_filtered_data():
    """Fixture to provide a sample pandas DataFrame."""
    return pd.DataFrame(
        {
            "Patient_Number": [1, 2, 3],
            "Age": [25, 30, 35],
            "Condition": ["Healthy", "Sick", "Healthy"],
        }
    )


@patch("app.chatgpt_handler.openai.ChatCompletion.create")
def test_generate_python_code(mock_openai):
    """Test the generate_python_code function with mocked OpenAI API response."""
    mock_openai.return_value = {
        "choices": [
            {
                "message": {
                    "content": "```python\nfiltered_data = data_a[data_a['Age'] > 30]\n```"
                }
            }
        ]
    }

    query = "Filter Dataset A where Age is greater than 30"
    data_columns_a = ["Patient_Number", "Age", "Condition"]
    data_columns_b = ["Patient_Number", "Blood_Type"]

    result = generate_python_code(query, data_columns_a, data_columns_b)
    expected_code = "filtered_data = data_a[data_a['Age'] > 30]"

    assert result == expected_code
    mock_openai.assert_called_once()


def test_extract_code_from_response():
    """Test the extract_code_from_response function."""
    response_content = "```python\nfiltered_data = data_a[data_a['Age'] > 30]\n```"
    result = extract_code_from_response(response_content)
    expected_code = "filtered_data = data_a[data_a['Age'] > 30]"

    assert result == expected_code


@patch("app.chatgpt_handler.openai.ChatCompletion.create")
def test_interpret_data(mock_openai, sample_filtered_data):
    """Test the interpret_data function with mocked OpenAI API response."""
    mock_openai.return_value = {
        "choices": [
            {"message": {"content": "The average age of the filtered data is 30."}}
        ]
    }

    query = "What is the average age in the dataset?"
    result = interpret_data(query, sample_filtered_data)
    expected_response = "The average age of the filtered data is 30."

    assert result == expected_response
    mock_openai.assert_called_once()


@patch("app.chatgpt_handler.openai.ChatCompletion.create")
def test_generate_python_code_error_handling(mock_openai):
    """Test generate_python_code error handling with a mocked API error."""
    mock_openai.side_effect = OpenAIError("Mocked API Error")

    query = "Filter Dataset A where Age is greater than 30"
    data_columns_a = ["Patient_Number", "Age", "Condition"]
    data_columns_b = ["Patient_Number", "Blood_Type"]

    # The function should handle the error and return None
    result = generate_python_code(query, data_columns_a, data_columns_b)

    assert result is None


@patch("app.chatgpt_handler.openai.ChatCompletion.create")
def test_interpret_data_error_handling(mock_openai, sample_filtered_data):
    """Test interpret_data error handling with a mocked API error."""
    mock_openai.side_effect = OpenAIError("Mocked API Error")

    query = "What is the average age in the dataset?"
    result = interpret_data(query, sample_filtered_data)

    assert result is None


@pytest.fixture
def sample_data():
    """Fixture to provide sample data."""
    data_a_columns = ["Patient_Number", "Age", "Condition"]
    data_b_columns = ["Patient_Number", "Blood_Type"]
    query = "Filter Dataset A where Age is greater than 30"
    return query, data_a_columns, data_b_columns


@pytest.fixture
def expected_python_code():
    """Fixture to provide the expected Python code output."""
    return """filtered_data = data_a[data_a['Age'] > 30]"""


@patch("app.chatgpt_handler.openai.ChatCompletion.create")
def test_generate_python_code_metrics(mock_openai, sample_data, expected_python_code):
    """Test generate_python_code using BLEU, ROUGE, and BERTScore metrics."""
    query, data_columns_a, data_columns_b = sample_data

    # Mocked response from OpenAI API
    mock_response_content = f"""```python
filtered_data = data_a[data_a['Age'] > 30]
```"""
    mock_openai.return_value = {
        "choices": [{"message": {"content": mock_response_content}}]
    }

    # Call the function
    generated_code = generate_python_code(query, data_columns_a, data_columns_b)

    # Compute BLEU score
    reference = [expected_python_code.split()]
    candidate = generated_code.split()
    bleu_score = sentence_bleu(reference, candidate)

    # Compute ROUGE score
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge_scores = scorer.score(expected_python_code, generated_code)

    # Compute BERTScore
    P, R, F1 = bert_score(
        [generated_code], [expected_python_code], lang="en", verbose=False
    )
    bert_f1 = F1.item()

    # Assertions
    assert bleu_score == 1.0, f"BLEU score is lower than expected: {bleu_score}"
    assert (
        rouge_scores["rouge1"].fmeasure == 1.0
    ), f"ROUGE-1 F1 is lower than expected: {rouge_scores['rouge1'].fmeasure}"
    assert (
        rouge_scores["rougeL"].fmeasure == 1.0
    ), f"ROUGE-L F1 is lower than expected: {rouge_scores['rougeL'].fmeasure}"
    assert bert_f1 > 0.99, f"BERTScore F1 is lower than expected: {bert_f1}"


@pytest.fixture
def sample_filtered_data():
    """Fixture to provide a sample pandas DataFrame."""
    return pd.DataFrame({"Patient_Number": [2], "Age": [40], "Condition": ["Healthy"]})


@pytest.fixture
def expected_interpretation():
    """Fixture to provide the expected interpretation output."""
    return (
        "The dataset contains 1 patient older than 30 years old. "
        "Patient_Number 2 is 40 years old and is Healthy. "
        "Please consult a doctor before making any medical decisions."
    )


@patch("app.chatgpt_handler.openai.ChatCompletion.create")
def test_interpret_data_metrics(
    mock_openai, sample_filtered_data, expected_interpretation
):
    """Test interpret_data using BLEU, ROUGE, and BERTScore metrics."""
    query = "Provide details about patients older than 30"
    filtered_data = sample_filtered_data

    # Mocked response from OpenAI API
    mock_response_content = (
        "The dataset contains 1 patient older than 30 years old. "
        "Patient_Number 2 is 40 years old and is Healthy. "
        "Please consult a doctor before making any medical decisions."
    )
    mock_openai.return_value = {
        "choices": [{"message": {"content": mock_response_content}}]
    }

    # Call the function
    generated_interpretation = interpret_data(query, filtered_data)

    # Compute BLEU score
    reference = [expected_interpretation.split()]
    candidate = generated_interpretation.split()
    bleu_score = sentence_bleu(reference, candidate)

    # Compute ROUGE score
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge_scores = scorer.score(expected_interpretation, generated_interpretation)

    # Compute BERTScore
    P, R, F1 = bert_score(
        [generated_interpretation], [expected_interpretation], lang="en", verbose=False
    )
    bert_f1 = F1.item()

    # Assertions
    assert bleu_score > 0.9, f"BLEU score is lower than expected: {bleu_score}"
    assert (
        rouge_scores["rouge1"].fmeasure > 0.9
    ), f"ROUGE-1 F1 is lower than expected: {rouge_scores['rouge1'].fmeasure}"
    assert (
        rouge_scores["rougeL"].fmeasure > 0.9
    ), f"ROUGE-L F1 is lower than expected: {rouge_scores['rougeL'].fmeasure}"
    assert bert_f1 > 0.9, f"BERTScore F1 is lower than expected: {bert_f1}"
