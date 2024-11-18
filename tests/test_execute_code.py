import pytest
import pandas as pd
from app.execute_code import execute_code


def test_execute_code_success():
    """Test execute_code with valid Python code that produces 'filtered_data'."""
    data_a = pd.DataFrame({"Patient_Number": [1, 2, 3], "Age": [25, 40, 35]})
    data_b = pd.DataFrame({"Patient_Number": [1, 2, 3], "Blood_Type": ["A", "B", "O"]})

    python_code = """
filtered_data = data_a[data_a['Age'] > 30]
"""

    result = execute_code(python_code, data_a, data_b)

    expected_result = pd.DataFrame({"Patient_Number": [2, 3], "Age": [40, 35]})
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected_result)


def test_execute_code_no_filtered_data():
    """Test execute_code with Python code that does not produce 'filtered_data'."""
    data_a = pd.DataFrame({"Patient_Number": [1, 2, 3], "Age": [25, 40, 35]})
    data_b = pd.DataFrame({"Patient_Number": [1, 2, 3], "Blood_Type": ["A", "B", "O"]})

    python_code = """
# This code does not create 'filtered_data'
result = data_a[data_a['Age'] > 30]
"""

    with pytest.raises(
        ValueError,
        match="The executed code did not produce a 'filtered_data' variable.",
    ):
        execute_code(python_code, data_a, data_b)


def test_execute_code_invalid_python_code():
    """Test execute_code with invalid Python code."""
    data_a = pd.DataFrame({"Patient_Number": [1, 2, 3], "Age": [25, 40, 35]})
    data_b = pd.DataFrame({"Patient_Number": [1, 2, 3], "Blood_Type": ["A", "B", "O"]})
