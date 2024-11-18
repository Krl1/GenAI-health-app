import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import pandas as pd

from app.main import app


# Initialize the FastAPI test client
client = TestClient(app)


@pytest.fixture
def mock_data():
    """Fixture to provide mock datasets."""
    patients_mock = pd.DataFrame({"Patient_Number": [1, 2], "Age": [25, 40]})
    activity_mock = pd.DataFrame(
        {"Patient_Number": [1, 2], "Activity": ["Running", "Walking"]}
    )
    return patients_mock, activity_mock


@patch("app.main.patients_data", create=True)
@patch("app.main.activity_data", create=True)
def test_serve_index(mock_patients, mock_activity, tmp_path):
    """Test serving the index.html file."""
    index_file = tmp_path / "index.html"
    index_file.write_text("<html><body>Test Index</body></html>")

    # Patch the path to point to the temporary directory
    with patch("app.main.Path", return_value=tmp_path):
        response = client.get("/")
        assert response.status_code == 200
        assert response.text == "<html><body>Test Index</body></html>"


@patch("app.main.patients_data")
@patch("app.main.activity_data")
@patch("app.main.generate_python_code")
@patch("app.main.execute_code")
@patch("app.main.interpret_data")
def test_process_query(
    mock_interpret, mock_execute, mock_generate, mock_activity, mock_patients
):
    """Test the /query endpoint."""
    # Mock the datasets
    mock_patients.columns = ["Patient_Number", "Age"]
    mock_activity.columns = ["Patient_Number", "Activity"]

    # Mock the external function calls
    mock_generate.return_value = "filtered_data = data_a[data_a['Age'] > 30]"
    mock_execute.return_value = pd.DataFrame({"Patient_Number": [2], "Age": [40]})
    mock_interpret.return_value = "Patient 2 is 40 years old."

    # Test payload
    payload = {"user_query": "Get patients older than 30"}
    response = client.post("/query", json=payload)

    # Assertions
    assert response.status_code == 200
    assert response.json() == {"response": "Patient 2 is 40 years old."}

    # Verify the mocked calls
    mock_generate.assert_called_once_with(
        "Get patients older than 30",
        ["Patient_Number", "Age"],
        ["Patient_Number", "Activity"],
    )
    mock_execute.assert_called_once()
    mock_interpret.assert_called_once()


def test_process_query_error_handling():
    """Test the /query endpoint with a failure in processing."""
    with patch("app.main.generate_python_code", side_effect=Exception("Mocked Error")):
        payload = {"user_query": "Invalid query"}
        response = client.post("/query", json=payload)

        assert response.status_code == 500
        assert response.json() == {"detail": "Error processing query: Mocked Error"}
