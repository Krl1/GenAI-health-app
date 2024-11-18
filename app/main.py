import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd

from app.chatgpt_handler import generate_python_code, interpret_data
from app.execute_code import execute_code


# Initialize the FastAPI app
app = FastAPI()

# Load datasets
DATA_DIR = Path("data")
patients_data = pd.read_csv(DATA_DIR / "cleaned_patients.csv")
activity_data = pd.read_csv(DATA_DIR / "cleaned_activity.csv")


class Query(BaseModel):
    user_query: str


@app.get("/", response_class=HTMLResponse)
def serve_index():
    """
    Serve the main HTML index page for the application.
    """
    index_path = Path("frontend") / "index.html"

    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Index file not found.")

    try:
        with open(index_path, "r") as file:
            html_content = file.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading index file: {e}")


@app.post("/query")
async def process_query(query: Query):
    """
    Process the user query by generating Python code, executing it,
    and interpreting the results.

    Args:
        query (Query): The user query payload.

    Returns:
        dict: A response containing the interpreted results.
    """
    try:
        # Generate Python code
        python_code = generate_python_code(
            query.user_query, list(patients_data.columns), list(activity_data.columns)
        )

        # Execute the generated code
        filtered_data = execute_code(python_code, patients_data, activity_data)

        # Interpret the results
        response = interpret_data(query.user_query, filtered_data)

        return {"response": response}

    except Exception as e:
        # Handle any unexpected errors and provide a meaningful response
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")
