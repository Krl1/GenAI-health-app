import os
import openai
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_python_code(
    query: str, data_columns_a: list[str], data_columns_b: list[str]
) -> str:
    """
    Generate Python code to filter datasets based on a query using OpenAI's ChatCompletion API.

    Args:
        query (str): The query to filter the datasets.
        data_columns_a (list): List of columns in Dataset A.
        data_columns_b (list): List of columns in Dataset B.

    Returns:
        str: Generated Python code as a string or None if an error occurs.
    """
    system_message = "You are a helpful assistant that writes Python scripts for data analysis tasks."
    user_message = (
        f"I have two datasets with the following columns:\n"
        f"Dataset A: {data_columns_a}\n"
        f"Dataset B: {data_columns_b}\n"
        f'Write a Python script to filter these datasets based on this query: "{query}".\n'
        f"The datasets are named 'data_a' and 'data_b' and are linked by a column called 'Patient_Number'.\n"
        f"Don't print the result, just store it in a variable called 'filtered_data'."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            max_tokens=300,
            temperature=0,  # Deterministic response
        )

        # Extract the generated Python code
        response_content = response["choices"][0]["message"]["content"]
        python_code = extract_code_from_response(response_content)
        return python_code

    except openai.error.OpenAIError as error:
        print(f"OpenAI API Error: {error}")
        return None


def extract_code_from_response(response_content: str) -> str:
    if "```python" in response_content:
        return response_content.split("```python")[1].split("```")[0].strip()
    return response_content.strip()


def interpret_data(query: str, filtered_data: pd.DataFrame) -> str:
    """
    Interpret filtered data and provide a response based on the user query using OpenAI's ChatCompletion API.
    """
    system_message = "You are a helpful assistant that analyzes data and answers questions based on it."
    user_message = f"""
    Given the following dataset: {filtered_data.to_dict()},
    answer this user query: "{query}".
    Provide a concise and relevant response based on the context of the data.
    If you have generated a recommendation, add information about the need to consult it with a doctor before implementing it.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            max_tokens=200,
            temperature=0.7,  # Balance determinism and creativity
        )
        return response["choices"][0]["message"]["content"]
    except openai.error.OpenAIError as e:
        print(f"OpenAI API Error: {e}")
        return None
