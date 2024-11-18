import pandas as pd


def execute_code(
    python_code: str, data_a: pd.DataFrame, data_b: pd.DataFrame
) -> pd.DataFrame:
    """
    Execute dynamically generated Python code to filter datasets.

    Args:
        python_code (str): The Python code to execute.
        data_a (pd.DataFrame): The first dataset.
        data_b (pd.DataFrame): The second dataset.

    Returns:
        pd.DataFrame: The resulting filtered data.

    Raises:
        ValueError: If the code does not produce a 'filtered_data' variable.
        RuntimeError: If an error occurs during code execution.
    """
    # Prepare the execution context
    local_context = {
        "data_a": data_a,
        "data_b": data_b,
        "pd": pd,
    }

    try:
        # Execute the provided Python code
        exec(python_code, {}, local_context)

        # Check for the required 'filtered_data' variable
        if "filtered_data" not in local_context:
            raise ValueError(
                "The executed code did not produce a 'filtered_data' variable."
            )

        return local_context["filtered_data"]

    except ValueError:
        # Allow ValueError to propagate as is
        raise

    except Exception as error:
        # Provide a clear error message for debugging
        raise RuntimeError(
            f"An error occurred while executing the Python code: {error}"
        )
