# GenAI Health App

This repository contains the GenAI Health App, a project designed to analyze health datasets and provide insights using machine learning and data visualization techniques.

## Project Structure

```
.
├── .env
├── .gitignore
├── app/
│   ├── __init__.py
│   ├── chatgpt_handler.py
│   ├── execute_code.py
│   ├── main.py
├── data/
│   ├── cleaned_activity.csv
│   ├── cleaned_patients.csv
│   ├── Health Dataset 1 (N=2000).csv
│   ├── Health Dataset 2 (N=20000).csv
├── data_analyse.html
├── data_analyse.ipynb
├── frontend/
│   ├── index.html
├── README.md
├── requirements.txt
├── tests/
│   ├── test_chatgpt_handler.py
│   ├── test_execute_code.py
│   ├── test_main.py
├── venv/
```

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Krl1/GenAI-health-app.git
    cd GenAI-health-app
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Start the FastAPI server:
    ```sh
    uvicorn app.main:app --reload
    ```

2. Open your browser and navigate to `http://127.0.0.1:8000` to access the application.

## Notebooks

- `data_analyse.ipynb`: Jupyter notebook for analyzing health datasets using pandas, matplotlib, and seaborn.

## Tests

Run the tests using pytest:
```sh
python -m pytest
```