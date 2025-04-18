# Installation Guide

Follow these steps to set up the **Customer Financial Health Analyzer** project locally.

## Prerequisites

- **Python**: 3.12 or higher
- **Git**: For cloning the repository
- **Ollama**: For hosting the LLM model (e.g., `llama3`)
- **uv**: Python package and environment manager
- **No API Keys Required**: Uses open-source libraries for PDF parsing and data processing

## Step 1: Clone the Repository

```bash
git clone https://github.com/Purusharth1/Customer-Financial-Health-Analyzer.git
cd Customer-Financial-Health-Analyzer
```

## Step 2: Install uv

Install the uv package manager if not already installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Step 3: Install Dependencies

Set up the Python environment and install dependencies:

```bash
just setup
```

Key dependencies:
- prefect: Workflow orchestration
- fastapi: API server
- streamlit: Interactive UI
- ollama: LLM integration
- pdfplumber, tabula-py, PyPDF2: PDF parsing
- pandas, pydantic, mlflow: Data processing and validation
- loguru: Logging

## Step 4: Set Up Ollama

Install and run Ollama, then pull the LLM model (e.g., llama3):

```bash
# Install Ollama (follow platform-specific instructions: https://ollama.ai)
ollama pull llama3
ollama run llama3
```

## Step 5: Run the Application

Start the workflow, API, or UI:

```bash
# Run the full workflow
just run-workflow

# Start FastAPI server
just start-api

# Start Streamlit UI
just start-ui
```

## Next Steps

- [Usage Guide](usage.md)
- [API Documentation](api.md)