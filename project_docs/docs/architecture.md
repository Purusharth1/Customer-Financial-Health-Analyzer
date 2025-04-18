# Project Architecture

The **Customer Financial Health Analyzer** uses a modular architecture to process bank statements, analyze financial data, and deliver insights via a pipeline orchestrated by Prefect. Below is the system workflow.

## System Workflow

```mermaid
graph TD;
    A[Bank Statement PDFs] -->|Input| B[Prefect Workflow];
    B -->|PDF Parsing| C[pdf_parser.py];
    B -->|Timeline Building| D[timeline.py];
    B -->|Categorization| E[categorizer.py];
    B -->|Visualization| F[visualizer.py];
    B -->|Storytelling| G[storyteller.py];
    C -->|Transactions| H[all_transactions.csv];
    D -->|Sorted Transactions| I[timeline.csv];
    E -->|Categorized Transactions| J[categorized.csv];
    F -->|Charts Data| K[visualization_data.json];
    G -->|Narrative| L[stories.txt];
    H -->|Input| D;
    I -->|Input| E;
    J -->|Input| F;
    J -->|Input| G;
    K -->|Output| M[Streamlit UI];
    L -->|Output| M;
    K -->|Output| N[FastAPI];
    L -->|Output| N;
```

## Components
* **Bank Statement PDFs**: Input files containing transaction data.
* **Prefect Workflow**: Orchestrates tasks (workflows.py).
* **PDF Parser**: Extracts transactions and customer info using pdfplumber, tabula-py, and PyPDF2.
* **Timeline Builder**: Sorts transactions by date and adds parsed_date.
* **Categorizer**: Assigns categories (e.g., Shopping, Income) to transactions.
* **Visualizer**: Generates spending trends, expense breakdowns, and account overviews.
* **Storyteller**: Creates a financial narrative using Ollama LLM.
* **Outputs**: CSVs, JSON files, and text files stored in data/output/.
* **Streamlit UI**: Interactive dashboard for viewing transactions and insights.
* **FastAPI**: API for processing PDFs and retrieving results.

## Next Steps
* [Features](features.md)
* [Installation Guide](installation.md)