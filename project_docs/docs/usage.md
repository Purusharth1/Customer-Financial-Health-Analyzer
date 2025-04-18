# Usage Guide

This guide explains how to use the **Customer Financial Health Analyzer** to process bank statements and analyze financial data.

## Workflow Examples

### Process Bank Statements
- Place PDF bank statements in `data/input/`.
- Run the workflow to parse PDFs, categorize transactions, and generate insights:
  ```bash
  just run-workflow
  ```
- Outputs are saved in `data/output/` (e.g., all_transactions.csv, visualization_data.json, stories.txt).

### View Insights via Streamlit
- Start the Streamlit UI:
  ```bash
  just start-ui
  ```
- Open http://localhost:8501, upload PDFs, and view:
   * **Transactions Tab**: Parsed transactions.
   * **Analysis Tab**: Spending trends, expense breakdowns, and financial narratives.

### Query via API
- Start the FastAPI server:
  ```bash
  just start-api
  ```
- Process PDFs:
  ```bash
  curl http://localhost:8000/process
  ```
- Check API documentation at http://localhost:8000/docs.

## Steps to Use
1. **Prepare Input**: Place bank statement PDFs in data/input/.
2. **Run Workflow**: Execute uv run src/workflows.py to process data.
3. **Explore Results**:
   * View CSVs and JSON files in data/output/.
   * Use Streamlit (http://localhost:8501) for interactive insights.
   * Query the FastAPI endpoint (http://localhost:8000/process).
4. **Review Logs**: Check logs in logs/ for debugging.

## Troubleshooting
* **No Data Parsed**: Ensure PDFs are valid bank statements and placed in data/input/.
* **LLM Errors**: Verify Ollama is running (ollama ps) and the model is loaded (ollama list).
* **Serialization Errors**: Check for Pydantic model mismatches in logs.
* **Slow Processing**: Optimize PDF parsing by limiting the number of PDFs (1-10).

## Next Steps
* [API Documentation](api.md)
* [Experience and Learnings](learnings.md)