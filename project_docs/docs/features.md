# Features

The **Customer Financial Health Analyzer** project offers the following features:

## 1. PDF Transaction Extraction
- Parse bank statement PDFs to extract transactions and customer information (e.g., name, account number).
- Supports multiple PDFs (1-10 per run).

## 2. Transaction Timeline
- Organize transactions chronologically with standardized date formats (e.g., `YYYY-MM-DD`).
- Drop invalid dates for data integrity.

## 3. Spending Categorization
- Classify transactions into categories (e.g., Shopping, Income, Expense (Other)) based on narration.

## 4. Financial Visualizations
- Generate:
  - **Spending Trends**: Monthly expenses vs. budget.
  - **Expense Breakdown**: Category-wise percentage of spending.
  - **Account Overview**: Total balance, monthly income/expense, and growth percentages.

## 5. Financial Narrative
- Create a cohesive, LLM-generated story (300-500 words) summarizing financial habits, trends, and recommendations.

## 6. Interactive Dashboard
- Streamlit UI for uploading PDFs and viewing transactions, visualizations, and narratives.
- FastAPI endpoints for programmatic access (`/process`).

## 7. Workflow Orchestration
- Prefect manages the pipeline, ensuring robust task execution and retries.
- MLflow logs artifacts (CSVs, JSON, text) and metrics (e.g., transactions processed).

---

## Next Steps

- [Installation Guide](installation.md)
- [Usage Guide](usage.md)