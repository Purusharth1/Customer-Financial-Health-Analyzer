"""API Endpoint for Parsing HDFC Bank Statement PDFs.

This module provides a FastAPI endpoint to parse HDFC bank statement PDFs
and return extracted transactions as JSON. It integrates with MLflow for
logging and tracking.
"""
import logging
from pathlib import Path

import mlflow
from fastapi import FastAPI, HTTPException, UploadFile

from src.pdf_parser import parse_hdfc_statement_improved
from src.utils import setup_mlflow

app = FastAPI(title="PDF Transaction Parser")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.post("/parse_pdf", response_model=list[dict])
async def parse_pdf(file: UploadFile):
    """Parse an HDFC bank statement PDF and return transactions as JSON."""
    setup_mlflow()
    with mlflow.start_run(run_name="PDF_Parsing_API"):
        mlflow.log_param("input_pdf", file.filename)

        if not file.filename.lower().endswith(".pdf"):
            logger.error("Invalid file type: %s", file.filename)
            mlflow.log_param("error", "Invalid file type")
            raise HTTPException(status_code=400, detail="File must be a PDF")

        temp_path = Path(f"/tmp/{file.filename}")
        try:
            with temp_path.open("wb") as f:
                content = await file.read()
                f.write(content)
            logger.info("Saved temporary PDF: %s", temp_path)

            df = parse_hdfc_statement_improved(str(temp_path))
            logger.info("Extracted %d transactions", len(df))

            mlflow.log_metric("transactions_extracted", len(df))
            temp_csv = temp_path.with_suffix(".csv")
            df.to_csv(temp_csv, index=False)
            mlflow.log_artifact(str(temp_csv))

            transactions = df.to_dict(orient="records")

            temp_path.unlink()
            temp_csv.unlink(missing_ok=True)

            return transactions
        except Exception as e:
            logger.error("Failed to parse PDF: %s", str(e))
            mlflow.log_param("error", str(e))
            if temp_path.exists():
                temp_path.unlink()
            raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {e!s}")
