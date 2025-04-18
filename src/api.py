import time
from pathlib import Path

import mlflow
from fastapi import FastAPI, HTTPException, UploadFile

from src.pdf_parser import parse_hdfc_statement_improved
from src.utils import ensure_no_active_run, setup_mlflow

app = FastAPI(title="PDF Transaction Parser")


@app.post("/parse_pdf", response_model=list[dict])
async def parse_pdf(file: UploadFile):
    """Parse an HDFC bank statement PDF and return transactions as JSON."""
    setup_mlflow()
    ensure_no_active_run()
    with mlflow.start_run(run_name="PDF_Parsing_API"):
        start_time = time.time()
        mlflow.log_param("input_pdf", file.filename)

        # Validate file type
        if not file.filename.lower().endswith(".pdf"):
            mlflow.log_param("error", "Invalid file type")
            raise HTTPException(status_code=400, detail="File must be a PDF")

        temp_path = Path(f"/tmp/{file.filename}")
        try:
            # Read and save the PDF content
            content = await file.read()
            file_size = len(content)
            mlflow.log_metric("input_file_size_bytes", file_size)
            with temp_path.open("wb") as f:
                f.write(content)

            # Parse the PDF
            df = parse_hdfc_statement_improved(str(temp_path))

            # Log metrics and artifacts
            mlflow.log_metric("transactions_extracted", len(df))
            temp_csv = temp_path.with_suffix(".csv")
            df.to_csv(temp_csv, index=False)
            mlflow.log_artifact(str(temp_csv))

            # Prepare response
            transactions = df.to_dict(orient="records")

            # Clean up temporary files
            temp_path.unlink()
            temp_csv.unlink(missing_ok=True)

            # Log duration
            duration = time.time() - start_time
            mlflow.log_metric("parsing_duration_seconds", duration)

            return transactions
        except Exception as e:
            mlflow.log_param("error", str(e))
            if temp_path.exists():
                temp_path.unlink()
            raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {e!s}")
