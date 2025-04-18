"""API Endpoint for Building Transaction Timeline from CSV.

This module provides a FastAPI endpoint to build a chronological timeline from a
transactions CSV and return the results as JSON. It integrates with MLflow for
logging and tracking.
"""

import time
from pathlib import Path

import mlflow
from fastapi import FastAPI, HTTPException, UploadFile

from src.models import TimelineInput, TimelineOutput
from src.timeline import build_timeline
from src.utils import ensure_no_active_run, sanitize_metric_name, setup_mlflow

app = FastAPI(title="Transaction Timeline Builder")


@app.post("/build_timeline", response_model=TimelineOutput)
async def build_timeline_endpoint(file: UploadFile):
    """Build a chronological timeline from a transactions CSV."""
    setup_mlflow()
    ensure_no_active_run()
    with mlflow.start_run(run_name="Timeline_API"):
        start_time = time.time()
        mlflow.log_param("input_csv", file.filename)

        # Validate file type
        if not file.filename.lower().endswith(".csv"):
            mlflow.log_param("error", "Invalid file type")
            raise HTTPException(status_code=400, detail="File must be a CSV")

        temp_input = Path(f"/tmp/{file.filename}")
        temp_output = temp_input.with_stem(f"{temp_input.stem}_timeline").with_suffix(
            ".csv",
        )
        try:
            # Save temporary CSV
            content = await file.read()
            file_size = len(content)
            mlflow.log_metric("input_file_size_bytes", file_size)
            with temp_input.open("wb") as f:
                f.write(content)

            # Build timeline
            input_model = TimelineInput(
                transactions_csv=temp_input,
                output_csv=temp_output,
            )
            timeline_output = build_timeline(input_model)

            # Log metrics and artifacts
            mlflow.log_metric(
                sanitize_metric_name("transactions_timed"),
                len(timeline_output.transactions),
            )
            if temp_output.exists():
                mlflow.log_artifact(str(temp_output))

            # Clean up temporary files
            temp_input.unlink()
            temp_output.unlink(missing_ok=True)

            # Log duration
            duration = time.time() - start_time
            mlflow.log_metric("build_timeline_duration_seconds", duration)

            return timeline_output
        except Exception as e:
            mlflow.log_param("error", str(e))
            if temp_input.exists():
                temp_input.unlink()
            if temp_output.exists():
                temp_output.unlink()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to build timeline: {e!s}",
            )
