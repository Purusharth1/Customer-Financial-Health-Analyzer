"""FastAPI backend for Customer Financial Health Analyzer.

Provides API endpoints for PDF uploads, NLP queries, visualizations, transactions,
analysis, and stories. To be used with a Streamlit frontend.
"""
import json
import logging
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.analyzer import analyze_transactions
from src.models import AnalyzerInput, FinancialPipelineInput, StorytellerInput
from src.storyteller import generate_stories
from src.workflows import financial_analysis_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Financial Health Analyzer")

# Add CORS middleware to allow requests from Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input/output directories
INPUT_DIR = Path("data/input")
OUTPUT_DIR = Path("data/output")
CHARTS_DIR = Path("data/output/charts")
CATEGORIZED_CSV = OUTPUT_DIR / "categorized.csv"
NLP_RESPONSE = OUTPUT_DIR / "nlp_response.txt"
VIZ_FILE = CHARTS_DIR / "visualization_data.json"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
STORIES_FILE = OUTPUT_DIR / "stories.txt"
NLP_VIZ_FILE = OUTPUT_DIR / "visualization_data.json"

# Ensure directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Constants
MAX_PDF_UPLOADS = 10
ALLOWED_FILE_EXTENSION = ".pdf"

class QueryRequest(BaseModel):
    """Schema for query requests."""

    query: str

class QueryResponse(BaseModel):
    """Schema for query responses."""

    response: str
    visualization: dict[str, Any] | None = None

def _raise_invalid_file_type_error() -> None:
    """Raise HTTPException for invalid file type."""
    raise HTTPException(status_code=400, detail="Only PDF files allowed")

def _raise_categorized_csv_missing_error() -> None:
    """Raise HTTPException for missing categorized transactions."""
    raise HTTPException(
        status_code=500, detail="Failed to generate categorized transactions",
    )

def _raise_empty_transactions_error() -> None:
    """Raise HTTPException for empty transactions."""
    raise HTTPException(status_code=404, detail="No transactions found")

@app.post("/upload-pdfs/", status_code=200)  # Changed from 202 to 200
async def upload_pdfs(
    files: Annotated[list[UploadFile], File()],
) -> dict[str, Any]:
    """Upload and process PDF bank statements."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    if len(files) > MAX_PDF_UPLOADS:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_PDF_UPLOADS} PDFs allowed",
        )

    try:
        # Save PDFs
        for file in files:
            if (not file.filename or
                not file.filename.lower().endswith(ALLOWED_FILE_EXTENSION)):
                _raise_invalid_file_type_error()
            file_path = INPUT_DIR / file.filename
            with file_path.open("wb") as f:
                f.write(await file.read())
            logger.info("Saved PDF: %s", file_path)

        # Run workflow to parse and categorize
        input_model = FinancialPipelineInput(input_dir=INPUT_DIR, query="")
        financial_analysis_pipeline(input_model)
        if not CATEGORIZED_CSV.exists():
            _raise_categorized_csv_missing_error()

        # Run analysis and stories
        analyzer_input = AnalyzerInput(
            input_csv=CATEGORIZED_CSV,
            output_dir=ANALYSIS_DIR,
        )
        analyze_transactions(analyzer_input)
        storyteller_input = StorytellerInput(
            input_csv=CATEGORIZED_CSV,
            output_file=STORIES_FILE,
        )
        generate_stories(storyteller_input)

        return {"status": "PDFs processed", "files_uploaded": len(files)}
    except (ValueError, OSError, pd.errors.EmptyDataError) as e:
        logger.exception("PDF upload error")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/query/")
async def process_query(request: QueryRequest) -> QueryResponse:
    """Process NLP query and return response/visualization."""
    if not CATEGORIZED_CSV.exists():
        raise HTTPException(
            status_code=400,
            detail="No processed transactions. Upload PDFs first.",
        )

    try:
        input_model = FinancialPipelineInput(input_dir=INPUT_DIR, query=request.query)
        result = financial_analysis_pipeline(input_model)
        response_text = result.nlp_response
        visualization = None
        if NLP_VIZ_FILE.exists():
            with NLP_VIZ_FILE.open() as f:
                visualization = json.load(f)
            logger.info("Loaded visualization for query: %s", request.query)

        return QueryResponse(response=response_text, visualization=visualization)
    except (ValueError, OSError, json.JSONDecodeError) as e:
        logger.exception("Query error")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/visualizations/")
async def get_visualizations() -> dict[str, Any]:
    """Retrieve latest visualization data."""
    if not VIZ_FILE.exists():
        raise HTTPException(status_code=404, detail="No visualizations available")
    try:
        with VIZ_FILE.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.exception("Visualization fetch error")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/transactions/")
async def get_transactions() -> list[dict[str, Any]]:
    """Fetch processed transactions for display."""
    if not CATEGORIZED_CSV.exists():
        raise HTTPException(status_code=404, detail="No transactions available")
    try:
        transactions_df = pd.read_csv(CATEGORIZED_CSV)
        if transactions_df.empty:
            _raise_empty_transactions_error()
        return transactions_df[
            [
                "parsed_date", "Narration", "Withdrawal (INR)",
                "Deposit (INR)", "category",
            ]
        ].to_dict("records")
    except (OSError, pd.errors.ParserError) as e:
        logger.exception("Transactions fetch error")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/analysis/")
async def get_analysis() -> Any:
    """Retrieve financial analysis results (patterns, fees,recurring."""
    if not ANALYSIS_DIR.exists() or not any(ANALYSIS_DIR.iterdir()):
        raise HTTPException(
            status_code=404,
            detail="No analysis available. Upload PDFs first.",
        )

    try:
        analyzer_input = AnalyzerInput(
            input_csv=CATEGORIZED_CSV,
            output_dir=ANALYSIS_DIR,
        )
        return analyze_transactions(analyzer_input)
    except (ValueError, OSError) as e:
        logger.exception("Analysis fetch error")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/stories/")
async def get_stories() -> list[str]:
    """Retrieve financial stories."""
    if not STORIES_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="No stories available. Upload PDFs first.",
        )
    try:
        with STORIES_FILE.open("r") as f:
            return f.read().splitlines()
    except OSError as e:
        logger.exception("Stories fetch error")
        raise HTTPException(status_code=500, detail=str(e)) from e
