"""FastAPI backend for Customer Financial Health Analyzer.

Provides endpoints for PDF uploads, NLP queries, visualizations, transactions, analysis, and stories.
Serves a basic UI for user interaction.
"""
import json
import logging
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.analyzer import analyze_transactions
from src.storyteller import generate_stories
from src.workflows import financial_analysis_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Financial Health Analyzer")

# Mount static files for UI
app.mount("/static", StaticFiles(directory="ui/static"), name="static")

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

class QueryRequest(BaseModel):
    """Schema for query requests."""

    query: str

class QueryResponse(BaseModel):
    """Schema for query responses."""

    response: str
    visualization: dict | None = None

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """Serve the main UI."""
    with open("ui/index.html") as f:
        return HTMLResponse(content=f.read())

@app.post("/upload-pdfs/", status_code=202)
async def upload_pdfs(files: list[UploadFile] = File(...)):
    """Upload and process PDF bank statements.

    Saves PDFs to data/input/, runs workflow, and generates categorized.csv.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 PDFs allowed")

    try:
        # Save PDFs
        for file in files:
            if not file.filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail="Only PDF files allowed")
            file_path = INPUT_DIR / file.filename
            with file_path.open("wb") as f:
                f.write(await file.read())
            logger.info(f"Saved PDF: {file_path}")

        # Run workflow to parse and categorize
        result = financial_analysis_pipeline(input_dir=str(INPUT_DIR), query="")
        if not CATEGORIZED_CSV.exists():
            raise HTTPException(status_code=500, detail="Failed to generate categorized transactions")

        # Run analysis and stories
        analyze_transactions(str(CATEGORIZED_CSV), str(ANALYSIS_DIR))
        generate_stories(str(CATEGORIZED_CSV), str(STORIES_FILE))

        return {"status": "PDFs processed", "files_uploaded": len(files)}
    except Exception as e:
        logger.error(f"PDF upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process NLP query and return response/visualization.
    """
    if not CATEGORIZED_CSV.exists():
        raise HTTPException(status_code=400, detail="No processed transactions. Upload PDFs first.")

    try:
        result = financial_analysis_pipeline(
            input_dir=str(INPUT_DIR),
            query=request.query,
        )
        response_text = result["nlp_response"]
        visualization = None
        if NLP_VIZ_FILE.exists():
            with NLP_VIZ_FILE.open() as f:
                visualization = json.load(f)
            logger.info(f"Loaded visualization for query: {request.query}")

        return QueryResponse(response=response_text, visualization=visualization)
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualizations/")
async def get_visualizations():
    """Retrieve latest visualization data.
    """
    if not VIZ_FILE.exists():
        raise HTTPException(status_code=404, detail="No visualizations available")
    try:
        with VIZ_FILE.open() as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Visualization fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transactions/")
async def get_transactions():
    """Fetch processed transactions for display.
    """
    if not CATEGORIZED_CSV.exists():
        raise HTTPException(status_code=404, detail="No transactions available")
    try:
        df = pd.read_csv(CATEGORIZED_CSV)
        if df.empty:
            raise HTTPException(status_code=404, detail="No transactions found")
        return df[["parsed_date", "Narration", "Withdrawal (INR)", "Deposit (INR)", "category"]].to_dict("records")
    except Exception as e:
        logger.error(f"Transactions fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis/")
async def get_analysis():
    """Retrieve financial analysis results (patterns, fees, recurring, anomalies, cash flow).
    """
    if not ANALYSIS_DIR.exists() or not any(ANALYSIS_DIR.iterdir()):
        raise HTTPException(status_code=404, detail="No analysis available. Upload PDFs first.")
    try:
        results = analyze_transactions(str(CATEGORIZED_CSV), str(ANALYSIS_DIR))
        return results
    except Exception as e:
        logger.error(f"Analysis fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stories/")
async def get_stories():
    """Retrieve financial stories.
    """
    if not STORIES_FILE.exists():
        raise HTTPException(status_code=404, detail="No stories available. Upload PDFs first.")
    try:
        with STORIES_FILE.open("r") as f:
            stories = f.read().splitlines()
        return stories
    except Exception as e:
        logger.error(f"Stories fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
