"""Financial Analysis API module."""
import logging
import shutil
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import functions from the src folder
from src.workflows import (
    analyze_transactions,
    build_timeline,
    categorize_transactions,
    financial_analysis_pipeline,
    generate_stories,
    generate_visualizations,
    process_nlp_queries,
    process_pdf_statements,
)

app = FastAPI(title="Financial Analysis API")
app.mount("/static", StaticFiles(directory="ui/static"), name="static")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic model for query input
class NLPQuery(BaseModel):
    """Model for NLP query input."""

    query: str

# Ensure output directories exist
OUTPUT_DIR = Path("data/output")
INPUT_DIR = Path("data/input")
for directory in [OUTPUT_DIR, INPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """Serve the main UI."""
    try:
        with open("ui/index.html") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="UI not found")


@app.post("/parse_pdfs")
async def parse_pdfs(
    files: Annotated[list[UploadFile], File(description="PDF files to parse")],
) -> dict[str, Any]:
    """Parse uploaded PDF statements to transactions."""
    def _raise_http_exception(e: Exception) -> None:
        logger.exception("Error parsing PDFs")
        raise HTTPException(status_code=500, detail=str(e)) from e

    try:
        # Save uploaded PDFs to input directory
        input_paths = []
        for file in files:
            file_path = INPUT_DIR / file.filename
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            input_paths.append(file_path)

        output_csv = OUTPUT_DIR / "all_transactions.csv"
        process_pdf_statements(str(INPUT_DIR), str(output_csv))
    except (OSError, ValueError, RuntimeError) as e:
        _raise_http_exception(e)
    else:
        return {"message": "PDFs parsed successfully", "output_csv": str(output_csv)}


@app.post("/build_timeline")
async def build_timeline_endpoint(
    input_csv: str, output_csv: str = str(OUTPUT_DIR / "timeline.csv"),
) -> dict[str, Any]:
    """Build transaction timeline from input CSV."""
    def _raise_not_found() -> None:
        raise HTTPException(status_code=404, detail="Input CSV not found")

    def _raise_http_exception(e: Exception) -> None:
        logger.exception("Error building timeline")
        raise HTTPException(status_code=500, detail=str(e)) from e

    try:
        if not Path(input_csv).exists():
            _raise_not_found()
        build_timeline(input_csv, output_csv)
    except (OSError, ValueError, RuntimeError) as e:
        _raise_http_exception(e)
    else:
        return {"message": "Timeline built successfully", "output_csv": output_csv}


@app.post("/categorize_transactions")
async def categorize_transactions_endpoint(
    input_csv: str, output_csv: str = str(OUTPUT_DIR / "categorized.csv"),
) -> dict[str, Any]:
    """Categorize transactions from input CSV."""
    def _raise_not_found() -> None:
        raise HTTPException(status_code=404, detail="Input CSV not found")

    def _raise_http_exception(e: Exception) -> None:
        logger.exception("Error categorizing transactions")
        raise HTTPException(status_code=500, detail=str(e)) from e

    try:
        if not Path(input_csv).exists():
            _raise_not_found()
        categorize_transactions(input_csv, output_csv)
    except (OSError, ValueError, RuntimeError) as e:
        _raise_http_exception(e)
    else:
        return {
            "message": "Transactions categorized successfully",
            "output_csv": output_csv,
        }


@app.post("/analyze_transactions")
async def analyze_transactions_endpoint(
    input_csv: str, output_dir: str = str(OUTPUT_DIR / "analysis"),
) -> dict[str, Any]:
    """Analyze transactions from input CSV."""
    def _raise_not_found() -> None:
        raise HTTPException(status_code=404, detail="Input CSV not found")

    def _raise_http_exception(e: Exception) -> None:
        logger.exception("Error analyzing transactions")
        raise HTTPException(status_code=500, detail=str(e)) from e

    try:
        if not Path(input_csv).exists():
            _raise_not_found()
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        result = analyze_transactions(input_csv, output_dir)
    except (OSError, ValueError, RuntimeError) as e:
        _raise_http_exception(e)
    else:
        return {"message": "Analysis completed successfully", "result": result}


@app.post("/generate_visualizations")
async def generate_visualizations_endpoint(
    input_csv: str, output_dir: str = str(OUTPUT_DIR / "charts"),
) -> dict[str, Any]:
    """Generate visualizations from input CSV."""
    def _raise_not_found() -> None:
        raise HTTPException(status_code=404, detail="Input CSV not found")

    def _raise_http_exception(e: Exception) -> None:
        logger.exception("Error generating visualizations")
        raise HTTPException(status_code=500, detail=str(e)) from e

    try:
        if not Path(input_csv).exists():
            _raise_not_found()
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        result = generate_visualizations(input_csv, output_dir)
    except (OSError, ValueError, RuntimeError) as e:
        _raise_http_exception(e)
    else:
        return {"message": "Visualizations generated successfully", "result": result}


@app.post("/generate_stories")
async def generate_stories_endpoint(
    input_csv: str, output_file: str = str(OUTPUT_DIR / "stories.txt"),
) -> dict[str, Any]:
    """Generate financial stories from input CSV."""
    def _raise_not_found() -> None:
        raise HTTPException(status_code=404, detail="Input CSV not found")

    def _raise_http_exception(e: Exception) -> None:
        logger.exception("Error generating stories")
        raise HTTPException(status_code=500, detail=str(e)) from e

    try:
        if not Path(input_csv).exists():
            _raise_not_found()
        result = generate_stories(input_csv, output_file)
    except (OSError, ValueError, RuntimeError) as e:
        _raise_http_exception(e)
    else:
        return {"message": "Stories generated successfully", "result": result}


@app.post("/process_nlp_queries")
async def process_nlp_queries_endpoint(
    input_csv: str,
    nlp_query: NLPQuery,
    output_file: str = str(OUTPUT_DIR / "nlp_response.txt"),
) -> dict[str, Any]:
    """Process NLP query on transactions."""
    def _raise_not_found() -> None:
        raise HTTPException(status_code=404, detail="Input CSV not found")

    def _raise_http_exception(e: Exception) -> None:
        logger.exception("Error processing NLP query")
        raise HTTPException(status_code=500, detail=str(e)) from e

    try:
        if not Path(input_csv).exists():
            _raise_not_found()
        result = process_nlp_queries(input_csv, nlp_query.query, output_file)
    except (OSError, ValueError, RuntimeError) as e:
        _raise_http_exception(e)
    else:
        return {"message": "NLP query processed successfully", "result": result}


@app.post("/run_pipeline")
async def run_pipeline_endpoint(
    input_dir: str = str(INPUT_DIR), query: str = "Restaurant spending last month",
) -> dict[str, Any]:
    """Run the full financial analysis pipeline."""
    def _raise_http_exception(e: Exception) -> None:
        logger.exception("Error running pipeline")
        raise HTTPException(status_code=500, detail=str(e)) from e

    try:
        result = financial_analysis_pipeline(input_dir, query)
    except (OSError, ValueError, RuntimeError) as e:
        _raise_http_exception(e)
    else:
        return {"message": "Pipeline executed successfully", "result": result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
