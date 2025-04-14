import logging
import os
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic model for query input
class NLPQuery(BaseModel):
    query: str

# Ensure output directories exist
OUTPUT_DIR = "data/output/"
INPUT_DIR = "data/input/"
for directory in [OUTPUT_DIR, INPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

@app.post("/parse_pdfs")
async def parse_pdfs(files: list[UploadFile] = File(...)):
    """Parse uploaded PDF statements to transactions."""
    try:
        # Save uploaded PDFs to input directory
        input_paths = []
        for file in files:
            file_path = os.path.join(INPUT_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            input_paths.append(file_path)

        output_csv = os.path.join(OUTPUT_DIR, "all_transactions.csv")
        process_pdf_statements(INPUT_DIR, output_csv)
        return {"message": "PDFs parsed successfully", "output_csv": output_csv}
    except Exception as e:
        logger.error(f"Error parsing PDFs: {e!s}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/build_timeline")
async def build_timeline_endpoint(input_csv: str, output_csv: str = os.path.join(OUTPUT_DIR, "timeline.csv")):
    """Build transaction timeline from input CSV."""
    try:
        if not os.path.exists(input_csv):
            raise HTTPException(status_code=404, detail="Input CSV not found")
        build_timeline(input_csv, output_csv)
        return {"message": "Timeline built successfully", "output_csv": output_csv}
    except Exception as e:
        logger.error(f"Error building timeline: {e!s}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/categorize_transactions")
async def categorize_transactions_endpoint(input_csv: str, output_csv: str = os.path.join(OUTPUT_DIR, "categorized.csv")):
    """Categorize transactions from input CSV."""
    try:
        if not os.path.exists(input_csv):
            raise HTTPException(status_code=404, detail="Input CSV not found")
        categorize_transactions(input_csv, output_csv)
        return {"message": "Transactions categorized successfully", "output_csv": output_csv}
    except Exception as e:
        logger.error(f"Error categorizing transactions: {e!s}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_transactions")
async def analyze_transactions_endpoint(input_csv: str, output_dir: str = os.path.join(OUTPUT_DIR, "analysis")):
    """Analyze transactions from input CSV."""
    try:
        if not os.path.exists(input_csv):
            raise HTTPException(status_code=404, detail="Input CSV not found")
        os.makedirs(output_dir, exist_ok=True)
        result = analyze_transactions(input_csv, output_dir)
        return {"message": "Analysis completed successfully", "result": result}
    except Exception as e:
        logger.error(f"Error analyzing transactions: {e!s}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_visualizations")
async def generate_visualizations_endpoint(input_csv: str, output_dir: str = os.path.join(OUTPUT_DIR, "charts")):
    """Generate visualizations from input CSV."""
    try:
        if not os.path.exists(input_csv):
            raise HTTPException(status_code=404, detail="Input CSV not found")
        os.makedirs(output_dir, exist_ok=True)
        result = generate_visualizations(input_csv, output_dir)
        return {"message": "Visualizations generated successfully", "result": result}
    except Exception as e:
        logger.error(f"Error generating visualizations: {e!s}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_stories")
async def generate_stories_endpoint(input_csv: str, output_file: str = os.path.join(OUTPUT_DIR, "stories.txt")):
    """Generate financial stories from input CSV."""
    try:
        if not os.path.exists(input_csv):
            raise HTTPException(status_code=404, detail="Input CSV not found")
        result = generate_stories(input_csv, output_file)
        return {"message": "Stories generated successfully", "result": result}
    except Exception as e:
        logger.error(f"Error generating stories: {e!s}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_nlp_queries")
async def process_nlp_queries_endpoint(input_csv: str, nlp_query: NLPQuery, output_file: str = os.path.join(OUTPUT_DIR, "nlp_response.txt")):
    """Process NLP query on transactions."""
    try:
        if not os.path.exists(input_csv):
            raise HTTPException(status_code=404, detail="Input CSV not found")
        result = process_nlp_queries(input_csv, nlp_query.query, output_file)
        return {"message": "NLP query processed successfully", "result": result}
    except Exception as e:
        logger.error(f"Error processing NLP query: {e!s}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run_pipeline")
async def run_pipeline_endpoint(input_dir: str = INPUT_DIR, query: str = "Restaurant spending last month"):
    """Run the full financial analysis pipeline."""
    try:
        result = financial_analysis_pipeline(input_dir, query)
        return {"message": "Pipeline executed successfully", "result": result}
    except Exception as e:
        logger.error(f"Error running pipeline: {e!s}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
