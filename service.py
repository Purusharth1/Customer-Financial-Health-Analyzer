import bentoml
from bentoml.io import JSON, File
from pydantic import BaseModel
import os
import shutil
import logging
from typing import List, Annotated
from fastapi import UploadFile

# Import functions from the src folder
from src.workflows import (
    process_pdf_statements,
    build_timeline,
    categorize_transactions,
    analyze_transactions,
    generate_visualizations,
    generate_stories,
    process_nlp_queries,
    financial_analysis_pipeline
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for input validation
class CSVInput(BaseModel):
    input_csv: str
    output_csv: str | None = None

class AnalysisInput(BaseModel):
    input_csv: str
    output_dir: str | None = None

class StoriesInput(BaseModel):
    input_csv: str
    output_file: str | None = None

class NLPQueryInput(BaseModel):
    input_csv: str
    query: str
    output_file: str | None = None

class PipelineInput(BaseModel):
    input_dir: str | None = None
    query: str | None = None

# Define directories
OUTPUT_DIR = "data/output/"
INPUT_DIR = "data/input/"

# Ensure directories exist
for directory in [OUTPUT_DIR, INPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Define BentoML service with the new decorator
@bentoml.service(
    name="financial_analysis_service",
    resources={"cpu": "1"}
)
class FinancialAnalysisService:
    
    @bentoml.api
    async def parse_pdfs(self, files: Annotated[List[UploadFile], File()]) -> dict:
        """Parse uploaded PDF statements to transactions."""
        try:
            # Save uploaded PDFs to input directory
            for file in files:
                file_path = os.path.join(INPUT_DIR, file.filename)
                with open(file_path, "wb") as f:
                    shutil.copyfileobj(file.file, f)
            
            output_csv = os.path.join(OUTPUT_DIR, "all_transactions.csv")
            process_pdf_statements(INPUT_DIR, output_csv)
            return {"message": "PDFs parsed successfully", "output_csv": output_csv}
        except Exception as e:
            logger.error(f"Error parsing PDFs: {str(e)}")
            raise bentoml.BentoMLException(str(e))

    @bentoml.api(input=JSON(pydantic_model=CSVInput), output=JSON())
    async def build_timeline(self, payload: CSVInput) -> dict:
        """Build transaction timeline from input CSV."""
        try:
            input_csv = payload.input_csv
            output_csv = payload.output_csv or os.path.join(OUTPUT_DIR, "timeline.csv")
            if not input_csv or not os.path.exists(input_csv):
                raise bentoml.BentoMLException("Input CSV not found")
            build_timeline(input_csv, output_csv)
            return {"message": "Timeline built successfully", "output_csv": output_csv}
        except Exception as e:
            logger.error(f"Error building timeline: {str(e)}")
            raise bentoml.BentoMLException(str(e))

    @bentoml.api(input=JSON(pydantic_model=CSVInput), output=JSON())
    async def categorize_transactions(self, payload: CSVInput) -> dict:
        """Categorize transactions from input CSV."""
        try:
            input_csv = payload.input_csv
            output_csv = payload.output_csv or os.path.join(OUTPUT_DIR, "categorized.csv")
            if not input_csv or not os.path.exists(input_csv):
                raise bentoml.BentoMLException("Input CSV not found")
            categorize_transactions(input_csv, output_csv)
            return {"message": "Transactions categorized successfully", "output_csv": output_csv}
        except Exception as e:
            logger.error(f"Error categorizing transactions: {str(e)}")
            raise bentoml.BentoMLException(str(e))

    @bentoml.api(input=JSON(pydantic_model=AnalysisInput), output=JSON())
    async def analyze_transactions(self, payload: AnalysisInput) -> dict:
        """Analyze transactions from input CSV."""
        try:
            input_csv = payload.input_csv
            output_dir = payload.output_dir or os.path.join(OUTPUT_DIR, "analysis")
            if not input_csv or not os.path.exists(input_csv):
                raise bentoml.BentoMLException("Input CSV not found")
            os.makedirs(output_dir, exist_ok=True)
            result = analyze_transactions(input_csv, output_dir)
            return {"message": "Analysis completed successfully", "result": result}
        except Exception as e:
            logger.error(f"Error analyzing transactions: {str(e)}")
            raise bentoml.BentoMLException(str(e))

    @bentoml.api(input=JSON(pydantic_model=AnalysisInput), output=JSON())
    async def generate_visualizations(self, payload: AnalysisInput) -> dict:
        """Generate visualizations from input CSV."""
        try:
            input_csv = payload.input_csv
            output_dir = payload.output_dir or os.path.join(OUTPUT_DIR, "charts")
            if not input_csv or not os.path.exists(input_csv):
                raise bentoml.BentoMLException("Input CSV not found")
            os.makedirs(output_dir, exist_ok=True)
            result = generate_visualizations(input_csv, output_dir)
            return {"message": "Visualizations generated successfully", "result": result}
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            raise bentoml.BentoMLException(str(e))

    @bentoml.api(input=JSON(pydantic_model=StoriesInput), output=JSON())
    async def generate_stories(self, payload: StoriesInput) -> dict:
        """Generate financial stories from input CSV."""
        try:
            input_csv = payload.input_csv
            output_file = payload.output_file or os.path.join(OUTPUT_DIR, "stories.txt")
            if not input_csv or not os.path.exists(input_csv):
                raise bentoml.BentoMLException("Input CSV not found")
            result = generate_stories(input_csv, output_file)
            return {"message": "Stories generated successfully", "result": result}
        except Exception as e:
            logger.error(f"Error generating stories: {str(e)}")
            raise bentoml.BentoMLException(str(e))

    @bentoml.api(input=JSON(pydantic_model=NLPQueryInput), output=JSON())
    async def process_nlp_queries(self, payload: NLPQueryInput) -> dict:
        """Process NLP query on transactions."""
        try:
            input_csv = payload.input_csv
            query = payload.query
            output_file = payload.output_file or os.path.join(OUTPUT_DIR, "nlp_response.txt")
            if not input_csv or not os.path.exists(input_csv):
                raise bentoml.BentoMLException("Input CSV not found")
            if not query:
                raise bentoml.BentoMLException("Query is required")
            result = process_nlp_queries(input_csv, query, output_file)
            return {"message": "NLP query processed successfully", "result": result}
        except Exception as e:
            logger.error(f"Error processing NLP query: {str(e)}")
            raise bentoml.BentoMLException(str(e))

    @bentoml.api(input=JSON(pydantic_model=PipelineInput), output=JSON())
    async def run_pipeline(self, payload: PipelineInput) -> dict:
        """Run the full financial analysis pipeline."""
        try:
            input_dir = payload.input_dir or INPUT_DIR
            query = payload.query or "Restaurant spending last month"
            result = financial_analysis_pipeline(input_dir, query)
            return {"message": "Pipeline executed successfully", "result": result}
        except Exception as e:
            logger.error(f"Error running pipeline: {str(e)}")
            raise bentoml.BentoMLException(str(e))