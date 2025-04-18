"""BentoML service for financial analysis."""
import logging

import bentoml
from pydantic import BaseModel

# Import functions from the provided file
from src.workflows import (
    analyze_transactions,
    build_timeline,
    categorize_transactions,
    generate_stories,
    generate_visualizations,
    process_nlp_queries,
    process_pdf_statements,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""Financial analysis service module."""


# Define Pydantic models for input validation
class PDFParseInput(BaseModel):
    """Input model for PDF parsing."""

    input_dir: str  # Path to directory containing PDFs
    output_csv: str  # Path to output CSV file


class TimelineInput(BaseModel):
    """Input model for timeline building."""

    input_csv: str  # Path to input CSV file
    output_csv: str  # Path to output CSV file


class CategorizeInput(BaseModel):
    """Input model for transaction categorization."""

    input_csv: str  # Path to input CSV file
    output_csv: str  # Path to output CSV file


class AnalyzeInput(BaseModel):
    """Input model for transaction analysis."""

    input_csv: str  # Path to input CSV file
    output_dir: str  # Path to output directory


class VisualizeInput(BaseModel):
    """Input model for visualization generation."""

    input_csv: str  # Path to input CSV file
    output_dir: str  # Path to output directory


class StoriesInput(BaseModel):
    """Input model for story generation."""

    input_csv: str  # Path to input CSV file
    output_file: str  # Path to output file


class NLPQueryInput(BaseModel):
    """Input model for NLP query processing."""

    input_csv: str  # Path to input CSV file
    query: str  # NLP query string
    output_file: str  # Path to output file


# Create BentoML service
@bentoml.service(
    name="financial_analysis_service",
    resources={"cpu": "2", "memory": "4Gi"},
)
class FinancialAnalysisService:
    """Service for financial data analysis and processing."""

    @bentoml.api
    async def parse_pdfs(self, input_data: PDFParseInput) -> dict:
        """Parse PDFs to transactions."""
        try:
            process_pdf_statements(input_data.input_dir, input_data.output_csv)
        except ValueError as e:
            logger.exception("Error in parse_pdfs")
            return {"status": "error", "message": str(e)}
        else:
            return {
                "status": "success",
                "message": f"PDFs parsed to {input_data.output_csv}",
            }

    @bentoml.api
    async def build_timeline(self, input_data: TimelineInput) -> dict:
        """Build transaction timeline."""
        try:
            build_timeline(input_data.input_csv, input_data.output_csv)
        except ValueError as e:
            logger.exception("Error in build_timeline")
            return {"status": "error", "message": str(e)}
        else:
            return {
                "status": "success",
                "message": f"Timeline built at {input_data.output_csv}",
            }

    @bentoml.api
    async def categorize_transactions(self, input_data: CategorizeInput) -> dict:
        """Categorize transactions."""
        try:
            categorize_transactions(input_data.input_csv, input_data.output_csv)
        except ValueError as e:
            logger.exception("Error in categorize_transactions")
            return {"status": "error", "message": str(e)}
        else:
            return {
                "status": "success",
                "message": f"Transactions categorized at {input_data.output_csv}",
            }

    @bentoml.api
    async def analyze_transactions(self, input_data: AnalyzeInput) -> dict:
        """Analyze transactions."""
        try:
            result = analyze_transactions(input_data.input_csv, input_data.output_dir)
        except ValueError as e:
            logger.exception("Error in analyze_transactions")
            return {"status": "error", "message": str(e)}
        else:
            return {"status": "success", "analysis": result}

    @bentoml.api
    async def generate_visualizations(self, input_data: VisualizeInput) -> dict:
        """Generate visualizations."""
        try:
            result = generate_visualizations(
                input_data.input_csv,
                input_data.output_dir,
            )
        except ValueError as e:
            logger.exception("Error in generate_visualizations")
            return {"status": "error", "message": str(e)}
        else:
            return {"status": "success", "visualizations": result}

    @bentoml.api
    async def generate_stories(self, input_data: StoriesInput) -> dict:
        """Generate financial stories."""
        try:
            result = generate_stories(input_data.input_csv, input_data.output_file)
        except ValueError as e:
            logger.exception("Error in generate_stories")
            return {"status": "error", "message": str(e)}
        else:
            return {"status": "success", "stories": result}

    @bentoml.api
    async def process_nlp_queries(self, input_data: NLPQueryInput) -> dict:
        """Process NLP query."""
        try:
            result = process_nlp_queries(
                input_data.input_csv,
                input_data.query,
                input_data.output_file,
            )
        except ValueError as e:
            logger.exception("Error in process_nlp_queries")
            return {"status": "error", "message": str(e)}
        else:
            return {"status": "success", "nlp_response": result}
