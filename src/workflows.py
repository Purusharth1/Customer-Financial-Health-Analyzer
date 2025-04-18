"""Prefect Workflows for Orchestration.

This module defines Prefect workflows to orchestrate the end-to-end financial analysis pipeline.
Key functionalities include:
- Coordinating tasks such as PDF parsing, categorization, and analysis.
- Managing dependencies between different modules.
- Scheduling and automating workflows for batch processing.
- Monitoring and logging workflow execution.
"""

import logging
import sys
from pathlib import Path

from prefect import flow, task

sys.path.append(str(Path(__file__).parent.parent))
from src.analyzer import analyze_transactions
from src.categorizer import categorize_transactions
from src.models import (
    AnalyzerInput,
    AnalyzerOutput,
    CategorizerInput,
    CategorizerOutput,
    FinancialPipelineInput,
    FinancialPipelineOutput,
    NlpProcessorInput,
    NlpProcessorOutput,
    PdfProcessingInput,
    PdfProcessingOutput,
    StorytellerInput,
    StorytellerOutput,
    TimelineInput,
    TimelineOutput,
    VisualizerInput,
    VisualizerOutput,
)
from src.nlp_processor import process_nlp_queries
from src.pdf_parser import process_pdf_statements
from src.storyteller import generate_stories
from src.timeline import build_timeline
from src.visualizer import generate_visualizations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@task
def parse_pdfs_task(input_dir: str, output_csv: str) -> PdfProcessingOutput:
    """Parse PDFs to transactions."""
    input_model = PdfProcessingInput(folder_path=Path(input_dir), output_csv=Path(output_csv))
    logger.info(f"Parsing PDFs from {input_dir} to {output_csv}")
    try:
        result = process_pdf_statements(input_model)
        logger.info(f"Parsed {len(result.transactions)} PDFs to {output_csv}")
        return result
    except Exception as e:
        logger.error(f"PDF parsing failed: {e}")
        raise


@task
def build_timeline_task(input_csv: str, output_csv: str) -> TimelineOutput:
    """Build transaction timeline."""
    input_model = TimelineInput(transactions_csv=Path(input_csv), output_csv=Path(output_csv))
    logger.info(f"Building timeline from {input_csv} to {output_csv}")
    try:
        result = build_timeline(input_model)
        logger.info(f"Built timeline to {output_csv}")
        return result
    except Exception as e:
        logger.error(f"Timeline building failed: {e}")
        raise

@task
def categorize_transactions_task(input_csv: str, output_csv: str) -> CategorizerOutput:
    """Categorize transactions."""
    input_model = CategorizerInput(timeline_csv=Path(input_csv), output_csv=Path(output_csv))
    logger.info(f"Categorizing transactions from {input_csv} to {output_csv}")
    try:
        result = categorize_transactions(input_model)
        logger.info(f"Categorized transactions to {output_csv}")
        return result
    except Exception as e:
        logger.error(f"Categorization failed: {e}")
        raise

@task
def analyze_transactions_task(input_csv: str, output_dir: str) -> AnalyzerOutput:
    """Analyze transactions."""
    input_model = AnalyzerInput(input_csv=Path(input_csv), output_dir=Path(output_dir))
    logger.info(f"Analyzing transactions from {input_csv} to {output_dir}")
    try:
        result = analyze_transactions(input_model)
        logger.info(f"Analysis saved to {output_dir}")
        return result
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

@task
def generate_visualizations_task(input_csv: str, output_dir: str) -> VisualizerOutput:
    """Generate visualizations."""
    input_model = VisualizerInput(input_csv=Path(input_csv), output_dir=Path(output_dir))
    logger.info(f"Generating visualizations from {input_csv} to {output_dir}")
    try:
        result = generate_visualizations(input_model)
        logger.info(f"Visualizations saved to {output_dir}")
        return result
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        raise

@task
def generate_stories_task(input_csv: str, output_file: str) -> StorytellerOutput:
    """Generate financial stories."""
    input_model = StorytellerInput(input_csv=Path(input_csv), output_file=Path(output_file))
    logger.info(f"Generating stories from {input_csv} to {output_file}")
    try:
        result = generate_stories(input_model)
        logger.info(f"Stories saved to {output_file}")
        return result
    except Exception as e:
        logger.error(f"Story generation failed: {e}")
        raise

@task
def process_nlp_queries_task(
    input_csv: str,
    query: str,
    output_file: str,
    visualization_file: str | None = None,
) -> NlpProcessorOutput:
    """Process NLP query."""
    input_model = NlpProcessorInput(
        input_csv=Path(input_csv),
        query=query,
        output_file=Path(output_file),
        visualization_file=Path(visualization_file) if visualization_file else None,
    )
    logger.info(f"Processing NLP query from {input_csv} to {output_file}")
    try:
        result = process_nlp_queries(input_model)
        logger.info(f"NLP response saved to {output_file}")
        return result
    except Exception as e:
        logger.error(f"NLP query failed: {e}")
        raise


@flow(name="Financial_Analysis_Pipeline")
def financial_analysis_pipeline(
    input_model: FinancialPipelineInput = FinancialPipelineInput(),
) -> FinancialPipelineOutput:
    """Orchestrate financial analysis tasks.

    Args:
    ----
        input_dir: Directory with PDFs.
        query: NLP query to process.

    Returns:
    -------
        Dictionary with task results.

    """
    logger.info("Starting financial analysis pipeline")

    # Define paths relative to project root
    base_dir = Path(__file__).parent.parent
    input_dir = (
        base_dir / input_model.input_dir if not input_model.input_dir.is_absolute() else input_model.input_dir
    )
    output_dir = base_dir / "data" / "output"

    transactions_csv = output_dir / "all_transactions.csv"
    timeline_csv = output_dir / "timeline.csv"
    categorized_csv = output_dir / "categorized.csv"
    analysis_dir = output_dir / "analysis"
    charts_dir = output_dir / "charts"
    stories_file = output_dir / "stories.txt"
    nlp_file = output_dir / "nlp_response.txt"
    visualization_file = output_dir / "visualization_data.json"

    # Ensure directories exist
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    # Run tasks
    parse_pdfs_task(str(input_dir), str(transactions_csv))
    build_timeline_task(str(transactions_csv), str(timeline_csv))
    categorize_transactions_task(str(timeline_csv), str(categorized_csv))
    analysis = analyze_transactions_task(str(categorized_csv), str(analysis_dir))
    visualizations = generate_visualizations_task(str(categorized_csv), str(charts_dir))
    stories = generate_stories_task(str(categorized_csv), str(stories_file))
    nlp_response = process_nlp_queries_task(
        str(categorized_csv),
        input_model.query,
        str(nlp_file),
        str(visualization_file),
    )

    return FinancialPipelineOutput(
        analysis=analysis,
        visualizations=visualizations,
        stories=stories.stories,
        nlp_response=nlp_response.text_response,
    )


if __name__ == "__main__":
    results = financial_analysis_pipeline()
    print(results)
