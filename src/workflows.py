"""Prefect Workflows for Orchestration.

This module defines Prefect workflows to orchestrate the end-to-end financial analysis pipeline.
Key functionalities include:
- Coordinating tasks such as PDF parsing, categorization, and analysis.
- Managing dependencies between different modules.
- Scheduling and automating workflows for batch processing.
- Monitoring and logging workflow execution.
"""
import logging

from prefect import flow, task

from analyzer import analyze_transactions
from categorizer import categorize_transactions
from nlp_processor import process_nlp_queries
from pdf_parser import process_pdf_statements
from storyteller import generate_stories
from timeline import build_timeline
from visualizer import generate_visualizations

logging.basicConfig(level=logging.INFO)

@task
def parse_pdfs_task(input_dir: str, output_csv: str) -> None:
    """Parse PDFs to transactions."""
    process_pdf_statements(input_dir, output_csv)

@task
def build_timeline_task(input_csv: str, output_csv: str) -> None:
    """Build transaction timeline."""
    build_timeline(input_csv, output_csv)

@task
def categorize_transactions_task(input_csv: str, output_csv: str) -> None:
    """Categorize transactions."""
    categorize_transactions(input_csv, output_csv)

@task
def analyze_transactions_task(input_csv: str, output_dir: str) -> dict:
    """Analyze transactions."""
    return analyze_transactions(input_csv, output_dir)

@task
def generate_visualizations_task(input_csv: str, output_dir: str) -> dict:
    """Generate visualizations."""
    return generate_visualizations(input_csv, output_dir)

@task
def generate_stories_task(input_csv: str, output_file: str) -> list:
    """Generate financial stories."""
    return generate_stories(input_csv, output_file)

@task
def process_nlp_queries_task(input_csv: str, query: str, output_file: str) -> str:
    """Process NLP query."""
    return process_nlp_queries(input_csv, query, output_file)

@flow(name="Financial_Analysis_Pipeline")
def financial_analysis_pipeline(input_dir: str = "data/input/", query: str = "Restaurant spending last month") -> dict:
    """Orchestrate financial analysis tasks.
    
    Args:
        input_dir: Directory with PDFs.
        query: NLP query to process.
    
    Returns:
        Dictionary with task results.

    """
    logging.info("Starting financial analysis pipeline")

    # Define paths
    transactions_csv = "data/output/all_transactions.csv"
    timeline_csv = "data/output/timeline.csv"
    categorized_csv = "data/output/categorized.csv"
    analysis_dir = "data/output/analysis"
    charts_dir = "data/output/charts"
    stories_file = "data/output/stories.txt"
    nlp_file = "data/output/nlp_response.txt"

    # Run tasks
    parse_pdfs_task(input_dir, "data/output/")
    build_timeline_task(transactions_csv, timeline_csv)
    categorize_transactions_task(timeline_csv, categorized_csv)
    analysis = analyze_transactions_task(categorized_csv, analysis_dir)
    visualizations = generate_visualizations_task(categorized_csv, charts_dir)
    stories = generate_stories_task(categorized_csv, stories_file)
    nlp_response = process_nlp_queries_task(categorized_csv, query, nlp_file)

    return {
        "analysis": analysis,
        "visualizations": visualizations,
        "stories": stories,
        "nlp_response": nlp_response,
    }

if __name__ == "__main__":
    results = financial_analysis_pipeline()
    print(results)
