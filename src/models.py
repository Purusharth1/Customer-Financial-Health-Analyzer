"""Pydantic models for data validation and serialization.

This module defines Pydantic models used across the financial pipeline for
validating and serializing data related to transactions, categorization,
analysis, visualization, storytelling, NLP processing, and workflows.
"""
import re
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, constr, field_validator


# Utility models
class FilePath(BaseModel):
    """Model for validating existing file paths."""

    path: Path

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: Path) -> Path:
        """Validate that the file path exists."""
        path_not_exists_msg = f"Path does not exist: {v}"
        if not v.exists():
            raise ValueError(path_not_exists_msg)
        return v


class OutputPath(BaseModel):
    """Model for validating and creating output file paths."""

    path: Path

    @field_validator("path")
    @classmethod
    def validate_output_path(cls, v: Path) -> Path:
        """Ensure the parent directory of the output path exists."""
        v.parent.mkdir(parents=True, exist_ok=True)
        return v


class NonEmptyStr(BaseModel):
    """Model for non-empty strings with whitespace stripped."""

    value: constr(min_length=1, strip_whitespace=True)  # type: ignore[call-overload]

# Transaction models
class Transaction(BaseModel):
    """Model for raw transaction data extracted from PDFs."""

    Date: str
    Narration: str
    Reference_Number: str | None = Field(default="", alias="Reference Number")
    Value_Date: str | None = Field(default="", alias="Value Date")
    Withdrawal_INR: float = Field(default=0.0, alias="Withdrawal (INR)")
    Deposit_INR: float = Field(default=0.0, alias="Deposit (INR)")
    Closing_Balance_INR: float = Field(default=0.0, alias="Closing Balance (INR)")
    Source_File: str | None = ""

    @field_validator("Date", "Value_Date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Validate date format (e.g., DD/MM/YYYY or DD-MM-YY)."""
        invalid_date_msg = f"Invalid date format: {v}"
        if v and not re.match(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", v):
            raise ValueError(invalid_date_msg)
        return v


class CategorizedTransaction(Transaction):
    """Model for transactions with categorization and temporal metadata."""

    parsed_date: str
    category: str
    month: str | None = None
    weekday: int | None = None
    is_weekend: bool | None = None
    day: int | None = None
    time_of_month: str | None = None


class CustomerInfo(BaseModel):
    """Model for customer information extracted from PDFs."""

    name: str | None = ""
    email: str | None = ""
    account_number: str | None = ""
    city: str | None = ""
    state: str | None = ""
    pdf_files: list[str] = []


# PDF Parser models
class PdfProcessingInput(BaseModel):
    """Input model for PDF processing."""

    folder_path: Path
    output_csv: Path


class PdfProcessingOutput(BaseModel):
    """Output model for PDF processing results."""

    customer_info: list[CustomerInfo]
    transactions: list[list[Transaction]]


# Timeline models
class TimelineInput(BaseModel):
    """Input model for transaction timeline generation."""

    transactions_csv: Path
    output_csv: Path


class TimelineOutput(BaseModel):
    """Output model for transaction timeline results."""

    transactions: list[CategorizedTransaction]


# Categorizer models
class CategorizerInput(BaseModel):
    """Input model for transaction categorization."""

    timeline_csv: Path
    output_csv: Path


class CategorizerOutput(BaseModel):
    """Output model for categorized transactions."""

    transactions: list[CategorizedTransaction]


# Analyzer models
class AnalyzerInput(BaseModel):
    """Input model for financial analysis."""

    input_csv: Path
    output_dir: Path


class Pattern(BaseModel):
    """Model for spending patterns identified in analysis."""

    description: str


class Fee(BaseModel):
    """Model for fee-related transactions."""

    parsed_date: datetime
    Narration: str
    amount: float
    type: str
    fee_type: str
    category: str


class Recurring(BaseModel):
    """Model for recurring transactions."""

    narration: str
    amount: float | str
    frequency: str
    category: str
    type: str
    match_type: str
    regularity: str
    first_date: str
    last_date: str
    occurrence_count: int


class Anomaly(BaseModel):
    """Model for anomalous transactions."""

    parsed_date: datetime | None = None
    Narration: str
    amount: float
    type: str
    severity: str
    category: str
    detection_method: str


class CashFlow(BaseModel):
    """Model for monthly cash flow analysis."""

    month: str
    income: float
    expenses: float
    net_cash_flow: float
    status: str


class AccountOverview(BaseModel):
    """Model for account overview metrics."""

    total_balance: float
    monthly_income: float
    monthly_expense: float
    balance_percentage: float
    income_percentage: float
    expense_percentage: float


class AnalyzerOutput(BaseModel):
    """Output model for financial analysis results."""

    patterns: list[Pattern]
    fees: list[Fee]
    recurring: list[Recurring]
    anomalies: list[Anomaly]
    cash_flow: list[CashFlow]
    account_overview: AccountOverview


# Visualizer models
class VisualizerInput(BaseModel):
    """Input model for financial visualizations."""

    input_csv: Path
    output_dir: Path


class SpendingTrends(BaseModel):
    """Model for spending trends visualization."""

    labels: list[str]
    expenses: list[float]
    budget: list[float]


class ExpenseBreakdown(BaseModel):
    """Model for expense breakdown visualization."""

    categories: list[str]
    percentages: list[float]


class VisualizerOutput(BaseModel):
    """Output model for visualization results."""

    spending_trends: SpendingTrends
    expense_breakdown: ExpenseBreakdown
    account_overview: AccountOverview


# Storyteller models
class StorytellerInput(BaseModel):
    """Input model for financial storytelling."""

    input_csv: Path
    output_file: Path


class StorytellerOutput(BaseModel):
    """Output model for financial stories."""

    stories: list[str]


# NLP Processor models
class NlpProcessorInput(BaseModel):
    """Input model for NLP-based financial queries."""

    input_csv: Path
    query: str
    output_file: Path
    visualization_file: Path | None = None


class VisualizationData(BaseModel):
    """Model for visualization data in NLP responses."""

    type: str
    data: list[list[str | float]]
    columns: list[str]
    title: str


class NlpProcessorOutput(BaseModel):
    """Output model for NLP query responses."""

    text_response: str
    visualization: VisualizationData | None = None


# Workflow models
class FinancialPipelineInput(BaseModel):
    """Input model for the financial analysis pipeline."""

    input_dir: Path = Path("data/input")
    query: str = "give me a summary for Expense loan in January 2016"


class FinancialPipelineOutput(BaseModel):
    """Output model for the financial analysis pipeline."""

    analysis: AnalyzerOutput
    visualizations: VisualizerOutput
    stories: list[str]
    nlp_response: str


# Financial Memory models
class QueryRecord(BaseModel):
    """Model for storing query records in financial memory."""

    query: str
    result: str
    timestamp: str


class FinancialMemoryState(BaseModel):
    """Model for financial memory state with query history."""

    queries: list[QueryRecord]
    context: dict[str, str]
