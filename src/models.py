"""Pydantic models for data validation and serialization."""

import re
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, constr, validator


# Utility models
class FilePath(BaseModel):
    path: Path

    @validator("path")
    def validate_path(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Path does not exist: {v}")
        return v

class OutputPath(BaseModel):
    path: Path

    @validator("path")
    def validate_output_path(cls, v: Path) -> Path:
        v.parent.mkdir(parents=True, exist_ok=True)
        return v

class NonEmptyStr(BaseModel):
    value: constr(min_length=1, strip_whitespace=True)  # type: ignore

# Transaction models
class Transaction(BaseModel):
    Date: str
    Narration: str
    Reference_Number: str | None = Field(default="", alias="Reference Number")
    Value_Date: str | None = Field(default="", alias="Value Date")
    Withdrawal_INR: float = Field(default=0.0, alias="Withdrawal (INR)")
    Deposit_INR: float = Field(default=0.0, alias="Deposit (INR)")
    Closing_Balance_INR: float = Field(default=0.0, alias="Closing Balance (INR)")
    Source_File: str | None = ""

    @validator("Date", "Value_Date")
    def validate_date_format(cls, v: str) -> str:
        if v and not re.match(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", v):
            raise ValueError(f"Invalid date format: {v}")
        return v

class CategorizedTransaction(Transaction):
    parsed_date: str
    category: str
    month: str | None = None
    weekday: int | None = None
    is_weekend: bool | None = None
    day: int | None = None
    time_of_month: str | None = None

class CustomerInfo(BaseModel):
    name: str | None = ""
    email: str | None = ""
    account_number: str | None = ""
    city: str | None = ""
    state: str | None = ""
    pdf_files: list[str] = []

# PDF Parser models
class PdfProcessingInput(BaseModel):
    folder_path: Path
    output_csv: Path

class PdfProcessingOutput(BaseModel):
    customer_info: list[CustomerInfo]
    transactions: list[list[Transaction]]

# Timeline models
class TimelineInput(BaseModel):
    transactions_csv: Path
    output_csv: Path

class TimelineOutput(BaseModel):
    transactions: list[CategorizedTransaction]

# Categorizer models
class CategorizerInput(BaseModel):
    timeline_csv: Path
    output_csv: Path

class CategorizerOutput(BaseModel):
    transactions: list[CategorizedTransaction]

# Analyzer models
class AnalyzerInput(BaseModel):
    input_csv: Path
    output_dir: Path

class Pattern(BaseModel):
    description: str

class Fee(BaseModel):
    parsed_date: datetime
    Narration: str
    amount: float
    type: str
    fee_type: str
    category: str

class Recurring(BaseModel):
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
    parsed_date: datetime | None = None
    Narration: str
    amount: float
    type: str
    severity: str
    category: str
    detection_method: str

class CashFlow(BaseModel):
    month: str
    income: float
    expenses: float
    net_cash_flow: float
    status: str

class AccountOverview(BaseModel):
    total_balance: float
    monthly_income: float
    monthly_expense: float
    balance_percentage: float
    income_percentage: float
    expense_percentage: float

class AnalyzerOutput(BaseModel):
    patterns: list[Pattern]
    fees: list[Fee]
    recurring: list[Recurring]
    anomalies: list[Anomaly]
    cash_flow: list[CashFlow]
    account_overview: AccountOverview

# Visualizer models
class VisualizerInput(BaseModel):
    input_csv: Path
    output_dir: Path

class SpendingTrends(BaseModel):
    labels: list[str]
    expenses: list[float]
    budget: list[float]

class ExpenseBreakdown(BaseModel):
    categories: list[str]
    percentages: list[float]

class VisualizerOutput(BaseModel):
    spending_trends: SpendingTrends
    expense_breakdown: ExpenseBreakdown
    account_overview: AccountOverview

# Storyteller models
class StorytellerInput(BaseModel):
    input_csv: Path
    output_file: Path

class StorytellerOutput(BaseModel):
    stories: list[str]

# NLP Processor models
class NlpProcessorInput(BaseModel):
    input_csv: Path
    query: str
    output_file: Path
    visualization_file: Path | None = None

class VisualizationData(BaseModel):
    type: str
    data: list[list[str | float]]
    columns: list[str]
    title: str

class NlpProcessorOutput(BaseModel):
    text_response: str
    visualization: VisualizationData | None = None

# Workflow models
class FinancialPipelineInput(BaseModel):
    input_dir: Path = Path("data/input")
    query: str = "give me a summary for Expense loan in January 2016"

class FinancialPipelineOutput(BaseModel):
    analysis: AnalyzerOutput
    visualizations: VisualizerOutput
    stories: list[str]
    nlp_response: str

# Financial Memory models
class QueryRecord(BaseModel):
    query: str
    result: str
    timestamp: str

class FinancialMemoryState(BaseModel):
    queries: list[QueryRecord]
    context: dict[str, str]
