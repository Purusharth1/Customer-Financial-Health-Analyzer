"""PDF Parsing and Data Extraction Logic.

This module contains functions to parse bank statement PDFs and extract raw
transaction data.
Key functionalities include:
- Reading and processing HDFC Bank PDF statements.
- Extracting structured data such as dates, amounts, and descriptions.
- Handling multi-page PDFs and unstructured text.
- Cleaning and formatting extracted data for further processing.
"""

import json
import re
import sys
from pathlib import Path

import mlflow
import pandas as pd
import pdfplumber
import PyPDF2
from tabula import read_pdf

sys.path.append(str(Path(__file__).parent.parent))
from src.models import (
    CustomerInfo,
    PdfProcessingInput,
    PdfProcessingOutput,
    Transaction,
)
from src.utils import setup_mlflow


def identify_column_name(col: str) -> str:
    """Identify column name based on common patterns."""
    col_lower = str(col).lower()
    mappings = {
        "date": ["date", "dt"],
        "narration": ["narration", "particulars", "description"],
        "reference number": ["ref", "chq", "reference"],
        "value date": ["value", "val dt", "val"],
        "withdrawal (inr)": ["debit", "withdrawal", "dr", "with"],
        "deposit (inr)": ["credit", "deposit", "cr", "dep"],
        "closing balance (inr)": ["balance", "bal"],
    }
    for key, terms in mappings.items():
        if any(term in col_lower for term in terms):
            return key
    return str(col)


def process_transaction_row(row: pd.Series, table_columns: list[str]) -> dict:
    """Process a single row of the table to extract transaction data."""
    transaction = {}
    for col in table_columns:
        value = row[col]
        if (col in ["Date", "Value Date"] and pd.notna(value)) or (
            col in ["Narration", "Reference Number"] and pd.notna(value)
        ):
            transaction[col] = str(value).strip()
        elif col in [
            "Withdrawal (INR)",
            "Deposit (INR)",
            "Closing Balance (INR)",
        ] and pd.notna(value):
            if isinstance(value, str):
                value = (
                    value.replace(",", "").replace("Dr", "").replace("Cr", "").strip()
                )
            try:
                transaction[col] = float(value) if value and value != "" else 0.0
            except (ValueError, TypeError):
                transaction[col] = 0.0
    return transaction


def validate_and_clean_transaction(transaction: dict) -> bool:
    """Validate and clean a transaction.

    Returns True if the transaction is valid, False otherwise.
    """
    if not (
        "Date" in transaction
        and transaction["Date"]
        and re.match(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", transaction["Date"])
    ):
        return False

    required_fields = [
        "Narration",
        "Reference Number",
        "Value Date",
        "Withdrawal (INR)",
        "Deposit (INR)",
        "Closing Balance (INR)",
    ]
    for field in required_fields:
        if field not in transaction:
            transaction[field] = 0.0 if field.endswith("(INR)") else ""

    if transaction["Withdrawal (INR)"] > 0 and transaction["Deposit (INR)"] > 0:
        if transaction["Withdrawal (INR)"] > transaction["Deposit (INR)"]:
            transaction["Deposit (INR)"] = 0.0
        else:
            transaction["Withdrawal (INR)"] = 0.0

    return True


def extract_transactions_tabula(pdf_path: str) -> pd.DataFrame:
    """Extract transactions using tabula-py for handling tabular PDF data."""
    try:
        tables = read_pdf(
            pdf_path,
            pages="all",
            multiple_tables=True,
            guess=True,
            lattice=True,
            stream=True,
        )

        if not tables:
            return pd.DataFrame()

        transactions: list[dict] = []
        min_columns = 5

        for table in tables:
            if len(table.columns) >= min_columns:
                column_names = [identify_column_name(col) for col in table.columns]

                if "Date" in column_names and (
                    "Withdrawal (INR)" in column_names
                    or "Deposit (INR)" in column_names
                ):
                    table.columns = column_names

                    for _, row in table.iterrows():
                        transaction = process_transaction_row(row, table.columns)
                        if validate_and_clean_transaction(transaction):
                            transactions.append(transaction)

        return pd.DataFrame(transactions)
    except ValueError:
        return pd.DataFrame()


def extract_text_better(pdf_path: str | Path) -> str:
    """Extract text from PDF using both PyPDF2 and pdfplumber for better results."""
    full_text = ""
    pdf_path = str(pdf_path)

    try:
        with Path(pdf_path).open("rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                full_text += page.extract_text() + "\n\n"
    except (OSError, ValueError, TypeError):
        pass

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                full_text += page.extract_text() + "\n\n"
    except (OSError, ValueError, TypeError):
        pass

    return full_text


MIN_AMOUNTS_FOR_WITHDRAWAL_DEPOSIT: int = 3
TWO_AMOUNTS: int = 2


def extract_date_and_rest(line: str) -> tuple[str | None, str]:
    """Extract date and remaining text from a line."""
    date_match = re.match(r"^(\d{2}/\d{2}/\d{2})", line)
    if date_match:
        date = date_match.group(1)
        rest_of_line = line[len(date) :].strip()
        return date, rest_of_line
    return None, line


def extract_value_date_and_ref_no(rest_of_line: str, date: str) -> tuple[str, str]:
    """Extract value date and reference number from the rest of the line."""
    value_date_pattern = r"(\d{2}/\d{2}/\d{2})"
    value_dates = re.findall(value_date_pattern, rest_of_line)
    value_date = value_dates[0] if value_dates else date

    ref_no = ""
    ref_pattern = r"([A-Z0-9]{6,})"
    value_date_pos = rest_of_line.find(value_date)
    if value_date_pos > 0:
        ref_text = rest_of_line[:value_date_pos]
        ref_matches = re.findall(ref_pattern, ref_text)
        if ref_matches:
            ref_no = ref_matches[-1]

    return value_date, ref_no


def extract_narration(rest_of_line: str, value_date: str, ref_no: str) -> str:
    """Extract and clean narration from the rest of the line."""
    narration = rest_of_line
    if value_date:
        narration = narration.replace(value_date, "").strip()
    if ref_no:
        narration = narration.replace(ref_no, "").strip()
    return narration


def extract_amounts(rest_of_line: str) -> tuple[list[str], list[str]]:
    """Extract amounts and their Cr/Dr flags."""
    amount_pattern = r"(\d{1,3}(?:,\d{3})*\.\d{2})\s*(Cr|Dr)?"
    amount_matches = re.findall(amount_pattern, rest_of_line)
    amounts = [amt[0] for amt in amount_matches]
    cr_dr_flags = [amt[1] for amt in amount_matches]
    return amounts, cr_dr_flags


def determine_debit_credit(cr_dr_flags: list[str], narration: str) -> str:
    """Determine transaction type based on flags and narration."""
    if "Dr" in cr_dr_flags:
        return "debit"
    if "Cr" in cr_dr_flags:
        return "credit"

    lower_narration = narration.lower()
    is_debit = any(
        keyword in lower_narration
        for keyword in [
            "debit",
            "withdrawal",
            "purchase",
            "payment",
            "fee",
            "charge",
            "outward",
            "paid",
            "dr",
            "by transfer",
            "bill payment",
            "emi",
            "neft-out",
            "rtgs-out",
            "pos debit",
            "upi payment",
            "charges",
            "service",
        ]
    )
    is_credit = any(
        keyword in lower_narration
        for keyword in [
            "credit",
            "deposit",
            "salary",
            "interest",
            "refund",
            "cashback",
            "inward",
            "received",
            "cr",
            "to account",
            "return",
            "neft-in",
            "rtgs-in",
            "imps-in",
            "upi credit",
            "reversal",
            "interest credited",
            "gmitsa",
        ]
    )

    if is_debit:
        return "debit"
    if is_credit:
        return "credit"
    return "unknown"


def process_amounts(
    amounts: list[str],
    transaction_type: str,
    transactions: list[dict],
) -> tuple[float, float, float]:
    """Process amounts to determine withdrawal, deposit, and closing balance."""
    cleaned_amounts = [round(float(amt.replace(",", "")), 2) for amt in amounts]
    withdrawal = 0.0
    deposit = 0.0
    closing_balance = round(cleaned_amounts[-1], 2) if cleaned_amounts else 0.0

    if cleaned_amounts:
        if len(cleaned_amounts) >= MIN_AMOUNTS_FOR_WITHDRAWAL_DEPOSIT:
            withdrawal = (
                round(cleaned_amounts[-3], 2) if transaction_type == "debit" else 0.0
            )
            deposit = (
                round(cleaned_amounts[-2], 2) if transaction_type == "credit" else 0.0
            )
        elif len(cleaned_amounts) == TWO_AMOUNTS:
            if transaction_type == "debit":
                withdrawal = round(cleaned_amounts[0], 2)
            elif transaction_type == "credit" or not transactions:
                deposit = round(cleaned_amounts[0], 2)
            else:
                prev_balance = transactions[-1]["Closing Balance (INR)"]
                if closing_balance > prev_balance:
                    deposit = round(closing_balance - prev_balance, 2)
                else:
                    withdrawal = round(prev_balance - closing_balance, 2)

    return withdrawal, deposit, closing_balance


def clean_narration(narration: str, amounts: list[str]) -> str:
    """Clean narration by removing amounts and extra spaces."""
    for amt in amounts:
        narration = narration.replace(amt, "").strip()
    return re.sub(r"\s{2,}", " ", narration).strip()


def parse_hdfc_statement_improved(pdf_path: str) -> pd.DataFrame:
    """Parse HDFC bank statement with advanced text extraction and pattern matching."""
    full_text = extract_text_better(pdf_path)
    transactions = []

    sections = re.split(r"\n{3,}", full_text)

    for section in sections:
        lines = section.split("\n")

        for _i, line in enumerate(lines):
            date, rest_of_line = extract_date_and_rest(line)
            if date:
                value_date, ref_no = extract_value_date_and_ref_no(rest_of_line, date)
                narration = extract_narration(rest_of_line, value_date, ref_no)
                amounts, cr_dr_flags = extract_amounts(rest_of_line)
                transaction_type = determine_debit_credit(cr_dr_flags, narration)
                withdrawal, deposit, closing_balance = process_amounts(
                    amounts,
                    transaction_type,
                    transactions,
                )
                narration = clean_narration(narration, amounts)

                if date and (withdrawal > 0 or deposit > 0 or closing_balance > 0):
                    if withdrawal and deposit:
                        if withdrawal > deposit:
                            deposit = 0.0
                        else:
                            withdrawal = 0.0

                    transactions.append(
                        {
                            "Date": date,
                            "Narration": narration,
                            "Reference Number": ref_no,
                            "Value Date": value_date,
                            "Withdrawal (INR)": withdrawal,
                            "Deposit (INR)": deposit,
                            "Closing Balance (INR)": closing_balance,
                        },
                    )

    return pd.DataFrame(transactions)


def extract_customer_info(pdf_path: str) -> dict[str, str]:
    """Extract customer information from the HDFC bank statement."""
    customer_info = {
        "name": "",
        "email": "",
        "account_number": "",
        "city": "",
        "state": "",
    }

    full_text = extract_text_better(pdf_path)

    account_pattern = r"Account No\s*:?\s*(\d{10,})"
    account_match = re.search(account_pattern, full_text)
    if account_match:
        customer_info["account_number"] = account_match.group(1)

    email_pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    email_match = re.search(email_pattern, full_text)
    if email_match:
        customer_info["email"] = email_match.group(0)

    name_pattern = r"(?:MR|MRS|MS|DR)\.?\s+([A-Z][A-Z\s]+)"
    name_match = re.search(name_pattern, full_text)
    if name_match:
        customer_info["name"] = name_match.group(0).strip()

    city_pattern = r"(?:City|CITY)\s*:?\s*([A-Z]+)"
    city_match = re.search(city_pattern, full_text)
    if city_match:
        customer_info["city"] = city_match.group(1)

    state_pattern = r"(?:State|STATE)\s*:?\s*([A-Z]+)"
    state_match = re.search(state_pattern, full_text)
    if state_match:
        customer_info["state"] = state_match.group(1)

    return customer_info


MIN_TRANSACTIONS_THRESHOLD = 5


def extract_pdf_files(folder_path: Path) -> list[Path]:
    """Find all PDF files in the folder."""
    return list(folder_path.rglob("*.[pP][dD][fF]"))


def validate_and_limit_pdfs(pdf_files: list[Path]) -> list[Path]:
    """Validate and limit the number of PDF files to 10."""
    if not pdf_files:
        return []
    return pdf_files[:10]


def process_single_pdf(
    pdf_file: Path,
    customer_info: dict,
    idx: int,
) -> tuple[pd.DataFrame, dict]:
    """Process a single PDF file, extracting transactions and updating customer info."""
    file_name = pdf_file.name

    if idx == 0:
        customer_info = extract_customer_info(str(pdf_file))
        customer_info["pdf_files"] = [file_name]
    elif customer_info:
        customer_info["pdf_files"].append(file_name)

    transactions_df = extract_transactions_tabula(str(pdf_file))

    if len(transactions_df) < MIN_TRANSACTIONS_THRESHOLD:
        transactions_df = parse_hdfc_statement_improved(str(pdf_file))

    if len(transactions_df) > 0:
        transactions_df["Source_File"] = file_name

        if (
            "Withdrawal (INR)" in transactions_df.columns
            and "Deposit (INR)" in transactions_df.columns
        ):
            for _, row in transactions_df.iterrows():
                if row["Withdrawal (INR)"] > 0 and row["Deposit (INR)"] > 0:
                    if row["Withdrawal (INR)"] > row["Deposit (INR)"]:
                        transactions_df.loc[_, "Deposit (INR)"] = 0.0
                    else:
                        transactions_df.loc[_, "Withdrawal (INR)"] = 0.0

    return transactions_df, customer_info


def standardize_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and remove duplicates."""
    # Map from various column names to standard names
    column_mapping = {
        "Reference_Number": "Reference Number",
        "Value_Date": "Value Date",
        "Withdrawal_INR": "Withdrawal (INR)",
        "Deposit_INR": "Deposit (INR)",
        "Closing_Balance_INR": "Closing Balance (INR)",
    }

    # Rename columns based on mapping
    dataframe = dataframe.rename(columns=column_mapping)

    # Select only unique columns (prefer standard names)
    standard_columns = [
        "Date", "Narration", "Reference Number", "Value Date",
        "Withdrawal (INR)", "Deposit (INR)", "Closing Balance (INR)", "Source_File",
    ]

    # Keep only standard columns that exist in the DataFrame
    existing_columns = [col for col in standard_columns if col in dataframe.columns]

    return dataframe[existing_columns]


def combine_transactions(all_transactions: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine all transaction DataFrames into one DataFrame with validation."""
    if not all_transactions:
        return pd.DataFrame()

    combined_df = pd.concat(all_transactions, ignore_index=True)

    # Standardize column names
    combined_df = standardize_column_names(combined_df)

    numeric_columns = ["Withdrawal (INR)", "Deposit (INR)", "Closing Balance (INR)"]
    for col in numeric_columns:
        if col in combined_df.columns:
            combined_df[col] = (
                pd.to_numeric(combined_df[col], errors="coerce").fillna(0.0)
            )
    required_columns = [
        "Date",
        "Narration",
        "Reference Number",
        "Value Date",
        "Withdrawal (INR)",
        "Deposit (INR)",
        "Closing Balance (INR)",
        "Source_File",
    ]
    for col in required_columns:
        if col not in combined_df.columns:
            combined_df[col] = 0.0 if col in numeric_columns else ""

    return combined_df


def save_combined_outputs(
    output_csv: Path,
    combined_df: pd.DataFrame,
    all_customer_info: list[CustomerInfo],
) -> None:
    """Save the combined transactions CSV and customer info JSON."""
    combined_df.to_csv(output_csv, index=False)

    json_path = output_csv.parent / "all_customers_info.json"
    with json_path.open("w") as f:
        json.dump([info.dict() for info in all_customer_info], f, indent=4)


def create_transaction_objects(dataframe: pd.DataFrame) -> list[Transaction]:
    """Create Transaction objects from DataFrame rows, handling column naming issues."""
    transactions = []
    column_mapping = {
        "Date": "Date",
        "Narration": "Narration",
        "Reference Number": "Reference Number",
        "Reference_Number": "Reference Number",
        "Value Date": "Value Date",
        "Value_Date": "Value Date",
        "Withdrawal (INR)": "Withdrawal (INR)",
        "Withdrawal_INR": "Withdrawal (INR)",
        "Deposit (INR)": "Deposit (INR)",
        "Deposit_INR": "Deposit (INR)",
        "Closing Balance (INR)": "Closing Balance (INR)",
        "Closing_Balance_INR": "Closing Balance (INR)",
        "Source_File": "Source_File",
    }

    for _, row in dataframe.iterrows():
        transaction_data = {}
        for column in dataframe.columns:
            standardized_col = column_mapping.get(column, column)
            if standardized_col in ["Withdrawal (INR)",
                                "Deposit (INR)", "Closing Balance (INR)"]:
                transaction_data[standardized_col] = (float(row[column])
                                                if pd.notna(row[column]) else 0.0)
            else:
                transaction_data[standardized_col] = row[column]

        try:
            transaction = Transaction(**transaction_data)
            transactions.append(transaction)
        except (ValueError, TypeError):
            # Log the error or handle it as needed
            pass

    return transactions


def process_pdf_statements(input_model: PdfProcessingInput) -> PdfProcessingOutput:
    """Process 1 to 10 PDF statements for one person, saving combined CSV and JSON."""
    folder_path = input_model.folder_path
    output_csv = input_model.output_csv

    setup_mlflow()
    with mlflow.start_run(run_name="PDF_Parsing"):
        mlflow.log_param("folder_path", folder_path)
        mlflow.log_param("output_csv", output_csv)

        folder_path = Path(folder_path)
        output_csv = Path(output_csv)

        output_csv.parent.mkdir(parents=True, exist_ok=True)

        pdf_files = extract_pdf_files(folder_path)
        mlflow.log_metric("pdf_files_found", len(pdf_files))

        pdf_files = validate_and_limit_pdfs(pdf_files)
        mlflow.log_metric("pdf_files_processed", len(pdf_files))

        if not pdf_files:
            mlflow.log_param("warning", "No PDF files found")
            return PdfProcessingOutput(customer_info=[], transactions=[])

        all_transactions_dfs = []
        customer_info = {}

        for idx, pdf_file in enumerate(pdf_files):
            transactions_df, customer_info = process_single_pdf(
                pdf_file,
                customer_info,
                idx,
            )
            if len(transactions_df) > 0:
                all_transactions_dfs.append(transactions_df)
                mlflow.log_metric(
                    f"transactions_extracted_pdf_{idx}",
                    len(transactions_df),
                )

        if all_transactions_dfs:
            # Combine all transaction DataFrames
            combined_df = combine_transactions(all_transactions_dfs)

            # Create customer info object
            all_customer_info = [CustomerInfo(**customer_info)] if customer_info else []

            # Save outputs
            save_combined_outputs(output_csv, combined_df, all_customer_info)

            # Group transactions by source file
            all_transactions = []
            grouped = combined_df.groupby("Source_File", dropna=False)
            for _source_file, group_df in grouped:
                group_df_clean = standardize_column_names(group_df)
                transactions = create_transaction_objects(group_df_clean)
                all_transactions.append(transactions)

            mlflow.log_metric("total_transactions_extracted", len(combined_df))
            mlflow.log_metric("customer_info_records", len(all_customer_info))

            if output_csv.exists():
                mlflow.log_artifact(str(output_csv))

            json_path = output_csv.parent / "all_customers_info.json"
            if json_path.exists():
                mlflow.log_artifact(str(json_path))

            return PdfProcessingOutput(
                customer_info=all_customer_info,
                transactions=all_transactions,
            )

        mlflow.log_param("warning", "No transactions extracted")
        return PdfProcessingOutput(customer_info=[], transactions=[])


def main() -> None:
    """Process PDF statements for one person."""
    default_input_path = Path(
        "../Customer-Financial-Health-Analyzer/data/input",
    ).resolve()
    default_output_path = Path(
        "../Customer-Financial-Health-Analyzer/data/output/all_transactions.csv",
    ).resolve()
    folder_path = input(
        "Enter folder path containing 1-10 PDF statements for one person "
        f"(default: {default_input_path}): ",
    ).strip()
    folder_path = Path(folder_path or default_input_path).resolve()
    if not folder_path.is_dir():
        # Use logging instead of print
        return

    output_csv = default_output_path
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Process PDFs
    input_model = PdfProcessingInput(folder_path=folder_path, output_csv=output_csv)
    process_pdf_statements(input_model)


if __name__ == "__main__":
    main()
