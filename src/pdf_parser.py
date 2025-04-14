"""PDF Parsing and Data Extraction Logic.

This module contains functions to parse bank statement PDFs and extract raw transaction data.
Key functionalities include:
- Reading and processing ICICI Bank PDF statements.
- Extracting structured data such as dates, amounts, and descriptions.
- Handling multi-page PDFs and unstructured text.
- Cleaning and formatting extracted data for further processing.
"""

import glob
import json
import os
import re

import pandas as pd
import pdfplumber


def parse_hdfc_statement(pdf_path):
    """Parse HDFC bank statement PDF and return transaction data as a DataFrame"""
    transactions = []

    # Open the PDF
    with pdfplumber.open(pdf_path) as pdf:
        # Process each page
        for page in pdf.pages:
            text = page.extract_text()

            # Skip the header and footer pages
            if "Date Narration Chq./Ref.No. Value Dt" not in text:
                continue

            # Split the text into lines
            lines = text.split("\n")

            # Find the lines containing transaction data
            for line in lines:
                # Look for date pattern at the beginning (DD/MM/YY)
                date_match = re.match(r"(\d{2}/\d{2}/\d{2})\s+(.*)", line)

                if date_match:
                    date = date_match.group(1)
                    rest_of_line = date_match.group(2)

                    # Handle different line formats - some lines might continue from previous transaction
                    if re.search(r"\d{2}/\d{2}/\d{2}\s+", rest_of_line):
                        # This line contains another transaction, skip for now
                        continue

                    # Pattern to extract transaction details
                    # Looking for reference number, value date, withdrawal, deposit, and closing balance
                    pattern = r"(.*?)\s+(\d+|[A-Z0-9]+)\s+(\d{2}/\d{2}/\d{2})(?:\s+([0-9,.]+))?(?:\s+([0-9,.]+))?\s+([0-9,.]+)$"
                    match = re.search(pattern, line)

                    if match:
                        narration = match.group(1).strip()
                        ref_no = match.group(2)
                        value_date = match.group(3)
                        withdrawal = match.group(4) if match.group(4) else ""
                        deposit = match.group(5) if match.group(5) else ""
                        closing_balance = match.group(6)

                        # Clean up numeric values
                        withdrawal = (
                            withdrawal.replace(",", "") if withdrawal else "0.00"
                        )
                        deposit = deposit.replace(",", "") if deposit else "0.00"
                        closing_balance = (
                            closing_balance.replace(",", "")
                            if closing_balance
                            else "0.00"
                        )

                        transactions.append(
                            {
                                "Date": date,
                                "Narration": narration,
                                "Reference Number": ref_no,
                                "Value Date": value_date,
                                "Withdrawal (INR)": float(withdrawal),
                                "Deposit (INR)": float(deposit),
                                "Closing Balance (INR)": float(closing_balance),
                            },
                        )
                    elif transactions:
                        transactions[-1]["Narration"] += " " + line.strip()

    # Create DataFrame from transactions
    df = pd.DataFrame(transactions)
    return df


def enhanced_parse_hdfc_statement(pdf_path):
    """Enhanced parsing for HDFC bank statements with improved pattern matching"""
    all_transactions = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            lines = text.split("\n")

            # Skip header and footer pages
            if not any(
                line.startswith(
                    (
                        "Date Narration",
                        "01/",
                        "02/",
                        "03/",
                        "04/",
                        "05/",
                        "06/",
                        "07/",
                        "08/",
                        "09/",
                        "10/",
                        "11/",
                        "12/",
                    ),
                )
                for line in lines
            ):
                continue

            # Find transaction lines
            for i, line in enumerate(lines):
                # Check if line starts with a date (DD/MM/YY)
                if re.match(r"^(\d{2}/\d{2}/\d{2})", line):
                    # Extract transaction components
                    parts = line.split()

                    # Need at least date, value_date, and closing_balance
                    if len(parts) < 4:
                        continue

                    date = parts[0]

                    # Find value date (which is also in date format)
                    value_date_idx = -1
                    for j, part in enumerate(parts[1:], 1):
                        if re.match(r"^\d{2}/\d{2}/\d{2}$", part):
                            value_date_idx = j
                            break

                    if value_date_idx == -1:
                        continue

                    value_date = parts[value_date_idx]

                    # Reference number is typically right before value date
                    ref_no = parts[value_date_idx - 1] if value_date_idx > 1 else ""

                    # Narration is between date and ref_no
                    narration = (
                        " ".join(parts[1 : value_date_idx - 1])
                        if value_date_idx > 2
                        else ""
                    )

                    # Find numeric values after value date
                    numeric_parts = []
                    for part in parts[value_date_idx + 1 :]:
                        if re.match(r"^[\d,.]+$", part):
                            numeric_parts.append(part.replace(",", ""))

                    # The last numeric value is the closing balance
                    if len(numeric_parts) >= 1:
                        closing_balance = numeric_parts[-1]

                        # If there are additional numeric values, they are withdrawal and deposit
                        withdrawal = "0.00"
                        deposit = "0.00"

                        if len(numeric_parts) == 3:
                            withdrawal = numeric_parts[0]
                            deposit = numeric_parts[1]
                        elif len(numeric_parts) == 2:
                            # Determine if it's a withdrawal or deposit
                            if float(numeric_parts[0]) > float(numeric_parts[1]):
                                withdrawal = numeric_parts[0]
                            else:
                                deposit = numeric_parts[0]

                        transaction = {
                            "Date": date,
                            "Narration": narration,
                            "Reference Number": ref_no,
                            "Value Date": value_date,
                            "Withdrawal (INR)": float(withdrawal),
                            "Deposit (INR)": float(deposit),
                            "Closing Balance (INR)": float(closing_balance),
                        }

                        all_transactions.append(transaction)

    return pd.DataFrame(all_transactions)


def extract_transactions_regex(pdf_path):
    """Extract transactions using regex patterns designed specifically for HDFC format"""
    transactions = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()

            # Find transaction lines with regex
            # Pattern: date, narration, ref_no, value_date, withdrawal?, deposit?, closing_balance
            pattern = r"(\d{2}/\d{2}/\d{2})\s+(.*?)\s+(\w+)\s+(\d{2}/\d{2}/\d{2})\s+(?:([0-9,.]+)\s+)?(?:([0-9,.]+)\s+)?([0-9,.]+)"

            for match in re.finditer(pattern, text):
                date = match.group(1)
                narration = match.group(2).strip()
                ref_no = match.group(3)
                value_date = match.group(4)

                # Group 5 and 6 could be withdrawal or deposit
                amount1 = match.group(5)
                amount2 = match.group(6)
                closing_balance = match.group(7)

                # Determine which amount is withdrawal and which is deposit
                withdrawal = "0.00"
                deposit = "0.00"

                if amount1 and amount2:
                    withdrawal = amount1
                    deposit = amount2
                elif amount1 and not amount2:
                    # If only one amount exists, determine if it's withdrawal or deposit
                    # by checking if the balance decreases
                    if transactions:
                        last_balance = float(transactions[-1]["Closing Balance (INR)"])
                        current_balance = float(closing_balance.replace(",", ""))

                        if current_balance < last_balance:
                            withdrawal = amount1
                        else:
                            deposit = amount1

                # Clean amounts
                withdrawal = withdrawal.replace(",", "") if withdrawal else "0.00"
                deposit = deposit.replace(",", "") if deposit else "0.00"
                closing_balance = closing_balance.replace(",", "")

                transactions.append(
                    {
                        "Date": date,
                        "Narration": narration,
                        "Reference Number": ref_no,
                        "Value Date": value_date,
                        "Withdrawal (INR)": float(withdrawal),
                        "Deposit (INR)": float(deposit),
                        "Closing Balance (INR)": float(closing_balance),
                    },
                )

    return pd.DataFrame(transactions)


def extract_customer_info(pdf_path):
    """Extract customer information from the HDFC bank statement with improved patterns"""
    customer_info = {
        "name": "",
        "email": "",
        "account_number": "",
        "city": "",
        "state": "",
    }

    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        # Check first few pages for customer info
        for page_idx in range(min(3, len(pdf.pages))):
            full_text += pdf.pages[page_idx].extract_text() + "\n"

        # Extract account number - looking for patterns like "Account No : 50100158077633"
        account_pattern = r"Account No\s*:\s*(\d+)"
        account_match = re.search(account_pattern, full_text)
        if account_match:
            customer_info["account_number"] = account_match.group(1)

        # Extract email - improved pattern that handles different email formats
        email_pattern = r"Email\s*:\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})"
        email_match = re.search(email_pattern, full_text)
        if email_match:
            customer_info["email"] = email_match.group(1)

        # Extract name - more robust pattern that finds the name on the statement
        # Looking for patterns like "MR. ANUP DUBEY" at the beginning of a line
        name_pattern = r"(?:^|\n)((?:MR|MRS|MS|DR)\.?\s+[A-Z][A-Z\s]+)"
        name_match = re.search(name_pattern, full_text)
        if name_match:
            customer_info["name"] = name_match.group(1).strip()

        # Extract city and state - improved pattern that handles different formats
        city_pattern = r"(?:City|CITY)\s*:\s*([A-Z]+)"
        city_match = re.search(city_pattern, full_text)
        if city_match:
            customer_info["city"] = city_match.group(1)

        state_pattern = r"(?:State|STATE)\s*:\s*([A-Z]+)"
        state_match = re.search(state_pattern, full_text)
        if state_match:
            customer_info["state"] = state_match.group(1)

        # If name wasn't found with the previous pattern, try another approach
        if not customer_info["name"]:
            # Look for customer name in address section
            address_pattern = r"(?:A/C OPEN DATE|JOINT HOLDERS).*?\n(.*?)\n"
            address_match = re.search(address_pattern, full_text)
            if address_match:
                customer_info["name"] = address_match.group(1).strip()

    # Fallback: If email is found but name isn't, extract name from email
    if not customer_info["name"] and customer_info["email"]:
        email_name = customer_info["email"].split("@")[0]
        if email_name:
            # Convert email name like "anupdubey788" to "Anup Dubey"
            name_parts = re.findall(r"[a-zA-Z]+", email_name)
            formatted_name = " ".join(part.capitalize() for part in name_parts)
            customer_info["name"] = formatted_name

    # Clean up data
    if customer_info["name"]:
        # Remove any extra spaces, line breaks, etc.
        customer_info["name"] = re.sub(r"\s+", " ", customer_info["name"]).strip()

    return customer_info


def process_pdf_statements(folder_path, output_folder):
    """Process 1 to 10 PDF statements for one person, saving only combined CSV and one JSON to output_folder"""
    # Normalize folder paths
    folder_path = os.path.normpath(folder_path)
    output_folder = os.path.normpath(output_folder)

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Debug: Print folder paths
    print(f"Searching for PDFs in: {folder_path}")
    print(f"Will save outputs to: {output_folder}")

    # Find all PDF files (case-insensitive for .pdf or .PDF)
    pdf_files = glob.glob(os.path.join(folder_path, "*.[pP][dD][fF]"))

    # Debug: Print found files
    print(f"Found {len(pdf_files)} PDF files: {pdf_files}")

    if not pdf_files:
        print(
            f"No PDF files found in {folder_path}. Please check the folder and file extensions.",
        )
        return [], []

    # Limit to 10 files
    pdf_files = pdf_files[:10]
    print(f"Processing {len(pdf_files)} PDF files (limited to 10).")

    all_transactions = []
    customer_info = None

    for idx, pdf_file in enumerate(pdf_files):
        file_name = os.path.basename(pdf_file)
        print(f"\nProcessing: {file_name}")

        # Extract customer info from first PDF only
        if idx == 0:
            customer_info = extract_customer_info(pdf_file)
            customer_info["pdf_files"] = [file_name]
        else:
            customer_info["pdf_files"].append(file_name)

        # Try parsing methods
        df = enhanced_parse_hdfc_statement(pdf_file)

        if len(df) == 0:
            print("Enhanced method failed, trying regex method...")
            df = extract_transactions_regex(pdf_file)

        if len(df) == 0:
            print("Regex method failed, trying basic method...")
            df = parse_hdfc_statement(pdf_file)

        if len(df) > 0:
            df["Source_File"] = file_name
            df["Account_Number"] = customer_info["account_number"]
            all_transactions.append(df)
            print(f"Successfully extracted {len(df)} transactions from {file_name}")
        else:
            print(f"Failed to extract transactions from {file_name}")

    # Save combined transactions
    all_customer_info = [customer_info] if customer_info else []
    if all_transactions:
        combined_df = pd.concat(all_transactions, ignore_index=True)
        combined_csv = os.path.join(output_folder, "all_transactions.csv")
        combined_df.to_csv(combined_csv, index=False)
        print(
            f"\nCombined {len(combined_df)} transactions from {len(all_transactions)} files into {combined_csv}",
        )

    # Save customer info
    if all_customer_info:
        combined_json = os.path.join(output_folder, "all_customers_info.json")
        with open(combined_json, "w") as f:
            json.dump(all_customer_info, f, indent=4)
        print(f"Saved information for 1 customer to {combined_json}")

    return all_customer_info, all_transactions


def main():
    """Main function to process PDF statements for one person"""
    # Default folder paths (relative)
    default_input_path = "../Customer-Financial-Health-Analyzer/data/input"
    default_output_path = "../Customer-Financial-Health-Analyzer/data/output"

    # Resolve to absolute paths
    default_input_path = os.path.abspath(default_input_path)
    default_output_path = os.path.abspath(default_output_path)

    folder_path = input(
        f"Enter folder path containing 1-10 PDF statements for one person (default: {default_input_path}): ",
    ).strip()

    # Use default input path if empty
    if not folder_path:
        folder_path = default_input_path
    else:
        folder_path = os.path.abspath(folder_path)

    # Verify input folder
    if not os.path.isdir(folder_path):
        print(
            f"Error: Input folder '{folder_path}' does not exist or is not a directory.",
        )
        return

    # Use default output path
    output_folder = default_output_path

    # Verify output folder
    try:
        os.makedirs(output_folder, exist_ok=True)
        test_file = os.path.join(output_folder, ".test_write")
        with open(test_file, "w") as f:
            f.write("")
        os.remove(test_file)
    except Exception as e:
        print(f"Error: Cannot write to output folder '{output_folder}'. Reason: {e}")
        return

    # Process PDFs
    customer_info_list, transaction_dfs = process_pdf_statements(
        folder_path, output_folder,
    )

    # Print summary
    if customer_info_list:
        print("\nSummary of processed statements:")
        customer_info = customer_info_list[0]
        print("\nCustomer:")
        print(f"Name: {customer_info['name']}")
        print(f"Account Number: {customer_info['account_number']}")
        print(f"Email: {customer_info['email']}")
        print(f"City: {customer_info['city']}")
        print(f"State: {customer_info['state']}")
        print(f"PDF Files: {', '.join(customer_info['pdf_files'])}")

        if transaction_dfs:
            combined_df = pd.concat(transaction_dfs, ignore_index=True)
            print(f"Total Transactions: {len(combined_df)}")
            print(
                f"Date Range: {combined_df['Date'].min()} to {combined_df['Date'].max()}",
            )
            print(f"Total Deposits: {combined_df['Deposit (INR)'].sum():.2f} INR")
            print(f"Total Withdrawals: {combined_df['Withdrawal (INR)'].sum():.2f} INR")
    else:
        print(
            "\nNo statements were processed. Please verify the folder contains 1-10 PDF files.",
        )


if __name__ == "__main__":
    main()
