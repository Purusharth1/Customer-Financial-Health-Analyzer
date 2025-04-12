import os
import re
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pdfplumber


def extract_hdfc_personal_details(text):
    """
    Extract personal details from HDFC bank statement text.
    
    Args:
        text (str): Extracted text from PDF
        
    Returns:
        dict: Personal details including name, address, account info
    """
    # Extract account holder name
    name_pattern = r"(MR\.|MS\.|MRS\.)\s+([A-Z\s\.]+)"
    name_match = re.search(name_pattern, text)
    title = name_match.group(1) if name_match else ""
    name = name_match.group(2).strip() if name_match else "Unknown"
    
    # Extract address
    address_lines = []
    lines = text.split('\n')
    name_index = -1
    
    # Find the name line
    for i, line in enumerate(lines):
        if name in line:
            name_index = i
            break
    
    # Extract address lines after name (typically 4-5 lines)
    if name_index != -1 and name_index + 5 < len(lines):
        address_lines = [lines[i].strip() for i in range(name_index + 1, name_index + 6) 
                        if lines[i].strip() and "JOINT HOLDERS" not in lines[i]]
    
    address = "\n".join(address_lines) if address_lines else "Address not found"
    
    # Extract account number - more robust pattern checking several formats
    account_number = "Unknown"
    account_patterns = [
        r"Account No\s*:?\s*(\d+)",
        r"Account No\s*:?\s*([0-9]+)\s+[A-Z]+",
        r"Account No\s*:?\s*([0-9]+\s*[A-Z]*)"
    ]
    
    for pattern in account_patterns:
        account_match = re.search(pattern, text)
        if account_match:
            account_number = account_match.group(1).strip()
            # Remove any whitespace in the account number
            account_number = re.sub(r'\s+', '', account_number)
            break
    
    # Try alternative method for account number by looking for patterns near "OTHER" or "SAVINGS"
    if account_number == "Unknown":
        for i, line in enumerate(lines):
            if "Account No" in line and i+1 < len(lines):
                parts = line.split(":")
                if len(parts) > 1:
                    acct_parts = parts[1].strip().split()
                    if acct_parts and acct_parts[0].isdigit():
                        account_number = acct_parts[0]
                        break
    
    # One more attempt - search for account number pattern in full text
    if account_number == "Unknown":
        possible_numbers = re.findall(r'5010\d{9}', text)
        if possible_numbers:
            account_number = possible_numbers[0]
    
    # Extract other fields
    city_pattern = r"City\s*:?\s*([A-Z]+\s*[0-9]*)"
    city_match = re.search(city_pattern, text)
    city = city_match.group(1).strip() if city_match else ""
    
    state_pattern = r"State\s*:?\s*([A-Z]+)"
    state_match = re.search(state_pattern, text)
    state = state_match.group(1).strip() if state_match else ""
    
    ifsc_pattern = r"RTGS/NEFT IFSC\s*:?\s*([A-Z0-9]+)"
    ifsc_match = re.search(ifsc_pattern, text)
    ifsc = ifsc_match.group(1) if ifsc_match else ""
    
    micr_pattern = r"MICR\s*:?\s*([0-9]+)"
    micr_match = re.search(micr_pattern, text)
    micr = micr_match.group(1) if micr_match else ""
    
    # Search for branch in a few patterns
    branch_patterns = [
        r"Account Branch\s*:?\s*(.*?)(?:\n|City)",
        r"Account Branch\s*:?\s*(.*?)(?:\nAddress)"
    ]
    branch = ""
    for pattern in branch_patterns:
        branch_match = re.search(pattern, text, re.DOTALL)
        if branch_match:
            branch = branch_match.group(1).strip()
            break
    
    email_pattern = r"Email\s*:?\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})"
    email_match = re.search(email_pattern, text)
    email = email_match.group(1) if email_match else ""
    
    cust_id_pattern = r"Cust ID\s*:?\s*(\d+)"
    cust_id_match = re.search(cust_id_pattern, text)
    customer_id = cust_id_match.group(1) if cust_id_match else ""
    
    open_date_pattern = r"A/C Open Date\s*:?\s*(\d{2}/\d{2}/\d{4})"
    open_date_match = re.search(open_date_pattern, text)
    open_date = open_date_match.group(1) if open_date_match else ""
    
    # Extract statement period
    period_pattern = r"From\s*:?\s*(\d{2}/\d{2}/\d{4})\s*To\s*:?\s*(\d{2}/\d{2}/\d{4})"
    period_match = re.search(period_pattern, text)
    statement_from = period_match.group(1) if period_match else ""
    statement_to = period_match.group(2) if period_match else ""
    
    # If we couldn't find statement dates, try the date range of transactions
    if not statement_from or not statement_to:
        date_matches = re.findall(r"\d{2}/\d{2}/\d{2}", text)
        if len(date_matches) >= 2:
            statement_from = date_matches[0]
            statement_to = date_matches[-1]
    
    # Create personal details dictionary
    personal_details = {
        "title": title,
        "name": name,
        "address": address,
        "city": city,
        "state": state,
        "account_number": account_number,
        "ifsc_code": ifsc,
        "micr_code": micr,
        "branch": branch,
        "email": email,
        "customer_id": customer_id,
        "account_opening_date": open_date,
        "statement_period": {
            "from": statement_from,
            "to": statement_to
        }
    }
    
    print(f"    -> Found Name: {name}, Account: {account_number}")
    
    return personal_details

def extract_hdfc_transactions(text):
    """
    Extract transaction data from HDFC bank statement text.
    
    Args:
        text (str): Extracted text from PDF
        
    Returns:
        pandas.DataFrame: DataFrame containing transaction data
    """
    # Split by lines
    lines = text.split('\n')
    
    # First locate the transaction section
    transaction_start_idx = -1
    transaction_end_idx = -1
    
    for i, line in enumerate(lines):
        if re.match(r"Date\s+Narration\s+Chq\./Ref\.No\.", line):
            transaction_start_idx = i + 1
        elif "STATEMENT SUMMARY" in line:
            transaction_end_idx = i
            break
    
    if transaction_start_idx == -1:
        print("    -> Could not find transaction header")
        return pd.DataFrame()
    
    if transaction_end_idx == -1:
        transaction_end_idx = len(lines)
    
    # Extract transaction data
    transactions = []
    
    # Improved regex pattern to match transaction rows
    transaction_pattern = re.compile(
        r'^(\d{2}/\d{2}/\d{2})\s+'  # Date
        r'(.+?)\s+'                  # Narration (non-greedy)
        r'([A-Z0-9]+)?\s*'           # Reference number (optional)
        r'(\d{2}/\d{2}/\d{2})?\s*'   # Value date (optional)
        r'(?:([\d,]+\.\d{2})\s+)?'   # Withdrawal amount (optional, non-capturing group)
        r'(?:([\d,]+\.\d{2})\s+)?'   # Deposit amount (optional, non-capturing group)
        r'([\d,]+\.\d{2})'          # Closing balance (required)
    )
    
    for i in range(transaction_start_idx, transaction_end_idx):
        line = lines[i].strip()
        print(f"Line: {line}")
        if not line:
            continue
        
        match = transaction_pattern.search(line)
        if match:
            date = match.group(1)
            narration = match.group(2).strip()
            ref_no = match.group(3) or ''
            value_date = match.group(4) or ''
            
            # Clean and convert amounts - leave as empty string if not present
            def clean_amount(amt):
                if not amt:
                    
                    return ''  # Empty string instead of 0
                return float(amt.replace(',', ''))
            
            withdrawal = clean_amount(match.group(5))
            deposit = clean_amount(match.group(6))
            closing_balance = float(match.group(7).replace(',', ''))
        
            # print(f"Withdrawal: {withdrawal}, Deposit: {deposit}")  # Always required
            
            # Ensure only one of withdrawal/deposit is present
            if withdrawal and deposit:
                # If both are present (shouldn't happen), keep the non-zero one
                if withdrawal == 0:
                    withdrawal = ''
                elif deposit == 0:
                    deposit = ''
                else:
                    # If both non-zero, assume it's a withdrawal (more common in statements)
                    deposit = ''
            
            transactions.append({
                'date': date,
                'narration': narration,
                'reference_number': ref_no,
                'value_date': value_date,
                'withdrawal': withdrawal,
                'deposit': deposit,
                'closing_balance': closing_balance
            })
    
    # Convert dates to standard format
    for transaction in transactions:
        try:
            date_obj = datetime.strptime(transaction['date'], "%d/%m/%y")
            transaction['date'] = date_obj.strftime("%Y-%m-%d")
            
            if transaction['value_date']:
                value_date_obj = datetime.strptime(transaction['value_date'], "%d/%m/%y")
                transaction['value_date'] = value_date_obj.strftime("%Y-%m-%d")
        except ValueError:
            # Keep original format if conversion fails
            pass
    
    print(f"    -> Found {len(transactions)} transactions.")
    
    # Create a DataFrame from the transactions
    df = pd.DataFrame(transactions)
    
    # Ensure numeric columns are handled properly - keep empty strings as is
    for col in ['withdrawal', 'deposit']:
        df[col] = df[col].apply(lambda x: x if x == '' else pd.to_numeric(x, errors='coerce'))
    
    # Closing balance should always be numeric
    df['closing_balance'] = pd.to_numeric(df['closing_balance'], errors='coerce')
    
    return df


def parse_hdfc_statement(pdf_path):
    """
    Parse HDFC Bank statement and extract data.
    
    Args:
        pdf_path (str or Path): Path to the PDF file
        
    Returns:
        tuple: (DataFrame with transactions, dict with personal details)
    """
    pdf_path = Path(pdf_path)
    print(f"  -> Opening PDF ({len(list(pdfplumber.open(pdf_path).pages))} pages)...")
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")
    
    all_text = ""
    
    with pdfplumber.open(pdf_path) as pdf:
        # Extract text from all pages
        for i, page in enumerate(pdf.pages):
            print(f"    -> Extracting text from page {i+1}/{len(pdf.pages)}...")
            text = page.extract_text()
            if text:
                all_text += text + "\n"
        
        print(f"  -> Text extraction complete. Total length: {len(all_text)} characters.")
        
        # Extract personal details
        print(f"  -> Extracting personal details...")
        personal_details = extract_hdfc_personal_details(all_text)
        
        # Extract transactions
        print(f"  -> Extracting transactions...")
        transactions = extract_hdfc_transactions(all_text)
        
        # Ensure we have transaction data
        if transactions.empty:
            print("  -> No transactions found. Creating empty dataframe.")
            # Create an empty DataFrame with the expected columns
            transactions = pd.DataFrame(columns=[
                'date', 'narration', 'reference_number', 'value_date', 
                'withdrawal', 'deposit', 'closing_balance'
            ])
        
        return transactions, personal_details


def save_to_csv(transactions, output_path="data/output/transactions.csv"):
    """
    Save transactions DataFrame to CSV.
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure we have all required columns
    required_columns = [
        'date', 'narration', 'reference_number', 'value_date', 
        'withdrawal', 'deposit', 'closing_balance'
    ]
    
    for col in required_columns:
        if col not in transactions.columns:
            transactions[col] = ''  # Add missing column with empty strings
    
    # Remove any unnecessary columns
    transactions = transactions[required_columns]
    
    # Save to CSV with proper handling of empty values
    transactions.to_csv(output_path, index=False, na_rep='')
    print(f"Transactions saved to: {output_path}")
    return output_path


def save_to_json(data, output_path="data/output/statement_data.json"):
    """
    Save data to JSON file.
    
    Args:
        data (dict): Data to save
        output_path (str): Path for output JSON
        
    Returns:
        str: Path to saved JSON file
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to: {output_path}")
    return output_path


def generate_filename(personal_details, statement_date=None):
    """
    Generate a descriptive filename based on account details and date.
    
    Args:
        personal_details (dict): Personal details dictionary
        statement_date (str, optional): Statement date to use in filename
        
    Returns:
        str: Descriptive filename
    """
    account_number = personal_details.get('account_number', 'unknown')
    name = personal_details.get('name', '').replace(' ', '_').lower()
    
    # Use provided date or extract from statement period
    if not statement_date:
        statement_date = personal_details.get('statement_period', {}).get('from', '')
    
    # Format date for filename
    if statement_date:
        try:
            # Try to parse the date string
            if len(statement_date) == 10 and statement_date.count('/') == 2:  # DD/MM/YYYY
                date_obj = datetime.strptime(statement_date, "%d/%m/%Y")
                date_str = date_obj.strftime("%Y%m")  # YYYYMM format
            else:
                # Use as is if we can't parse
                date_str = statement_date.replace('/', '')
        except ValueError:
            date_str = statement_date.replace('/', '')
    else:
        date_str = datetime.now().strftime("%Y%m")
    
    return f"hdfc_{name}_{account_number}_{date_str}"


def parse_statements(input_dir="data/input", output_dir="data/output", filename=None):
    """
    Parse HDFC bank statements in the input directory.
    
    Args:
        input_dir (str): Directory containing PDF statements
        output_dir (str): Directory to save parsed data
        filename (str, optional): Specific file to parse
        
    Returns:
        dict: Summary of processed files and their outputs
    """
    results = {}
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of PDF files
    if filename:
        pdf_files = [filename] if filename.endswith('.pdf') else []
    else:
        pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return results
    
    for filename in pdf_files:
        try:
            print(f"-> Starting to parse: {filename}")
            # Parse the file
            pdf_path = os.path.join(input_dir, filename)
            transactions, personal_details = parse_hdfc_statement(pdf_path)
            
            # Generate a descriptive base name for this statement's outputs
            base_name = generate_filename(personal_details)
            
            # Save transactions to CSV
            csv_path = save_to_csv(
                transactions, 
                output_path=os.path.join(output_dir, f"{base_name}_transactions.csv")
            )
            
            # Save personal details to JSON (without summary section)
            json_path = save_to_json(
                {"personal_details": personal_details}, 
                output_path=os.path.join(output_dir, f"{base_name}_details.json")
            )
            
            results[filename] = {
                "status": "success",
                "transactions_count": len(transactions),
                "csv_path": csv_path,
                "json_path": json_path
            }
            
            print(f"-> Finished parsing: {filename}")
            
        except Exception as e:
            import traceback
            print(f"Error processing {filename}: {str(e)}")
            print(traceback.format_exc())
            results[filename] = {
                "status": "error",
                "error_message": str(e)
            }
    
    return results


if __name__ == "__main__":
    """
    Main function to run the parser directly.
    """
    import argparse
    
    # Set up command-line argument parsing
    arg_parser = argparse.ArgumentParser(description='Parse HDFC bank statement PDFs')
    arg_parser.add_argument('--input_dir', default='data/input', help='Directory with input PDFs')
    arg_parser.add_argument('--output_dir', default='data/output', help='Directory for output files')
    arg_parser.add_argument('--file', help='Specific PDF file to parse (optional)')
    
    args = arg_parser.parse_args()
    
    # Run the parser
    if args.file:
        parse_statements(args.input_dir, args.output_dir, args.file)
    else:
        parse_statements(args.input_dir, args.output_dir)