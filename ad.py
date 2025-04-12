import datetime
import os
import random

from fpdf import FPDF


def generate_hdfc_statement(account_no, month, year, output_dir="statements"):
    """Generate realistic HDFC bank statement PDF"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pdf = FPDF()
    pdf.add_page()

    # Set font for entire document
    pdf.set_font("Arial", size=10)

    # HDFC Header
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "HDFC BANK LIMITED", 0, 1, "C")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 5, "Corporate Office: HDFC Bank House, Senapati Bapat Marg, Lower Parel, Mumbai - 400013", 0, 1, "C")
    pdf.ln(5)

    # Account Details
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "ACCOUNT STATEMENT", 0, 1)
    pdf.set_font("Arial", "", 10)

    customer_name = random.choice(["Aditya Sharma", "Priya Patel", "Rahul Gupta", "Neha Singh"])
    branch = random.choice(["Mumbai Main", "Delhi Central", "Bangalore Electronic City", "Hyderabad Jubilee"])

    pdf.cell(50, 6, "Customer Name:", 0)
    pdf.cell(0, 6, customer_name, 0, 1)
    pdf.cell(50, 6, "Account Number:", 0)
    pdf.cell(0, 6, account_no, 0, 1)
    pdf.cell(50, 6, "Branch:", 0)
    pdf.cell(0, 6, branch, 0, 1)
    pdf.cell(50, 6, "Period:", 0)
    pdf.cell(0, 6, f"01/{month:02d}/{year} to {datetime.date(year, month, 28).strftime('%d/%m/%Y')}", 0, 1)
    pdf.ln(5)

    # Transaction Table Header
    pdf.set_fill_color(200, 220, 255)
    pdf.set_font("Arial", "B", 10)
    pdf.cell(20, 8, "Date", 1, 0, "C", 1)
    pdf.cell(50, 8, "Description", 1, 0, "C", 1)
    pdf.cell(30, 8, "Ref No.", 1, 0, "C", 1)
    pdf.cell(25, 8, "Withdrawal", 1, 0, "C", 1)
    pdf.cell(25, 8, "Deposit", 1, 0, "C", 1)
    pdf.cell(30, 8, "Balance", 1, 1, "C", 1)

    # Generate realistic transactions
    opening_balance = random.randint(50000, 100000)
    balance = opening_balance
    transactions = []

    # Add salary credit at beginning of month
    salary_date = datetime.date(year, month, 1)
    salary_amount = random.randint(30000, 100000)
    transactions.append((salary_date, "SALARY CREDIT", f"SLR{random.randint(1000,9999)}", 0, salary_amount, balance + salary_amount))
    balance += salary_amount

    # Add regular transactions
    categories = {
        "Groceries": ["BIG BAZAAR", "MORE SUPERMARKET", "DMART", "RELIANCE FRESH"],
        "Dining": ["SWIGGY", "ZOMATO", "UBER EATS", "DOMINOS PIZZA"],
        "Shopping": ["AMAZON", "FLIPKART", "MYNTRA", "AJIO"],
        "Utilities": ["ELECTRICITY BILL", "WATER BILL", "GAS BILL", "PHONE BILL"],
        "Transfer": ["UPI TRANSFER", "NEFT TRANSFER", "IMPS TRANSFER", "SELF TRANSFER"],
        "ATM": ["ATM WITHDRAWAL", "CASH WITHDRAWAL"],
    }

    for day in range(2, 28):
        if random.random() > 0.7:  # 30% chance of transaction each day
            date = datetime.date(year, month, day)
            category = random.choice(list(categories.keys()))
            desc = random.choice(categories[category])
            ref_no = f"REF{random.randint(10000,99999)}"

            if category in ["Transfer", "ATM"]:
                amount = round(random.uniform(1000, 20000), 2)
                transactions.append((date, desc, ref_no, amount, 0, balance - amount))
                balance -= amount
            else:
                amount = round(random.uniform(100, 5000), 2)
                transactions.append((date, desc, ref_no, amount, 0, balance - amount))
                balance -= amount

    # Add some deposits
    for _ in range(3):
        day = random.randint(5, 25)
        date = datetime.date(year, month, day)
        desc = random.choice(["INTEREST CREDIT", "DIVIDEND CREDIT", "FUND TRANSFER"])
        amount = round(random.uniform(1000, 15000), 2)
        transactions.append((date, desc, f"REF{random.randint(10000,99999)}", 0, amount, balance + amount))
        balance += amount

    # Sort transactions by date
    transactions.sort()

    # Add transactions to PDF
    pdf.set_font("Arial", "", 8)
    for date, desc, ref, withdraw, deposit, bal in transactions:
        pdf.cell(20, 6, date.strftime("%d/%m/%Y"), 1)
        pdf.cell(50, 6, desc, 1)
        pdf.cell(30, 6, ref, 1)
        pdf.cell(25, 6, f"Rs.{withdraw:,.2f}" if withdraw > 0 else "", 1, 0, "R")
        pdf.cell(25, 6, f"Rs.{deposit:,.2f}" if deposit > 0 else "", 1, 0, "R")
        pdf.cell(30, 6, f"Rs.{bal:,.2f}", 1, 1, "R")

    # Footer Summary
    pdf.ln(5)
    pdf.set_font("Arial", "B", 10)
    pdf.cell(0, 8, "Summary:", 0, 1)
    pdf.set_font("Arial", "", 10)

    total_withdrawals = sum(t[3] for t in transactions)
    total_deposits = sum(t[4] for t in transactions)

    pdf.cell(50, 6, "Opening Balance:", 0)
    pdf.cell(0, 6, f"Rs. {opening_balance:,.2f}", 0, 1, "R")
    pdf.cell(50, 6, "Total Deposits:", 0)
    pdf.cell(0, 6, f"Rs. {total_deposits:,.2f}", 0, 1, "R")
    pdf.cell(50, 6, "Total Withdrawals:", 0)
    pdf.cell(0, 6, f"Rs. {total_withdrawals:,.2f}", 0, 1, "R")
    pdf.cell(50, 6, "Closing Balance:", 0)
    pdf.cell(0, 6, f"Rs. {balance:,.2f}", 0, 1, "R")

    # Disclaimer
    pdf.ln(10)
    pdf.set_font("Arial", "I", 8)
    pdf.multi_cell(0, 4, "Note: This is a computer generated statement and does not require signature. Please report any discrepancies within 7 days of receipt.")

    # Save PDF
    filename = os.path.join(output_dir, f"HDFC_{account_no}_{month:02d}_{year}.pdf")
    pdf.output(filename)

    return filename



# Generate 10 statements with different account numbers
for i in range(1, 11):
    account_no = f"5010{random.randint(10000000, 99999999)}"
    month = random.randint(1, 12)
    year = 2023
    generate_hdfc_statement(account_no, month, year)
