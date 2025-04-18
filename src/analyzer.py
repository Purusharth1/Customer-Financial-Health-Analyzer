"""Spending Patterns, Fees, Recurring Payments, and Anomalies Detection.

This module analyzes financial transactions to identify patterns, fees, recurring
payments, and anomalies. Key functionalities include:
- Detecting spending patterns using rule-based logic.
- Identifying recurring payments and subscriptions.
- Flagging unusual transactions or fees.
- Providing actionable insights into financial behavior.
- Computing account overview for frontend display.
"""
import logging
import sys
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import sanitize_metric_name, setup_mlflow

# Create a custom logger
logger = logging.getLogger(__name__)

def analyze_transactions(input_csv: str, output_dir: str) -> dict[str, Any]:
    """Analyze transactions for patterns, fees, recurring payments, anomalies, and account overview.

    Args:
        input_csv: Path to categorized transactions CSV.
        output_dir: Directory to save analysis outputs.

    Returns:
        Dictionary with patterns, fees, recurring, anomalies, cash flow, and account overview.

    """
    setup_mlflow()
    logger.info("Starting transaction analysis")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {
        "patterns": [],
        "fees": [],
        "recurring": [],
        "anomalies": [],
        "cash_flow": [],
        "account_overview": {
            "total_balance": 0.0,
            "monthly_income": 0.0,
            "monthly_expense": 0.0,
            "balance_percentage": 0.0,
            "income_percentage": 0.0,
            "expense_percentage": 0.0,
        },
    }

    with mlflow.start_run(run_name="Transaction_Analysis"):
        mlflow.log_param("input_csv", input_csv)
        start_time = pd.Timestamp.now()
        try:
            df = pd.read_csv(input_csv)
            logger.info(f"Read CSV: {(pd.Timestamp.now() - start_time).total_seconds():.3f}s")
        except FileNotFoundError:
            logger.exception("Input CSV not found: %s", input_csv)
            return results
        if df.empty:
            logger.warning(f"Empty CSV: {input_csv}")
            return results
        mlflow.log_metric("transactions_analyzed", len(df))

        # Validate columns
        required = ["parsed_date", "Narration", "Withdrawal (INR)", "Deposit (INR)", "category"]
        if not all(col in df for col in required):
            missing = [col for col in required if col not in df]
            logger.error("Missing columns: %s", missing)
            return results

        # Preprocess
        t = pd.Timestamp.now()
        df["parsed_date"] = pd.to_datetime(df["parsed_date"], errors="coerce")
        df["month"] = df["parsed_date"].dt.to_period("M")
        df["weekday"] = df["parsed_date"].dt.weekday
        df["is_weekend"] = df["weekday"].isin([5, 6])
        df["day"] = df["parsed_date"].dt.day
        df["time_of_month"] = df["day"].apply(
            lambda x: "start" if x <= 10 else "middle" if x <= 20 else "end",
        )
        logger.info(f"Preprocess: {(pd.Timestamp.now() - t).total_seconds():.3f}s")

        # Account Overview
        t = pd.Timestamp.now()
        total_income = df["Deposit (INR)"].sum()
        total_expense = df["Withdrawal (INR)"].sum()
        total_balance = total_income - total_expense
        latest_month = df["month"].max()
        prev_month = latest_month - 1
        latest_income = df[df["month"] == latest_month]["Deposit (INR)"].sum()
        latest_expense = df[df["month"] == latest_month]["Withdrawal (INR)"].sum()
        prev_income = df[df["month"] == prev_month]["Deposit (INR)"].sum()
        prev_expense = df[df["month"] == prev_month]["Withdrawal (INR)"].sum()
        prev_balance = prev_income - prev_expense
        results["account_overview"] = {
            "total_balance": float(total_balance),
            "monthly_income": float(latest_income),
            "monthly_expense": float(latest_expense),
            "balance_percentage": float(((total_balance - prev_balance) / prev_balance * 100) if prev_balance else 0),
            "income_percentage": float(((latest_income - prev_income) / prev_income * 100) if prev_income else 0),
            "expense_percentage": float(((latest_expense - prev_expense) / prev_expense * 100) if prev_expense else 0),
        }
        mlflow.log_metrics({
            "total_balance": total_balance,
            "monthly_income": latest_income,
            "monthly_expense": latest_expense,
        })
        logger.info(f"Account Overview: {(pd.Timestamp.now() - t).total_seconds():.3f}s")

        # Patterns
        t = pd.Timestamp.now()
        patterns = detect_patterns(df)
        results["patterns"] = patterns
        patterns_file = Path(output_dir) / "patterns.txt"
        with patterns_file.open("w") as file:
            file.write("\n".join(patterns))
        mlflow.log_artifact(str(patterns_file))
        logger.info(f"Patterns: {(pd.Timestamp.now() - t).total_seconds():.3f}s")

        # Fees
        t = pd.Timestamp.now()
        fees = detect_fees(df)
        results["fees"] = fees
        fees_file = Path(output_dir) / "fees.csv"
        pd.DataFrame(fees).to_csv(fees_file, index=False)
        mlflow.log_artifact(str(fees_file))
        if not fees:
            logger.warning("No fee transactions detected")
        logger.info(f"Fees: {(pd.Timestamp.now() - t).total_seconds():.3f}s")

        # Recurring
        t = pd.Timestamp.now()
        recurring = detect_recurring(df)
        results["recurring"] = recurring
        recurring_file = Path(output_dir) / "recurring.csv"
        pd.DataFrame(recurring).to_csv(recurring_file, index=False)
        mlflow.log_artifact(str(recurring_file))
        logger.info(f"Recurring: {(pd.Timestamp.now() - t).total_seconds():.3f}s")

        # Anomalies
        t = pd.Timestamp.now()
        anomalies = detect_anomalies(df)
        results["anomalies"] = anomalies
        anomalies_file = Path(output_dir) / "anomalies.csv"
        pd.DataFrame(anomalies).to_csv(anomalies_file, index=False)
        mlflow.log_artifact(str(anomalies_file))
        logger.info(f"Anomalies: {(pd.Timestamp.now() - t).total_seconds():.3f}s")

        # Cash Flow
        t = pd.Timestamp.now()
        cash_flow = analyze_cash_flow(df)
        results["cash_flow"] = cash_flow
        cash_flow_file = Path(output_dir) / "cash_flow.csv"
        pd.DataFrame(cash_flow).to_csv(cash_flow_file, index=False)
        mlflow.log_artifact(str(cash_flow_file))
        logger.info(f"Cash Flow: {(pd.Timestamp.now() - t).total_seconds():.3f}s")

        # Log counts
        for key, items in results.items():
            if key != "account_overview":
                mlflow.log_metric(sanitize_metric_name(f"{key}_count"), len(items))
            else:
                for subkey, value in items.items():
                    mlflow.log_metric(sanitize_metric_name(f"account_{subkey}"), value)

        logger.info(f"Total analysis: {(pd.Timestamp.now() - start_time).total_seconds():.3f}s")
        return results

# Constants
HIGH_SPENDING_MULTIPLIER = 1.5
LOW_SPENDING_MULTIPLIER = 0.7
WEEKEND_MULTIPLIER = 1.3
MIN_MONTHS_FOR_TREND = 2

def detect_patterns(df: pd.DataFrame) -> list[str]:
    """Identify spending patterns using rule-based logic."""
    logger.info("Detecting spending patterns")
    patterns = []

    if df.empty or df["parsed_date"].isna().all():
        logger.warning("No valid data for patterns")
        return ["No patterns detected"]

    # Category dominance
    cat_totals = df.groupby("category")["Withdrawal (INR)"].sum()
    if not cat_totals.empty:
        top_cat = cat_totals.idxmax()
        if cat_totals[top_cat] > cat_totals.mean() * HIGH_SPENDING_MULTIPLIER:
            patterns.append(f"High {top_cat} spending (₹{cat_totals[top_cat]:.2f})")

    # Monthly trends
    monthly_totals = df.groupby("month")["Withdrawal (INR)"].sum()
    if len(monthly_totals) > MIN_MONTHS_FOR_TREND:
        max_month = monthly_totals.idxmax()
        min_month = monthly_totals.idxmin()
        if monthly_totals[max_month] > monthly_totals.mean() * HIGH_SPENDING_MULTIPLIER:
            patterns.append(f"Higher spending in {max_month} (₹{monthly_totals[max_month]:.2f})")
        if monthly_totals[min_month] < monthly_totals.mean() * LOW_SPENDING_MULTIPLIER:
            patterns.append(f"Lower spending in {min_month} (₹{monthly_totals[min_month]:.2f})")

    # Weekend vs weekday
    weekend_spending = df[df["is_weekend"]]["Withdrawal (INR)"].sum() / df["is_weekend"].sum()
    weekday_spending = df[~df["is_weekend"]]["Withdrawal (INR)"].sum() / (~df["is_weekend"]).sum()
    if weekend_spending > weekday_spending * WEEKEND_MULTIPLIER:
        patterns.append(f"Higher weekend spending (₹{weekend_spending:.2f}/day) vs weekdays (₹{weekday_spending:.2f}/day)")
    elif weekday_spending > weekend_spending * WEEKEND_MULTIPLIER:
        patterns.append(f"Higher weekday spending (₹{weekday_spending:.2f}/day) vs weekends (₹{weekend_spending:.2f}/day)")

    return patterns if patterns else ["No patterns detected"]

# Constants for fees
WITHDRAWAL_THRESHOLD_MULTIPLIER = 0.2
MIN_REPEAT_COUNT = 2

def detect_fees(df: pd.DataFrame) -> list[dict]:
    """Identify fee or interest-related transactions."""
    keywords = [
        "FEE", "CHARGE", "INTEREST", "PENALTY", "TAX", "COMMISSION",
        "SERVICE CHARGE", "LATE FEE", "SURCHARGE", "GST",
        "MAINTENANCE", "AMC", "ANNUAL",
    ]
    mask = df["Narration"].str.upper().str.contains("|".join(keywords), na=False)
    fees = df[mask][["parsed_date", "Narration", "Withdrawal (INR)", "Deposit (INR)", "category"]].copy()

    # Small recurring fees
    mean_withdrawal = df["Withdrawal (INR)"].mean()
    potential_fee_mask = (
        (df["Withdrawal (INR)"] > 0) &
        (df["Withdrawal (INR)"] < mean_withdrawal * WITHDRAWAL_THRESHOLD_MULTIPLIER) &
        (~mask)
    )
    potential_fees = df[potential_fee_mask]
    if not potential_fees.empty:
        amount_counts = potential_fees.groupby("Withdrawal (INR)").size()
        recurring_amounts = amount_counts[amount_counts >= MIN_REPEAT_COUNT].index
        recurring_fees = potential_fees[potential_fees["Withdrawal (INR)"].isin(recurring_amounts)]
        fees = pd.concat([fees, recurring_fees[["parsed_date", "Narration", "Withdrawal (INR)", "Deposit (INR)", "category"]]])

    fees["amount"] = fees["Withdrawal (INR)"].where(fees["Withdrawal (INR)"] > 0, fees["Deposit (INR)"])
    fees["type"] = fees["Withdrawal (INR)"].where(fees["Withdrawal (INR)"] > 0, 0).apply(lambda x: "withdrawal" if x > 0 else "deposit")
    fees["fee_type"] = fees["Narration"].str.upper().apply(
        lambda x: "interest" if "INTEREST" in x
        else "tax" if any(w in x for w in ["TAX", "GST"])
        else "annual" if any(w in x for w in ["ANNUAL", "AMC", "YEARLY"])
        else "maintenance" if any(w in x for w in ["MAINTENANCE", "SERVICE"])
        else "penalty" if any(w in x for w in ["LATE", "PENALTY"])
        else "other",
    )

    return fees[["parsed_date", "Narration", "amount", "type", "fee_type", "category"]].to_dict("records")

# Constants for recurring
MONTHLY_RANGE = (25, 35)
WEEKLY_RANGE = (6, 8)
BIWEEKLY_RANGE = (13, 16)
QUARTERLY_RANGE = (85, 95)
ANNUAL_RANGE = (355, 370)
DAILY_WORKDAYS_THRESHOLD = 5
HIGH_REGULARITY_THRESHOLD = 3
MIN_OCCURRENCES = 3
AMOUNT_VARIATION_THRESHOLD = 0.15

def detect_recurring(df: pd.DataFrame) -> list[dict]:
    """Detect recurring payments or deposits."""
    recurring = []

    # Withdrawals
    w_df = df[df["Withdrawal (INR)"] > 0][["parsed_date", "Narration", "Withdrawal (INR)", "category"]]
    if not w_df.empty:
        detect_exact_amount_recurring(w_df, "withdrawal", "Withdrawal (INR)", recurring)
        detect_similar_amount_recurring(w_df, "withdrawal", "Withdrawal (INR)", recurring)

    # Deposits
    d_df = df[df["Deposit (INR)"] > 0][["parsed_date", "Narration", "Deposit (INR)", "category"]]
    if not d_df.empty:
        detect_exact_amount_recurring(d_df, "deposit", "Deposit (INR)", recurring)
        detect_similar_amount_recurring(d_df, "deposit", "Deposit (INR)", recurring)

    return recurring

def determine_frequency(days_delta: list[int]) -> str:
    """Determine the frequency of recurring transactions."""
    mean_delta = round(np.mean(days_delta))
    if MONTHLY_RANGE[0] <= mean_delta <= MONTHLY_RANGE[1]:
        return "monthly"
    if WEEKLY_RANGE[0] <= mean_delta <= WEEKLY_RANGE[1]:
        return "weekly"
    if BIWEEKLY_RANGE[0] <= mean_delta <= BIWEEKLY_RANGE[1]:
        return "biweekly"
    if QUARTERLY_RANGE[0] <= mean_delta <= QUARTERLY_RANGE[1]:
        return "quarterly"
    if ANNUAL_RANGE[0] <= mean_delta <= ANNUAL_RANGE[1]:
        return "annual"
    if mean_delta <= DAILY_WORKDAYS_THRESHOLD:
        return "daily/workdays"
    return f"approximately every {mean_delta} days"

def detect_exact_amount_recurring(df: pd.DataFrame, transaction_type: str, amount_col: str, recurring: list[dict]) -> None:
    """Detect recurring transactions with exact amounts."""
    grouped = df.groupby(["Narration", amount_col]).agg(
        dates=("parsed_date", list),
        count=("parsed_date", "count"),
        category=("category", "first"),
    ).reset_index()
    grouped = grouped[grouped["count"] > 1]
    for _, row in grouped.iterrows():
        dates = sorted(pd.to_datetime(row["dates"]))
        deltas = np.diff(dates).astype("timedelta64[D]").astype(int)
        if len(deltas) > 0 and np.std(deltas) < HIGH_REGULARITY_THRESHOLD:
            recurring.append({
                "narration": row["Narration"],
                "amount": row[amount_col],
                "frequency": determine_frequency(deltas),
                "category": row["category"],
                "type": transaction_type,
                "match_type": "exact_amount",
                "regularity": "high",
                "first_date": dates[0].strftime("%Y-%m-%d"),
                "last_date": dates[-1].strftime("%Y-%m-%d"),
                "occurrence_count": row["count"],
            })

def detect_similar_amount_recurring(df: pd.DataFrame, transaction_type: str, amount_col: str, recurring: list[dict]) -> None:
    """Detect recurring transactions with similar amounts."""
    grouped = df.groupby("Narration").agg(
        amounts=(amount_col, list),
        dates=("parsed_date", list),
        count=("parsed_date", "count"),
        category=("category", "first"),
    ).reset_index()
    grouped = grouped[grouped["count"] >= MIN_OCCURRENCES]
    for _, row in grouped.iterrows():
        amounts = np.array(row["amounts"])
        mean_amount = amounts.mean()
        std_amount = amounts.std()
        if 0 < std_amount < mean_amount * AMOUNT_VARIATION_THRESHOLD:
            dates = sorted(pd.to_datetime(row["dates"]))
            deltas = np.diff(dates).astype("timedelta64[D]").astype(int)
            if len(deltas) > 0 and np.std(deltas) < HIGH_REGULARITY_THRESHOLD:
                recurring.append({
                    "narration": row["Narration"],
                    "amount": f"{mean_amount:.2f} (±{std_amount:.2f})",
                    "frequency": determine_frequency(deltas),
                    "category": row["category"],
                    "type": transaction_type,
                    "match_type": "similar_amounts",
                    "regularity": "medium",
                    "first_date": dates[0].strftime("%Y-%m-%d"),
                    "last_date": dates[-1].strftime("%Y-%m-%d"),
                    "occurrence_count": row["count"],
                })

# Constants for anomalies
MIN_DATA_POINTS = 5
Z_SCORE_THRESHOLD = 3
SPIKE_THRESHOLD = 5
TIME_GAP_MULTIPLIER = 3

def detect_anomalies(df: pd.DataFrame) -> list[dict]:
    """Flag unusual transactions by category."""
    anomalies = []

    # Withdrawals
    w_df = df[df["Withdrawal (INR)"] > 0].copy()
    if not w_df.empty:
        stats = w_df.groupby("category")["Withdrawal (INR)"].agg(["mean", "std"]).fillna(0)
        w_df = w_df.merge(stats, on="category", how="left")
        w_df["z_score"] = (w_df["Withdrawal (INR)"] - w_df["mean"]) / w_df["std"].replace(0, w_df["Withdrawal (INR)"].mean())
        w_anomalies = w_df[w_df["z_score"] > Z_SCORE_THRESHOLD][["parsed_date", "Narration", "Withdrawal (INR)", "category", "z_score"]]
        if not w_anomalies.empty:
            w_anomalies["amount"] = w_anomalies["Withdrawal (INR)"]
            w_anomalies["type"] = "withdrawal"
            w_anomalies["severity"] = w_anomalies["z_score"].apply(lambda z: "high" if z > 5 else "moderate")
            w_anomalies["detection_method"] = "statistical"
            anomalies.extend(w_anomalies[["parsed_date", "Narration", "amount", "type", "severity", "category", "detection_method"]].to_dict("records"))

        # Sudden spikes
        w_df = w_df.sort_values("parsed_date")
        w_df["prev_amount"] = w_df.groupby("category")["Withdrawal (INR)"].shift(1)
        w_df["ratio"] = w_df["Withdrawal (INR)"] / w_df["prev_amount"]
        spikes = w_df[(w_df["ratio"] > SPIKE_THRESHOLD) & (w_df["prev_amount"].notna())]
        if not spikes.empty:
            spikes = spikes[["parsed_date", "Narration", "Withdrawal (INR)", "category"]]
            spikes["amount"] = spikes["Withdrawal (INR)"]
            spikes["type"] = "withdrawal"
            spikes["severity"] = "high"
            spikes["detection_method"] = "sudden_increase"
            anomalies.extend(spikes[["parsed_date", "Narration", "amount", "type", "severity", "category", "detection_method"]].to_dict("records"))

    # Deposits
    d_df = df[df["Deposit (INR)"] > 0].copy()
    if not d_df.empty:
        stats = d_df.groupby("category")["Deposit (INR)"].agg(["mean", "std"]).fillna(0)
        d_df = d_df.merge(stats, on="category", how="left")
        d_df["z_score"] = (d_df["Deposit (INR)"] - d_df["mean"]) / d_df["std"].replace(0, d_df["Deposit (INR)"].mean())
        d_anomalies = d_df[d_df["z_score"] > Z_SCORE_THRESHOLD][["parsed_date", "Narration", "Deposit (INR)", "category", "z_score"]]
        if not d_anomalies.empty:
            d_anomalies["amount"] = d_anomalies["Deposit (INR)"]
            d_anomalies["type"] = "deposit"
            d_anomalies["severity"] = d_anomalies["z_score"].apply(lambda z: "high" if z > 5 else "moderate")
            d_anomalies["detection_method"] = "statistical"
            anomalies.extend(d_anomalies[["parsed_date", "Narration", "amount", "type", "severity", "category", "detection_method"]].to_dict("records"))

        # Sudden spikes
        d_df = d_df.sort_values("parsed_date")
        d_df["prev_amount"] = d_df.groupby("category")["Deposit (INR)"].shift(1)
        d_df["ratio"] = d_df["Deposit (INR)"] / d_df["prev_amount"]
        spikes = d_df[(d_df["ratio"] > SPIKE_THRESHOLD) & (d_df["prev_amount"].notna())]
        if not spikes.empty:
            spikes = spikes[["parsed_date", "Narration", "Deposit (INR)", "category"]]
            spikes["amount"] = spikes["Deposit (INR)"]
            spikes["type"] = "deposit"
            spikes["severity"] = "high"
            spikes["detection_method"] = "sudden_increase"
            anomalies.extend(spikes[["parsed_date", "Narration", "amount", "type", "severity", "category", "detection_method"]].to_dict("records"))

    # Frequency anomalies
    df_sorted = df.sort_values("parsed_date")
    df_sorted["next_trans_days"] = (df_sorted["parsed_date"].shift(-1) - df_sorted["parsed_date"]).dt.days
    mean_gap = df_sorted["next_trans_days"].mean()
    std_gap = df_sorted["next_trans_days"].std()
    large_gaps = df_sorted[df_sorted["next_trans_days"] > mean_gap + TIME_GAP_MULTIPLIER * std_gap]
    if not large_gaps.empty:
        for _, row in large_gaps.iterrows():
            if pd.notna(row["next_trans_days"]):
                anomalies.append({
                    "parsed_date": row["parsed_date"],
                    "Narration": f"Unusual gap after this transaction ({row['next_trans_days']} days)",
                    "amount": 0,
                    "type": "gap",
                    "category": "timing_anomaly",
                    "severity": "moderate",
                    "detection_method": "timing_gap",
                })

    return anomalies

def analyze_cash_flow(df: pd.DataFrame) -> list[dict]:
    """Analyze cash flow."""
    cash_flow_analysis = []
    if df.empty or "parsed_date" not in df:
        return cash_flow_analysis

    monthly_cf = df.groupby("month").agg({
        "Deposit (INR)": "sum",
        "Withdrawal (INR)": "sum",
    }).reset_index()
    monthly_cf["net_cash_flow"] = monthly_cf["Deposit (INR)"] - monthly_cf["Withdrawal (INR)"]
    monthly_cf["month"] = monthly_cf["month"].astype(str)

    for _, row in monthly_cf.iterrows():
        cash_flow_analysis.append({
            "month": row["month"],
            "income": round(row["Deposit (INR)"], 2),
            "expenses": round(row["Withdrawal (INR)"], 2),
            "net_cash_flow": round(row["net_cash_flow"], 2),
            "status": "Positive" if row["net_cash_flow"] > 0 else "Negative",
        })

    return cash_flow_analysis

if __name__ == "__main__":
    input_csv = "data/output/categorized.csv"
    output_dir = "data/output/analysis"
    results = analyze_transactions(input_csv, output_dir)
    print(results)
