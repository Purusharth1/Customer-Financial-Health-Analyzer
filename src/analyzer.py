"""Spending Patterns, Fees, Recurring Payments, and Anomalies Detection.

This module analyzes financial transactions to identify patterns, fees, recurring
payments, and anomalies. Key functionalities include:
- Detecting spending patterns using rule-based logic or LLM-based insights.
- Identifying recurring payments and subscriptions.
- Flagging unusual transactions or fees.
- Providing actionable insights into financial behavior.
"""

import logging
import sys
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from utils import get_llm_config, sanitize_metric_name, setup_mlflow

sys.path.append(str(Path(__file__).parent.parent))

from llm_setup.config import LLMConfig
from llm_setup.ollama_manager import query_llm

# Create a custom logger
logger = logging.getLogger(__name__)
TIMEOUT_SECONDS = 30

def analyze_transactions(input_csv: str, output_dir: str) -> dict[str, list]:
    """Analyze transactions for patterns, fees, recurring payments, and anomalies.

    Args:
        input_csv: Path to categorized transactions CSV.
        output_dir: Directory to save analysis outputs.

    Returns:
        Dictionary with patterns, fees, recurring payments, and anomalies.

    """
    setup_mlflow()
    llm_config = get_llm_config()
    logger.info("Starting transaction analysis")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {
        "patterns": [],
        "fees": [],
        "recurring": [],
        "anomalies": [],
        "cash_flow": [],
    }

    with mlflow.start_run(run_name="Transaction_Analysis"):
        mlflow.log_param("input_csv", input_csv)
        try:
            transactions_df = pd.read_csv(input_csv)
        except FileNotFoundError:
            logger.exception("Input CSV not found: %s", input_csv)
            return results
        mlflow.log_metric("transactions_analyzed", len(transactions_df))

        # Validate columns
        required = [
            "parsed_date",
            "Narration",
            "Withdrawal (INR)",
            "Deposit (INR)",
            "category",
        ]
        if not all(col in transactions_df for col in required):
            missing = [col for col in required if col not in transactions_df]
            logger.error("Missing columns: %s", missing)
            return results

        transactions_df["parsed_date"] = pd.to_datetime(
        transactions_df["parsed_date"], errors="coerce")

        # Patterns
        patterns = detect_patterns(transactions_df, llm_config)
        results["patterns"] = patterns
        patterns_file = Path(output_dir) / "patterns.txt"
        with patterns_file.open("w") as file:
            file.write("\n".join(patterns))
        mlflow.log_artifact(str(patterns_file))

        # Fees
        fees = detect_fees(transactions_df)
        results["fees"] = fees
        fees_file = Path(output_dir) / "fees.csv"
        pd.DataFrame(fees).to_csv(fees_file, index=False)
        mlflow.log_artifact(str(fees_file))
        if not fees:
            logger.warning("No fee transactions detected")

        # Recurring
        recurring = detect_recurring(transactions_df)
        results["recurring"] = recurring
        recurring_file = Path(output_dir) / "recurring.csv"
        pd.DataFrame(recurring).to_csv(recurring_file, index=False)
        mlflow.log_artifact(str(recurring_file))

        # Anomalies
        anomalies = detect_anomalies(transactions_df)
        results["anomalies"] = anomalies
        anomalies_file = Path(output_dir) / "anomalies.csv"
        pd.DataFrame(anomalies).to_csv(anomalies_file, index=False)
        mlflow.log_artifact(str(anomalies_file))

        # Cash Flow
        cash_flow = analyze_cash_flow(transactions_df)
        results["cash_flow"] = cash_flow
        cash_flow_file = Path(output_dir) / "cash_flow.csv"
        pd.DataFrame(cash_flow).to_csv(cash_flow_file, index=False)
        mlflow.log_artifact(str(cash_flow_file))

        # Log counts
        for key, items in results.items():
            mlflow.log_metric(sanitize_metric_name(f"{key}_count"), len(items))

        logger.info("Analysis complete")
        return results




# Constants for readability
START_OF_MONTH_THRESHOLD = 10
MIDDLE_OF_MONTH_THRESHOLD = 20
MIN_MONTHS_FOR_TREND = 2
MIN_GROUPS_FOR_ANALYSIS = 2
HIGH_SPENDING_MULTIPLIER = 1.5
LOW_SPENDING_MULTIPLIER = 0.7
WEEKEND_MULTIPLIER = 1.3

def detect_patterns(transactions_df: pd.DataFrame, llm_config: LLMConfig) -> list[str]:
    """Identify spending patterns using LLM or fallback rules."""
    logger.info("Detecting spending patterns")

    # Validate data
    if transactions_df.empty or transactions_df["parsed_date"].isna().all():
        logger.warning("No valid data for patterns")
        return ["No patterns detected"]

    # Prepare data
    transactions_df = preprocess_data(transactions_df)

    # Generate summary statistics
    summary = generate_summary(transactions_df)

    # Try LLM-based pattern detection
    patterns = try_llm_detection(summary, llm_config)
    if patterns:
        return patterns[:10]

    # Fallback to rule-based pattern detection
    logger.info("Falling back to rule-based patterns")
    patterns = detect_rule_based_patterns(transactions_df)

    return patterns if patterns else ["No patterns detected"]


def preprocess_data(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the transaction data."""
    transactions_df = transactions_df.copy()
    transactions_df["month"] = transactions_df["parsed_date"].dt.to_period("M")
    transactions_df["weekday"] = transactions_df["parsed_date"].dt.day_name()
    transactions_df["day"] = transactions_df["parsed_date"].dt.day
    transactions_df["week"] = transactions_df["parsed_date"].dt.isocalendar().week
    transactions_df["time_of_month"] = transactions_df["day"].apply(
        lambda x: "start" if x <= START_OF_MONTH_THRESHOLD
        else "middle" if x <= MIDDLE_OF_MONTH_THRESHOLD else "end",
    )
    return transactions_df


def generate_summary(transactions_df: pd.DataFrame) -> dict[str, Any]:
    """Generate summary statistics for spending patterns."""
    top_categories = (
        transactions_df.groupby("category")["Withdrawal (INR)"]
        .sum()
        .nlargest(5)
        .index
        .tolist()
    )
    recent_months = transactions_df["month"].unique()[-3:]

    return {
        "categories": (
            transactions_df[transactions_df["category"].isin(top_categories)]
            .groupby("category")[["Withdrawal (INR)", "Deposit (INR)"]]
            .sum()
            .round(2)
            .to_dict()
        ),
        "monthly": (
            transactions_df[transactions_df["month"].isin(recent_months)]
            .groupby(["month", "category"])[["Withdrawal (INR)", "Deposit (INR)"]]
            .sum()
            .round(2)
            .to_dict()
        ),
        "weekday": (
            transactions_df[transactions_df["category"].isin(top_categories)]
            .groupby(["weekday", "category"])["Withdrawal (INR)"]
            .sum()
            .round(2)
            .to_dict()
        ),
        "time_of_month": (
            transactions_df[transactions_df["category"].isin(top_categories)]
            .groupby(["time_of_month", "category"])["Withdrawal (INR)"]
            .sum()
            .round(2)
            .to_dict()
        ),
        "counts": transactions_df.groupby("category").size().to_dict(),
        "weekend_vs_weekday": {
            "weekend": (
                transactions_df[
                    transactions_df["weekday"].isin(["Saturday", "Sunday"])
                ]["Withdrawal (INR)"]
                .sum()
                .round(2)
            ),
            "weekday": (
                transactions_df[
                    ~transactions_df["weekday"].isin(["Saturday", "Sunday"])
                ]["Withdrawal (INR)"]
                .sum()
                .round(2)
            ),
        },
    }


def try_llm_detection(summary: dict[str, Any], llm_config: LLMConfig) -> list[str]:
    """Attempt to detect patterns using an LLM."""
    prompt = (
        f"Analyze spending for categories {summary['categories']}. "
        f"Data: {summary}. "
        "Identify specific patterns like: "
        "1. High spending in specific categories "
        "2. Monthly or weekly spending trends "
        "3. Beginning/middle/end of month spending habits "
        "4. Weekend vs weekday spending differences "
        "5. Periodic spikes in specific categories "
        "6. Gradually increasing/decreasing trends "
        "Return one pattern per line, be specific about amounts and timing."
    )
    logger.debug("LLM prompt: %s", prompt)

    for attempt in range(2):  # Retry once
        try:
            response = query_llm(prompt, llm_config).strip()
            logger.debug("LLM response: %s", response)
            if response:
                return [p.strip() for p in response.split("\n") if p.strip()]
        except (TimeoutError, ValueError) as e:
            logger.warning("LLM attempt %d failed: %s", attempt + 1, e)
    return []

def detect_rule_based_patterns(transactions_df: pd.DataFrame) -> list[str]:
    """Detect patterns using rule-based logic."""
    patterns = []

    # Category dominance
    cat_totals = transactions_df.groupby("category")["Withdrawal (INR)"].sum()
    if not cat_totals.empty:
        top_cat = cat_totals.idxmax()
        if cat_totals[top_cat] > cat_totals.mean() * HIGH_SPENDING_MULTIPLIER:
            patterns.append(f"High {top_cat} spending (₹{cat_totals[top_cat]:.2f})")

    # Monthly trends
    monthly_totals = transactions_df.groupby("month")["Withdrawal (INR)"].sum()
    if len(monthly_totals) > MIN_MONTHS_FOR_TREND:
        max_month = monthly_totals.idxmax()
        min_month = monthly_totals.idxmin()
        if monthly_totals[max_month] > monthly_totals.mean() * HIGH_SPENDING_MULTIPLIER:
            patterns.append(
                f"Higher spending in {max_month} (₹{monthly_totals[max_month]:.2f})",
            )
        if monthly_totals[min_month] < monthly_totals.mean() * LOW_SPENDING_MULTIPLIER:
            patterns.append(
                f"Lower spending in {min_month} (₹{monthly_totals[min_month]:.2f})",
            )

    # Weekday vs weekend spikes
    weekday_totals = transactions_df.groupby("weekday")["Withdrawal (INR)"].sum()
    weekend_avg = (weekday_totals.get("Saturday", 0) +
                   weekday_totals.get("Sunday", 0)) / 2
    weekday_avg = (
        sum([weekday_totals.get(day, 0) for day in ["Monday",
        "Tuesday", "Wednesday", "Thursday", "Friday"]]) / 5
    )

    if weekend_avg > weekday_avg * WEEKEND_MULTIPLIER:
        patterns.append(
            f"""Higher weekend spending (₹{weekend_avg:.2f}/day)
            vs weekdays (₹{weekday_avg:.2f}/day)""",
        )
    elif weekday_avg > weekend_avg * WEEKEND_MULTIPLIER:
        patterns.append(
            f"""Higher weekday spending (₹{weekday_avg:.2f}/day)
            vs weekends (₹{weekend_avg:.2f}/day)""",
        )

    return patterns





# Constants for readability
MIN_REPEAT_COUNT = 2
WITHDRAWAL_THRESHOLD_MULTIPLIER = 0.2

def detect_fees(transactions_df: pd.DataFrame) -> list[dict]:
    """Identify fee or interest-related transactions with enhanced detection."""
    # Enhanced keywords list for better detection
    keywords = [
        "FEE", "CHARGE", "INTEREST", "PENALTY", "TAX", "COMMISSION",
        "SERVICE CHARGE", "LATE FEE", "SURCHARGE", "GST",
        "MAINTENANCE", "AMC", "ANNUAL",
    ]

    # Include both withdrawals and deposits (e.g., credited interest)
    mask = (
        transactions_df["Narration"]
        .str.upper()
        .str.contains("|".join(keywords), na=False)
    )
    fees = transactions_df[mask][[
        "parsed_date", "Narration", "Withdrawal (INR)", "Deposit (INR)", "category",
    ]].copy()

    # Look for small, regular withdrawals that might be subscription fees
    potential_fee_mask = (
        (transactions_df["Withdrawal (INR)"] > 0) &
        (transactions_df["Withdrawal (INR)"] <
         transactions_df["Withdrawal (INR)"].mean()
         * WITHDRAWAL_THRESHOLD_MULTIPLIER) &
        (~mask)  # Exclude already identified fees
    )
    potential_fees = transactions_df[potential_fee_mask]

    # Check if these small amounts repeat
    if not potential_fees.empty:
        for amount in potential_fees["Withdrawal (INR)"].unique():
            amount_df = potential_fees[potential_fees["Withdrawal (INR)"] == amount]
            if len(amount_df) >= MIN_REPEAT_COUNT:
                fees = pd.concat([
                    fees,
                    amount_df[[
                        "parsed_date", "Narration",
                        "Withdrawal (INR)", "Deposit (INR)", "category",
                    ]],
                ])

    # Add more metadata to each fee
    fees["amount"] = fees["Withdrawal (INR)"].where(
        fees["Withdrawal (INR)"] > 0, fees["Deposit (INR)"],
    )
    fees["type"] = (
        fees["Withdrawal (INR)"]
        .where(fees["Withdrawal (INR)"] > 0, 0)
        .apply(lambda x: "withdrawal" if x > 0 else "deposit")
    )

    # Add fee categorization
    fees["fee_type"] = "other"
    for row_idx, row in fees.iterrows():
        narration = row["Narration"].upper()
        if "INTEREST" in narration:
            fees.loc[row_idx, "fee_type"] = "interest"
        elif any(word in narration for word in ["TAX", "GST"]):
            fees.loc[row_idx, "fee_type"] = "tax"
        elif any(word in narration for word in ["ANNUAL", "AMC", "YEARLY"]):
            fees.loc[row_idx, "fee_type"] = "annual"
        elif any(word in narration for word in ["MAINTENANCE", "SERVICE"]):
            fees.loc[row_idx, "fee_type"] = "maintenance"
        elif any(word in narration for word in ["LATE", "PENALTY"]):
            fees.loc[row_idx, "fee_type"] = "penalty"

    return (
        fees[[
            "parsed_date", "Narration", "amount", "type", "fee_type", "category",
        ]]
        .to_dict("records")
    )






# Constants for readability
MONTHLY_RANGE = (25, 35)
WEEKLY_RANGE = (6, 8)
BIWEEKLY_RANGE = (13, 16)
QUARTERLY_RANGE = (85, 95)
ANNUAL_RANGE = (355, 370)
DAILY_WORKDAYS_THRESHOLD = 5
HIGH_REGULARITY_THRESHOLD = 3
MEDIUM_REGULARITY_THRESHOLD = 5
LOW_REGULARITY_THRESHOLD = 10
MIN_OCCURRENCES = 3
AMOUNT_VARIATION_THRESHOLD = 0.15


def detect_recurring(transactions_df: pd.DataFrame) -> list[dict]:
    """Detect recurring payments or deposits with enhanced analysis."""
    recurring = []

    # Withdrawals detection
    w_df = transactions_df[transactions_df["Withdrawal (INR)"] > 0].copy()
    detect_exact_amount_recurring(w_df, "withdrawal", recurring)
    detect_similar_amount_recurring(w_df, "withdrawal", recurring)

    # Deposits detection
    d_df = transactions_df[transactions_df["Deposit (INR)"] > 0].copy()
    detect_exact_amount_recurring(d_df, "deposit", recurring)
    detect_similar_amount_recurring(d_df, "deposit", recurring)

    return recurring


def determine_frequency(days_delta: list[int]) -> str:
    """Determine the frequency of recurring transactions."""
    mean_delta = round(np.mean(days_delta))

    # Define frequency determination logic
    if MONTHLY_RANGE[0] <= mean_delta <= MONTHLY_RANGE[1]:
        frequency = "monthly"
    elif WEEKLY_RANGE[0] <= mean_delta <= WEEKLY_RANGE[1]:
        frequency = "weekly"
    elif BIWEEKLY_RANGE[0] <= mean_delta <= BIWEEKLY_RANGE[1]:
        frequency = "biweekly"
    elif QUARTERLY_RANGE[0] <= mean_delta <= QUARTERLY_RANGE[1]:
        frequency = "quarterly"
    elif ANNUAL_RANGE[0] <= mean_delta <= ANNUAL_RANGE[1]:
        frequency = "annual"
    elif mean_delta <= DAILY_WORKDAYS_THRESHOLD:
        frequency = "daily/workdays"
    else:
        frequency = f"approximately every {mean_delta} days"

    return frequency


def detect_exact_amount_recurring(
    df: pd.DataFrame, transaction_type: str, recurring: list[dict],
) -> None:
    """Detect recurring transactions with exact amount matches."""
    grouped = df.groupby(["Narration", f"{transaction_type.capitalize()} (INR)"]).agg(
        dates=("parsed_date", list),
        count=("parsed_date", "count"),
        category=("category", "first"),
    )
    for (narration, amount), row in grouped.iterrows():
        if row["count"] > 1:
            dates = sorted(pd.to_datetime(row["dates"]))
            deltas = np.diff(dates).astype("timedelta64[D]").astype(int)

            if len(deltas) > 0 and np.std(deltas) < MEDIUM_REGULARITY_THRESHOLD:
                regularity = (
                    "high" if np.std(deltas) < HIGH_REGULARITY_THRESHOLD else "medium"
                )
                recurring.append({
                    "narration": narration,
                    "amount": amount,
                    "frequency": determine_frequency(deltas),
                    "category": row["category"],
                    "type": transaction_type,
                    "match_type": "exact_amount",
                    "regularity": regularity,
                    "first_date": dates[0].strftime("%Y-%m-%d"),
                    "last_date": dates[-1].strftime("%Y-%m-%d"),
                    "occurrence_count": row["count"],
                })


def detect_similar_amount_recurring(
    df: pd.DataFrame, transaction_type: str, recurring: list[dict],
) -> None:
    """Detect recurring transactions with similar amounts."""
    for narration in df["Narration"].unique():
        narr_df = df[df["Narration"] == narration]
        if len(narr_df) >= MIN_OCCURRENCES:
            amount_mean = narr_df[f"{transaction_type.capitalize()} (INR)"].mean()
            amount_std = narr_df[f"{transaction_type.capitalize()} (INR)"].std()

            if 0 < amount_std < amount_mean * AMOUNT_VARIATION_THRESHOLD:
                dates = sorted(narr_df["parsed_date"].tolist())
                deltas = np.diff(dates).astype("timedelta64[D]").astype(int)

                if len(deltas) > 0 and np.std(deltas) < LOW_REGULARITY_THRESHOLD:
                    regularity = (
                        "medium" if np.std(deltas) <HIGH_REGULARITY_THRESHOLD else "low"
                    )
                    recurring.append({
                        "narration": narration,
                        "amount": f"{amount_mean:.2f} (±{amount_std:.2f})",
                        "frequency": determine_frequency(deltas),
                        "category": narr_df["category"].iloc[0],
                        "type": transaction_type,
                        "match_type": "similar_amounts",
                        "regularity": regularity,
                        "first_date": dates[0].strftime("%Y-%m-%d"),
                        "last_date": dates[-1].strftime("%Y-%m-%d"),
                        "occurrence_count": len(narr_df),
                    })




# Constants for readability
MIN_DATA_POINTS = 5
Z_SCORE_EXTREME_THRESHOLD = 5
Z_SCORE_HIGH_THRESHOLD = 3
SPIKE_THRESHOLD = 5
LARGE_SPIKE_THRESHOLD = 10
TIME_GAP_MULTIPLIER = 3


def detect_anomalies(transactions_df: pd.DataFrame) -> list[dict]:
    """Flag unusual transactions by category with enhanced detection."""
    anomalies = []

    # Process each category separately
    for category in transactions_df["category"].unique():
        cat_df = transactions_df[transactions_df["category"] == category].copy()

        # Detect anomalies for withdrawals
        detect_withdrawal_anomalies(cat_df, anomalies)

        # Detect anomalies for deposits
        detect_deposit_anomalies(cat_df, anomalies)

    # Detect frequency anomalies
    detect_frequency_anomalies(transactions_df, anomalies)

    return anomalies


def detect_withdrawal_anomalies(cat_df: pd.DataFrame, anomalies: list[dict]) -> None:
    """Detect anomalies for withdrawal transactions."""
    w_df = cat_df[cat_df["Withdrawal (INR)"] > 0]
    if not w_df.empty:
        mean_w = w_df["Withdrawal (INR)"].mean()
        std_w = w_df["Withdrawal (INR)"].std()

        # Standard statistical outlier detection
        threshold_w = mean_w + 3 * std_w if std_w > 0 else mean_w * 2
        w_anomalies = w_df[w_df["Withdrawal (INR)"] > threshold_w][
            ["parsed_date", "Narration", "Withdrawal (INR)", "category"]
        ].rename(columns={"Withdrawal (INR)": "amount"})

        # Add anomaly score and type
        if not w_anomalies.empty:
            w_anomalies["z_score"] = (
                w_anomalies["amount"] - mean_w
            ) / (std_w if std_w > 0 else mean_w)
            w_anomalies["severity"] = w_anomalies["z_score"].apply(
                lambda z: "extreme" if z > Z_SCORE_EXTREME_THRESHOLD
                else "high" if z > Z_SCORE_HIGH_THRESHOLD else "moderate",
            )
            w_anomalies["type"] = "withdrawal"
            w_anomalies["detection_method"] = "statistical"
            anomalies.extend(w_anomalies.to_dict("records"))

        # Time-based anomaly detection
        if len(w_df) >= MIN_DATA_POINTS:
            w_df = w_df.sort_values("parsed_date")
            w_df["prev_amount"] = w_df["Withdrawal (INR)"].shift(1)
            w_df["amount_change_ratio"] = w_df["Withdrawal (INR)"] / w_df["prev_amount"]

            spikes = w_df[(w_df["amount_change_ratio"] > SPIKE_THRESHOLD) &
                          (~w_df["prev_amount"].isna())][
                ["parsed_date", "Narration", "Withdrawal (INR)", "prev_amount",
                 "amount_change_ratio", "category"]
            ].rename(columns={"Withdrawal (INR)": "amount"})

            if not spikes.empty:
                spikes["severity"] = spikes["amount_change_ratio"].apply(
                    lambda ratio: "extreme" if ratio>LARGE_SPIKE_THRESHOLD else "high",
                )
                spikes["type"] = "withdrawal"
                spikes["detection_method"] = "sudden_increase"
                anomalies.extend(spikes.to_dict("records"))


def detect_deposit_anomalies(cat_df: pd.DataFrame, anomalies: list[dict]) -> None:
    """Detect anomalies for deposit transactions."""
    d_df = cat_df[cat_df["Deposit (INR)"] > 0]
    if not d_df.empty:
        mean_d = d_df["Deposit (INR)"].mean()
        std_d = d_df["Deposit (INR)"].std()

        # Standard statistical outlier detection
        threshold_d = mean_d + 3 * std_d if std_d > 0 else mean_d * 2
        d_anomalies = d_df[d_df["Deposit (INR)"] > threshold_d][
            ["parsed_date", "Narration", "Deposit (INR)", "category"]
        ].rename(columns={"Deposit (INR)": "amount"})

        # Add anomaly score and type
        if not d_anomalies.empty:
            d_anomalies["z_score"] = (
                d_anomalies["amount"] - mean_d
            ) / (std_d if std_d > 0 else mean_d)
            d_anomalies["severity"] = d_anomalies["z_score"].apply(
                lambda z: "extreme" if z > Z_SCORE_EXTREME_THRESHOLD
                else "high" if z > Z_SCORE_HIGH_THRESHOLD else "moderate",
            )
            d_anomalies["type"] = "deposit"
            d_anomalies["detection_method"] = "statistical"
            anomalies.extend(d_anomalies.to_dict("records"))

        # Time-based anomaly detection
        if len(d_df) >= MIN_DATA_POINTS:
            d_df = d_df.sort_values("parsed_date")
            d_df["prev_amount"] = d_df["Deposit (INR)"].shift(1)
            d_df["amount_change_ratio"] = d_df["Deposit (INR)"] / d_df["prev_amount"]

            spikes = d_df[(d_df["amount_change_ratio"] > SPIKE_THRESHOLD) &
                          (~d_df["prev_amount"].isna())][
                ["parsed_date", "Narration", "Deposit (INR)", "prev_amount",
                 "amount_change_ratio", "category"]
            ].rename(columns={"Deposit (INR)": "amount"})

            if not spikes.empty:
                spikes["severity"] = spikes["amount_change_ratio"].apply(
                    lambda ratio: "extreme" if ratio>LARGE_SPIKE_THRESHOLD else "high",
                )
                spikes["type"] = "deposit"
                spikes["detection_method"] = "sudden_increase"
                anomalies.extend(spikes.to_dict("records"))


def detect_frequency_anomalies(transactions_df: pd.DataFrame,
                               anomalies: list[dict]) -> None:
    """Detect anomalies based on transaction frequency."""
    if not transactions_df.empty:
        df_sorted = transactions_df.sort_values("parsed_date")
        df_sorted["next_trans_days"] = (
            df_sorted["parsed_date"].shift(-1) - df_sorted["parsed_date"]
        ).dt.days

        mean_gap = df_sorted["next_trans_days"].mean()
        std_gap = df_sorted["next_trans_days"].std()

        large_gaps = df_sorted[df_sorted["next_trans_days"] > mean_gap +
                               TIME_GAP_MULTIPLIER * std_gap]
        if not large_gaps.empty:
            for _, row in large_gaps.iterrows():
                if pd.notna(row["next_trans_days"]):
                    anomalies.append({
                        "parsed_date": row["parsed_date"],
                        "Narration": f"""Unusual gap after this transaction
                        ({row['next_trans_days']} days)""",
                        "amount": 0,
                        "type": "gap",
                        "category": "timing_anomaly",
                        "severity": "moderate",
                        "detection_method": "timing_gap",
                    })


def analyze_cash_flow(df: pd.DataFrame) -> list[dict]:
    """Analyze cash flow with simplified output."""
    cash_flow_analysis = []

    if df.empty or "parsed_date" not in df.columns:
        return cash_flow_analysis

    df_copy = df.copy()
    df_copy["parsed_date"] = pd.to_datetime(df_copy["parsed_date"], errors="coerce")
    df_copy["month"] = df_copy["parsed_date"].dt.to_period("M")

    # Monthly cash flow
    monthly_cf = df_copy.groupby("month").agg({
        "Deposit (INR)": "sum",
        "Withdrawal (INR)": "sum",
    }).reset_index()

    monthly_cf["net_cash_flow"] = (
        monthly_cf["Deposit (INR)"] - monthly_cf["Withdrawal (INR)"]
    )
    monthly_cf["month"] = monthly_cf["month"].astype(str)

    # Add basic cash flow metrics for each month
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
