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

import mlflow
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from src.models import (
    AccountOverview,
    AnalyzerInput,
    AnalyzerOutput,
    Anomaly,
    CashFlow,
    Fee,
    Pattern,
    Recurring,
)
from src.utils import sanitize_metric_name, setup_mlflow

logger = logging.getLogger(__name__)

# Constants
HIGH_SPENDING_MULTIPLIER = 1.5
LOW_SPENDING_MULTIPLIER = 0.7
WEEKEND_MULTIPLIER = 1.3
MIN_MONTHS_FOR_TREND = 2
WITHDRAWAL_THRESHOLD_MULTIPLIER = 0.2
MIN_REPEAT_COUNT = 2
MONTHLY_RANGE = (25, 35)
WEEKLY_RANGE = (6, 8)
BIWEEKLY_RANGE = (13, 16)
QUARTERLY_RANGE = (85, 95)
ANNUAL_RANGE = (355, 370)
DAILY_WORKDAYS_THRESHOLD = 5
HIGH_REGULARITY_THRESHOLD = 3
MIN_OCCURRENCES = 3
AMOUNT_VARIATION_THRESHOLD = 0.15
MIN_DATA_POINTS = 5
Z_SCORE_THRESHOLD = 3
SPIKE_THRESHOLD = 5
TIME_GAP_MULTIPLIER = 3
START_MONTH_DAY = 10
MID_MONTH_DAY = 20
Z_SCORE_SEVERITY_THRESHOLD = 5

def _load_and_validate_data(input_csv: Path, required_cols: list[str]) -> pd.DataFrame:
    """Load and validate CSV data."""
    start_time = pd.Timestamp.now()
    try:
        transactions_df = pd.read_csv(input_csv)
        elapsed = (pd.Timestamp.now() - start_time).total_seconds()
        logger.info("Read CSV in %.3fs", elapsed)
    except FileNotFoundError:
        logger.exception("Input CSV not found: %s", input_csv)
        return pd.DataFrame()
    if transactions_df.empty:
        logger.warning("Empty CSV: %s", input_csv)
        return pd.DataFrame()
    missing_cols = [col for col in required_cols if col not in transactions_df]
    if missing_cols:
        logger.error("Missing columns: %s", missing_cols)
        return pd.DataFrame()
    return transactions_df

def _preprocess_data(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess transaction data."""
    start_time = pd.Timestamp.now()
    transactions_df["parsed_date"] = pd.to_datetime(
        transactions_df["parsed_date"], errors="coerce",
    )
    transactions_df["month"] = transactions_df["parsed_date"].dt.to_period("M")
    transactions_df["weekday"] = transactions_df["parsed_date"].dt.weekday
    transactions_df["is_weekend"] = transactions_df["weekday"].isin([5, 6])
    transactions_df["day"] = transactions_df["parsed_date"].dt.day
    transactions_df["time_of_month"] = transactions_df["day"].apply(
        lambda x: "start" if x <= START_MONTH_DAY
        else "middle" if x <= MID_MONTH_DAY
        else "end",
    )
    elapsed = (pd.Timestamp.now() - start_time).total_seconds()
    logger.info("Preprocessed data in %.3fs", elapsed)
    return transactions_df

def _compute_account_overview(transactions_df: pd.DataFrame) -> AccountOverview:
    """Compute account overview metrics."""
    start_time = pd.Timestamp.now()
    total_income = transactions_df["Deposit (INR)"].sum()
    total_expense = transactions_df["Withdrawal (INR)"].sum()
    total_balance = total_income - total_expense
    latest_month = transactions_df["month"].max()
    prev_month = latest_month - 1
    latest_income = transactions_df[
        transactions_df["month"] == latest_month
    ]["Deposit (INR)"].sum()
    latest_expense = transactions_df[
        transactions_df["month"] == latest_month
    ]["Withdrawal (INR)"].sum()
    prev_income = transactions_df[
        transactions_df["month"] == prev_month
    ]["Deposit (INR)"].sum()
    prev_expense = transactions_df[
        transactions_df["month"] == prev_month
    ]["Withdrawal (INR)"].sum()
    prev_balance = prev_income - prev_expense

    overview = AccountOverview(
        total_balance=float(total_balance),
        monthly_income=float(latest_income),
        monthly_expense=float(latest_expense),
        balance_percentage=float(
            ((total_balance - prev_balance) / prev_balance * 100)
            if prev_balance else 0,
        ),
        income_percentage=float(
            ((latest_income - prev_income) / prev_income * 100)
            if prev_income else 0,
        ),
        expense_percentage=float(
            ((latest_expense - prev_expense) / prev_expense * 100)
            if prev_expense else 0,
        ),
    )
    mlflow.log_metrics({
        "total_balance": total_balance,
        "monthly_income": latest_income,
        "monthly_expense": latest_expense,
    })
    elapsed = (pd.Timestamp.now() - start_time).total_seconds()
    logger.info("Computed account overview in %.3fs", elapsed)
    return overview

def _analyze_patterns(transactions_df: pd.DataFrame, output_dir: Path) -> list[Pattern]:
    """Analyze spending patterns and save results."""
    start_time = pd.Timestamp.now()
    patterns = detect_patterns(transactions_df)
    pattern_models = [Pattern(description=p) for p in patterns]
    patterns_file = output_dir / "patterns.txt"
    with patterns_file.open("w") as file:
        file.write("\n".join(patterns))
    mlflow.log_artifact(str(patterns_file))
    elapsed = (pd.Timestamp.now() - start_time).total_seconds()
    logger.info("Detected patterns in %.3fs", elapsed)
    return pattern_models

def _analyze_fees(transactions_df: pd.DataFrame, output_dir: Path) -> list[Fee]:
    """Analyze fees and save results."""
    start_time = pd.Timestamp.now()
    fees = detect_fees(transactions_df)
    fee_models = [Fee(**f) for f in fees]
    fees_file = output_dir / "fees.csv"
    pd.DataFrame(fees).to_csv(fees_file, index=False)
    mlflow.log_artifact(str(fees_file))
    if not fees:
        logger.warning("No fee transactions detected")
    elapsed = (pd.Timestamp.now() - start_time).total_seconds()
    logger.info("Detected fees in %.3fs", elapsed)
    return fee_models

def _analyze_recurring(transactions_df: pd.DataFrame,
                    output_dir: Path) -> list[Recurring]:
    """Analyze recurring payments and save results."""
    start_time = pd.Timestamp.now()
    recurring = detect_recurring(transactions_df)
    recurring_models = [Recurring(**r) for r in recurring]
    recurring_file = output_dir / "recurring.csv"
    pd.DataFrame(recurring).to_csv(recurring_file, index=False)
    mlflow.log_artifact(str(recurring_file))
    elapsed = (pd.Timestamp.now() - start_time).total_seconds()
    logger.info("Detected recurring payments in %.3fs", elapsed)
    return recurring_models

def _analyze_anomalies(transactions_df: pd.DataFrame,
                       output_dir: Path) -> list[Anomaly]:
    """Analyze anomalies and save results."""
    start_time = pd.Timestamp.now()
    anomalies = detect_anomalies(transactions_df)
    anomalies_file = output_dir / "anomalies.csv"
    pd.DataFrame([a.dict() for a in anomalies]).to_csv(anomalies_file, index=False)
    mlflow.log_artifact(str(anomalies_file))
    elapsed = (pd.Timestamp.now() - start_time).total_seconds()
    logger.info("Detected anomalies in %.3fs", elapsed)
    return anomalies

def _analyze_cash_flow(transactions_df: pd.DataFrame,
                       output_dir: Path) -> list[CashFlow]:
    """Analyze cash flow and save results."""
    start_time = pd.Timestamp.now()
    cash_flow = analyze_cash_flow(transactions_df)
    cash_flow_models = [CashFlow(**c) for c in cash_flow]
    cash_flow_file = output_dir / "cash_flow.csv"
    pd.DataFrame(cash_flow).to_csv(cash_flow_file, index=False)
    mlflow.log_artifact(str(cash_flow_file))
    elapsed = (pd.Timestamp.now() - start_time).total_seconds()
    logger.info("Analyzed cash flow in %.3fs", elapsed)
    return cash_flow_models

def analyze_transactions(input_model: AnalyzerInput) -> AnalyzerOutput:
    """Analyze transactions to identify financial patterns and metrics.

    Args:
        input_model: Input configuration with transaction CSV path and output directory.

    Returns:
        Analysis results including patterns, fees, recurring payments, anomalies,
        and cash flow.

    """
    setup_mlflow()
    logger.info("Starting transaction analysis")
    input_csv = input_model.input_csv
    output_dir = input_model.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = AnalyzerOutput(
        patterns=[],
        fees=[],
        recurring=[],
        anomalies=[],
        cash_flow=[],
        account_overview=AccountOverview(
            total_balance=0.0,
            monthly_income=0.0,
            monthly_expense=0.0,
            balance_percentage=0.0,
            income_percentage=0.0,
            expense_percentage=0.0,
        ),
    )

    with mlflow.start_run(run_name="Transaction_Analysis"):
        mlflow.log_param("input_csv", str(input_csv))
        start_time = pd.Timestamp.now()

        required_cols = [
            "parsed_date", "Narration", "Withdrawal (INR)",
            "Deposit (INR)", "category",
        ]
        transactions_df = _load_and_validate_data(input_csv, required_cols)
        if transactions_df.empty:
            return results

        mlflow.log_metric("transactions_analyzed", len(transactions_df))
        transactions_df = _preprocess_data(transactions_df)
        results.account_overview = _compute_account_overview(transactions_df)
        results.patterns = _analyze_patterns(transactions_df, output_dir)
        results.fees = _analyze_fees(transactions_df, output_dir)
        results.recurring = _analyze_recurring(transactions_df, output_dir)
        results.anomalies = _analyze_anomalies(transactions_df, output_dir)
        results.cash_flow = _analyze_cash_flow(transactions_df, output_dir)

        # Log counts
        mlflow.log_metrics({
            sanitize_metric_name("patterns_count"): len(results.patterns),
            sanitize_metric_name("fees_count"): len(results.fees),
            sanitize_metric_name("recurring_count"): len(results.recurring),
            sanitize_metric_name("anomalies_count"): len(results.anomalies),
            sanitize_metric_name("cash_flow_count"): len(results.cash_flow),
        })

        # Log account overview metrics
        for subkey, value in results.account_overview.dict().items():
            mlflow.log_metric(sanitize_metric_name(f"account_{subkey}"), value)

        total_elapsed = (pd.Timestamp.now() - start_time).total_seconds()
        logger.info("Completed analysis in %.3fs", total_elapsed)
        return results

def detect_patterns(transactions_df: pd.DataFrame) -> list[str]:
    """Identify spending patterns using rule-based logic."""
    logger.info("Detecting spending patterns")
    patterns = []
    if transactions_df.empty or transactions_df["parsed_date"].isna().all():
        logger.warning("No valid data for patterns")
        return ["No patterns detected"]

    # Category dominance
    cat_totals = transactions_df.groupby("category")["Withdrawal (INR)"].sum()
    if not cat_totals.empty:
        top_cat = cat_totals.idxmax()
        if cat_totals[top_cat] > cat_totals.mean() * HIGH_SPENDING_MULTIPLIER:
            patterns.append(
                f"High {top_cat} spending (₹{cat_totals[top_cat]:.2f})",
            )

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

    # Weekend vs weekday
    weekend_mask = transactions_df["is_weekend"]
    weekend_sum = transactions_df[weekend_mask]["Withdrawal (INR)"].sum()
    weekend_count = weekend_mask.sum()
    weekday_sum = transactions_df[~weekend_mask]["Withdrawal (INR)"].sum()
    weekday_count = (~weekend_mask).sum()
    weekend_spending = weekend_sum / weekend_count if weekend_count else 0
    weekday_spending = weekday_sum / weekday_count if weekday_count else 0
    if weekend_spending > weekday_spending * WEEKEND_MULTIPLIER:
        patterns.append(
            f"Higher weekend spending (₹{weekend_spending:.2f}/day) vs "
            f"weekdays (₹{weekday_spending:.2f}/day)",
        )
    elif weekday_spending > weekend_spending * WEEKEND_MULTIPLIER:
        patterns.append(
            f"Higher weekday spending (₹{weekday_spending:.2f}/day) vs "
            f"weekends (₹{weekend_spending:.2f}/day)",
        )

    return patterns if patterns else ["No patterns detected"]

def detect_fees(transactions_df: pd.DataFrame) -> list[dict]:
    """Identify fee or interest-related transactions."""
    keywords = [
        "FEE", "CHARGE", "INTEREST", "PENALTY", "TAX", "COMMISSION",
        "SERVICE CHARGE", "LATE FEE", "SURCHARGE", "GST",
        "MAINTENANCE", "AMC", "ANNUAL",
    ]
    mask = transactions_df["Narration"].str.upper().str.contains(
        "|".join(keywords), na=False,
    )
    fees_df = transactions_df[mask][[
        "parsed_date", "Narration", "Withdrawal (INR)", "Deposit (INR)", "category",
    ]].copy()

    # Small recurring fees
    mean_withdrawal = transactions_df["Withdrawal (INR)"].mean()
    potential_fee_mask = (
        (transactions_df["Withdrawal (INR)"] > 0) &
        (transactions_df["Withdrawal (INR)"] <
         mean_withdrawal * WITHDRAWAL_THRESHOLD_MULTIPLIER) &
        (~mask)
    )
    potential_fees = transactions_df[potential_fee_mask]
    if not potential_fees.empty:
        amount_counts = potential_fees.groupby("Withdrawal (INR)").size()
        recurring_amounts = amount_counts[amount_counts >= MIN_REPEAT_COUNT].index
        recurring_fees = potential_fees[
            potential_fees["Withdrawal (INR)"].isin(recurring_amounts)
        ]
        fees_df = pd.concat([
            fees_df,
            recurring_fees[[
                "parsed_date", "Narration", "Withdrawal (INR)",
                "Deposit (INR)", "category",
            ]],
        ])

    fees_df["amount"] = fees_df["Withdrawal (INR)"].where(
        fees_df["Withdrawal (INR)"] > 0, fees_df["Deposit (INR)"],
    )
    fees_df["type"] = fees_df["Withdrawal (INR)"].where(
        fees_df["Withdrawal (INR)"] > 0, 0,
    ).apply(lambda x: "withdrawal" if x > 0 else "deposit")
    fees_df["fee_type"] = fees_df["Narration"].str.upper().apply(
        lambda x: (
            "interest" if "INTEREST" in x else
            "tax" if any(w in x for w in ["TAX", "GST"]) else
            "annual" if any(w in x for w in ["ANNUAL", "AMC", "YEARLY"]) else
            "maintenance" if any(w in x for w in ["MAINTENANCE", "SERVICE"]) else
            "penalty" if any(w in x for w in ["LATE", "PENALTY"]) else
            "other"
        ),
    )

    return fees_df[[
        "parsed_date", "Narration", "amount", "type", "fee_type", "category",
    ]].to_dict("records")

def detect_recurring(transactions_df: pd.DataFrame) -> list[dict]:
    """Detect recurring payments or deposits."""
    recurring = []

    # Withdrawals
    withdrawals_df = transactions_df[transactions_df["Withdrawal (INR)"] > 0][[
        "parsed_date", "Narration", "Withdrawal (INR)", "category",
    ]]
    if not withdrawals_df.empty:
        detect_exact_amount_recurring(
            withdrawals_df, "withdrawal", "Withdrawal (INR)", recurring,
        )
        detect_similar_amount_recurring(
            withdrawals_df, "withdrawal", "Withdrawal (INR)", recurring,
        )

    # Deposits
    deposits_df = transactions_df[transactions_df["Deposit (INR)"] > 0][[
        "parsed_date", "Narration", "Deposit (INR)", "category",
    ]]
    if not deposits_df.empty:
        detect_exact_amount_recurring(
            deposits_df, "deposit", "Deposit (INR)", recurring,
        )
        detect_similar_amount_recurring(
            deposits_df, "deposit", "Deposit (INR)", recurring,
        )

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
    return (
        f"approximately every {mean_delta} days"
        if mean_delta > DAILY_WORKDAYS_THRESHOLD else "daily/workdays"
    )

def detect_exact_amount_recurring(
    transactions_df: pd.DataFrame,
    transaction_type: str,
    amount_col: str,
    recurring: list[dict],
) -> None:
    """Detect recurring transactions with exact amounts."""
    grouped = transactions_df.groupby(["Narration", amount_col]).agg(
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

def detect_similar_amount_recurring(
    transactions_df: pd.DataFrame,
    transaction_type: str,
    amount_col: str,
    recurring: list[dict],
) -> None:
    """Detect recurring transactions with similar amounts."""
    grouped = transactions_df.groupby("Narration").agg(
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

def _detect_transaction_anomalies(
    transactions_df: pd.DataFrame,
    amount_col: str,
    trans_type: str,
) -> list[Anomaly]:
    """Detect anomalies for withdrawals or deposits."""
    anomalies = []
    stats = transactions_df.groupby("category")[amount_col].agg([
        "mean", "std",
    ]).fillna(0)
    merged_df = transactions_df.merge(stats, on="category", how="left")
    merged_df["z_score"] = (
        (merged_df[amount_col] - merged_df["mean"]) /
        merged_df["std"].replace(0, merged_df[amount_col].mean())
    )
    anomaly_rows = merged_df[merged_df["z_score"] > Z_SCORE_THRESHOLD][[
        "parsed_date", "Narration", amount_col, "category", "z_score",
    ]]
    if not anomaly_rows.empty:
        for _, row in anomaly_rows.iterrows():
            anomalies.append(Anomaly(
                parsed_date=row["parsed_date"],
                Narration=row["Narration"],
                amount=row[amount_col],
                type=trans_type,
                severity=(
                    "high" if row["z_score"] > Z_SCORE_SEVERITY_THRESHOLD
                    else "moderate"
                ),
                category=row["category"],
                detection_method="statistical",
            ))

    # Sudden spikes
    sorted_df = merged_df.sort_values("parsed_date")
    sorted_df["prev_amount"] = sorted_df.groupby("category")[amount_col].shift(1)
    sorted_df["ratio"] = sorted_df[amount_col] / sorted_df["prev_amount"]
    spikes = sorted_df[
        (sorted_df["ratio"] > SPIKE_THRESHOLD) & (sorted_df["prev_amount"].notna())
    ]
    if not spikes.empty:
        for _, row in spikes.iterrows():
            anomalies.append(Anomaly(
                parsed_date=row["parsed_date"],
                Narration=row["Narration"],
                amount=row[amount_col],
                type=trans_type,
                severity="high",
                category=row["category"],
                detection_method="sudden_increase",
            ))

    return anomalies

def detect_anomalies(transactions_df: pd.DataFrame) -> list[Anomaly]:
    """Flag unusual transactions by category."""
    anomalies = []

    # Withdrawals
    withdrawals_df = transactions_df[transactions_df["Withdrawal (INR)"] > 0].copy()
    if not withdrawals_df.empty:
        anomalies.extend(_detect_transaction_anomalies(
            withdrawals_df, "Withdrawal (INR)", "withdrawal",
        ))

    # Deposits
    deposits_df = transactions_df[transactions_df["Deposit (INR)"] > 0].copy()
    if not deposits_df.empty:
        anomalies.extend(_detect_transaction_anomalies(
            deposits_df, "Deposit (INR)", "deposit",
        ))

    # Frequency anomalies
    sorted_df = transactions_df.sort_values("parsed_date")
    sorted_df["next_trans_days"] = (
        sorted_df["parsed_date"].shift(-1) - sorted_df["parsed_date"]
    ).dt.days
    mean_gap = sorted_df["next_trans_days"].mean()
    std_gap = sorted_df["next_trans_days"].std()
    large_gaps = sorted_df[
        sorted_df["next_trans_days"] > mean_gap + TIME_GAP_MULTIPLIER * std_gap
    ]
    if not large_gaps.empty:
        for _, row in large_gaps.iterrows():
            if pd.notna(row["next_trans_days"]):
                anomalies.append(Anomaly(
                    parsed_date=row["parsed_date"],
                    Narration=(
                        f"Unusual gap after this transaction "
                        f"({row['next_trans_days']} days)"
                    ),
                    amount=0,
                    type="gap",
                    category="timing_anomaly",
                    severity="moderate",
                    detection_method="timing_gap",
                ))

    return anomalies

def analyze_cash_flow(transactions_df: pd.DataFrame) -> list[dict]:
    """Analyze cash flow."""
    cash_flow_analysis = []
    if transactions_df.empty or "parsed_date" not in transactions_df:
        return cash_flow_analysis

    monthly_cf = transactions_df.groupby("month").agg({
        "Deposit (INR)": "sum",
        "Withdrawal (INR)": "sum",
    }).reset_index()
    monthly_cf["net_cash_flow"] = (
        monthly_cf["Deposit (INR)"] - monthly_cf["Withdrawal (INR)"]
    )
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
    input_model = AnalyzerInput(
        input_csv=Path("data/output/categorized.csv"),
        output_dir=Path("data/output/analysis"),
    )
    results = analyze_transactions(input_model)
    logger.info("Analysis results: %s", results.dict())
