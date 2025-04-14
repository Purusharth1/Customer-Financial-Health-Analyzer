import logging
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from utils import get_llm_config, sanitize_metric_name, setup_mlflow

sys.path.append(str(Path(__file__).parent.parent))

from llm_setup.config import LLMConfig
from llm_setup.ollama_manager import query_llm

logging.basicConfig(level=logging.INFO)

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
    logging.info("Starting transaction analysis")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {"patterns": [], "fees": [], "recurring": [], "anomalies": [],
               "cash_flow": []}

    with mlflow.start_run(run_name="Transaction_Analysis"):
        mlflow.log_param("input_csv", input_csv)
        try:
            df = pd.read_csv(input_csv)
        except FileNotFoundError:
            logging.exception(f"Input CSV not found: {input_csv}")
            return results
        mlflow.log_metric("transactions_analyzed", len(df))

        # Validate columns
        required = ["parsed_date", "Narration", "Withdrawal (INR)", "Deposit (INR)", "category"]
        if not all(col in df for col in required):
            missing = [col for col in required if col not in df]
            logging.error(f"Missing columns: {missing}")
            return results

        df["parsed_date"] = pd.to_datetime(df["parsed_date"], errors="coerce")

        # Patterns
        patterns = detect_patterns(df, llm_config)
        results["patterns"] = patterns
        patterns_file = Path(output_dir) / "patterns.txt"
        with patterns_file.open("w") as file:
            file.write("\n".join(patterns))
        mlflow.log_artifact(str(patterns_file))

        # Fees
        fees = detect_fees(df)
        results["fees"] = fees
        fees_file = Path(output_dir) / "fees.csv"
        pd.DataFrame(fees).to_csv(fees_file, index=False)
        mlflow.log_artifact(str(fees_file))
        if not fees:
            logging.warning("No fee transactions detected")

        # Recurring
        recurring = detect_recurring(df)
        results["recurring"] = recurring
        recurring_file = Path(output_dir) / "recurring.csv"
        pd.DataFrame(recurring).to_csv(recurring_file, index=False)
        mlflow.log_artifact(str(recurring_file))

        # Anomalies
        anomalies = detect_anomalies(df)
        results["anomalies"] = anomalies
        anomalies_file = Path(output_dir) / "anomalies.csv"
        pd.DataFrame(anomalies).to_csv(anomalies_file, index=False)
        mlflow.log_artifact(str(anomalies_file))

        # Cash Flow
        cash_flow = analyze_cash_flow(df)
        results["cash_flow"] = cash_flow
        cash_flow_file = Path(output_dir) / "cash_flow.csv"
        pd.DataFrame(cash_flow).to_csv(cash_flow_file, index=False)
        mlflow.log_artifact(str(cash_flow_file))

        # Log counts
        for key, items in results.items():
            mlflow.log_metric(sanitize_metric_name(f"{key}_count"), len(items))

        logging.info("Analysis complete")
        return results

def detect_patterns(df: pd.DataFrame, llm_config: LLMConfig) -> list[str]:
    """Identify spending patterns using LLM or fallback rules.

    Args:
        df: DataFrame with categorized transactions.
        llm_config: LLM configuration.

    Returns:
        List of pattern strings.

    """
    logging.info("Detecting spending patterns")

    # Validate data
    if df.empty or df["parsed_date"].isna().all():
        logging.warning("No valid data for patterns")
        return ["No patterns detected"]

    df = df.copy()
    df["month"] = df["parsed_date"].dt.to_period("M")
    df["weekday"] = df["parsed_date"].dt.day_name()
    df["day"] = df["parsed_date"].dt.day
    df["week"] = df["parsed_date"].dt.isocalendar().week

    # Add more detailed time-based analysis
    df["time_of_month"] = df["day"].apply(
        lambda x: "start" if x <= 10 else "middle" if x <= 20 else "end",
    )

    # Summarize top categories and recent months
    categories = df["category"].unique().tolist()
    top_categories = (
        df.groupby("category")["Withdrawal (INR)"]
        .sum()
        .nlargest(5)  # Increased from 3 to 5
        .index
        .tolist()
    )
    recent_months = df["month"].unique()[-3:]  # Last 3 months

    # Enhanced summary with more metrics
    summary = {
        "categories": df[df["category"].isin(top_categories)]
        .groupby("category")[["Withdrawal (INR)", "Deposit (INR)"]]
        .sum()
        .round(2)
        .to_dict(),
        "monthly": df[df["month"].isin(recent_months)]
        .groupby(["month", "category"])[["Withdrawal (INR)", "Deposit (INR)"]]
        .sum()
        .round(2)
        .to_dict(),
        "weekday": df[df["category"].isin(top_categories)]
        .groupby(["weekday", "category"])["Withdrawal (INR)"]
        .sum()
        .round(2)
        .to_dict(),
        "time_of_month": df[df["category"].isin(top_categories)]
        .groupby(["time_of_month", "category"])["Withdrawal (INR)"]
        .sum()
        .round(2)
        .to_dict(),
        "counts": df.groupby("category").size().to_dict(),
        "weekend_vs_weekday": {
            "weekend": df[df["weekday"].isin(["Saturday", "Sunday"])]["Withdrawal (INR)"].sum().round(2),
            "weekday": df[~df["weekday"].isin(["Saturday", "Sunday"])]["Withdrawal (INR)"].sum().round(2),
        },
    }

    # LLM prompt with more guidance for detailed patterns
    prompt = (
        f"Analyze spending for categories {top_categories}. "
        f"Data: {summary}. "
        "Identify specific patterns like: "
        "1. High spending in specific categories "
        "2. Monthly or weekly spending trends "
        "3. Beginning/middle/end of month spending habits "
        "4. Weekend vs weekday spending differences "
        "5. Periodic spikes in specific categories "
        "6. Gradually increasing/decreasing trends "
        "Return one pattern per line, be specific about amounts and timing, e.g., "
        "'High retail spending of ₹XX,XXX in July' or 'Consistently higher food expenses on weekends (₹X,XXX vs ₹X,XXX)'. "
        "No explanations, provide 10 distinct patterns."
    )
    logging.debug(f"LLM prompt: {prompt}")

    # Try LLM
    for attempt in range(2):  # Retry once
        try:
            response = query_llm(prompt, llm_config, timeout=TIMEOUT_SECONDS).strip()
            logging.debug(f"LLM response: {response}")
            if response:
                patterns = [p.strip() for p in response.split("\n") if p.strip()]
                if patterns:
                    return patterns[:10]  # Increased limit to 10 patterns
            break
        except Exception as e:
            logging.warning(f"LLM attempt {attempt + 1} failed: {e}")

    # Enhanced fallback: Rule-based patterns
    logging.info("Falling back to rule-based patterns")
    patterns = []

    # Category dominance
    cat_totals = df.groupby("category")["Withdrawal (INR)"].sum()
    if not cat_totals.empty:
        top_cat = cat_totals.idxmax()
        if cat_totals[top_cat] > cat_totals.mean() * 1.5:
            patterns.append(f"High {top_cat} spending (₹{cat_totals[top_cat]:.2f})")

        # Category growth/decline
        if len(df["month"].unique()) >= 2:
            for cat in top_categories:
                cat_monthly = df[df["category"] == cat].groupby("month")["Withdrawal (INR)"].sum()
                if len(cat_monthly) >= 2:
                    months = sorted(cat_monthly.index)
                    if cat_monthly[months[-1]] > cat_monthly[months[0]] * 1.5:
                        patterns.append(f"Increasing {cat} expenses over time")
                    elif cat_monthly[months[-1]] < cat_monthly[months[0]] * 0.67:
                        patterns.append(f"Decreasing {cat} expenses over time")

    # Monthly trends
    monthly_totals = df.groupby("month")["Withdrawal (INR)"].sum()
    if len(monthly_totals) > 1:
        max_month = monthly_totals.idxmax()
        min_month = monthly_totals.idxmin()
        if monthly_totals[max_month] > monthly_totals.mean() * 1.3:
            patterns.append(f"Higher spending in {max_month} (₹{monthly_totals[max_month]:.2f})")
        if monthly_totals[min_month] < monthly_totals.mean() * 0.7:
            patterns.append(f"Lower spending in {min_month} (₹{monthly_totals[min_month]:.2f})")

    # Weekday vs weekend spikes
    weekday_totals = df.groupby("weekday")["Withdrawal (INR)"].sum()
    weekend_avg = (weekday_totals.get("Saturday", 0) + weekday_totals.get("Sunday", 0)) / 2
    weekday_avg = sum([weekday_totals.get(day, 0) for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]]) / 5

    if weekend_avg > weekday_avg * 1.3:
        patterns.append(f"Higher weekend spending (₹{weekend_avg:.2f}/day) vs weekdays (₹{weekday_avg:.2f}/day)")
    elif weekday_avg > weekend_avg * 1.3:
        patterns.append(f"Higher weekday spending (₹{weekday_avg:.2f}/day) vs weekends (₹{weekend_avg:.2f}/day)")

    # Day of month patterns
    time_of_month_totals = df.groupby("time_of_month")["Withdrawal (INR)"].sum()
    if "start" in time_of_month_totals and "end" in time_of_month_totals:
        if time_of_month_totals["start"] > time_of_month_totals["end"] * 1.5:
            patterns.append("Much higher spending at the beginning of months")
        elif time_of_month_totals["end"] > time_of_month_totals["start"] * 1.5:
            patterns.append("Much higher spending at the end of months")

    # Look for specific high-spending days
    day_totals = df.groupby(["month", "day"])["Withdrawal (INR)"].sum().reset_index()
    if not day_totals.empty:
        high_days = day_totals[day_totals["Withdrawal (INR)"] > day_totals["Withdrawal (INR)"].mean() * 2]
        if not high_days.empty:
            for _, row in high_days.head(2).iterrows():
                patterns.append(f"Spending spike on {row['month']}-{row['day']} (₹{row['Withdrawal (INR)']:.2f})")

    # Look for category-specific day of week patterns
    for cat in top_categories[:3]:
        cat_df = df[df["category"] == cat]
        if not cat_df.empty:
            cat_day = cat_df.groupby("weekday")["Withdrawal (INR)"].sum()
            if not cat_day.empty:
                max_day = cat_day.idxmax()
                patterns.append(f"Highest {cat} spending on {max_day}s")

    return patterns if patterns else ["No patterns detected"]

def detect_fees(df: pd.DataFrame) -> list[dict]:
    """Identify fee or interest-related transactions with enhanced detection."""
    # Enhanced keywords list for better detection
    keywords = ["FEE", "CHARGE", "INTEREST", "PENALTY", "TAX", "COMMISSION", "SERVICE CHARGE",
                "LATE FEE", "SURCHARGE", "GST", "MAINTENANCE", "AMC", "ANNUAL"]

    # Include both withdrawals and deposits (e.g., credited interest)
    mask = df["Narration"].str.upper().str.contains("|".join(keywords), na=False)
    fees = df[mask][["parsed_date", "Narration", "Withdrawal (INR)", "Deposit (INR)", "category"]].copy()

    # Also look for small, regular withdrawals that might be subscription fees
    potential_fee_mask = (
        (df["Withdrawal (INR)"] > 0) &
        (df["Withdrawal (INR)"] < df["Withdrawal (INR)"].mean() * 0.2) &
        (~mask)  # Exclude already identified fees
    )
    potential_fees = df[potential_fee_mask]

    # Check if these small amounts repeat
    if not potential_fees.empty:
        for amount in potential_fees["Withdrawal (INR)"].unique():
            amount_df = potential_fees[potential_fees["Withdrawal (INR)"] == amount]
            if len(amount_df) >= 2:  # If this exact amount appears multiple times
                fees = pd.concat([fees, amount_df[["parsed_date", "Narration", "Withdrawal (INR)", "Deposit (INR)", "category"]]])

    # Add more metadata to each fee
    fees["amount"] = fees["Withdrawal (INR)"].where(fees["Withdrawal (INR)"] > 0, fees["Deposit (INR)"])
    fees["type"] = fees["Withdrawal (INR)"].where(fees["Withdrawal (INR)"] > 0, 0).apply(
        lambda x: "withdrawal" if x > 0 else "deposit",
    )

    # Add fee categorization
    fees["fee_type"] = "other"
    for row_idx, row in fees.iterrows():
        narration = row["Narration"].upper()
        if "INTEREST" in narration:
            fees.at[row_idx, "fee_type"] = "interest"
        elif any(word in narration for word in ["TAX", "GST"]):
            fees.at[row_idx, "fee_type"] = "tax"
        elif any(word in narration for word in ["ANNUAL", "AMC", "YEARLY"]):
            fees.at[row_idx, "fee_type"] = "annual"
        elif any(word in narration for word in ["MAINTENANCE", "SERVICE"]):
            fees.at[row_idx, "fee_type"] = "maintenance"
        elif any(word in narration for word in ["LATE", "PENALTY"]):
            fees.at[row_idx, "fee_type"] = "penalty"

    return fees[["parsed_date", "Narration", "amount", "type", "fee_type", "category"]].to_dict("records")

def detect_recurring(df: pd.DataFrame) -> list[dict]:
    """Detect recurring payments or deposits with enhanced analysis."""
    recurring = []

    # Helper function to determine frequency more accurately
    def determine_frequency(days_delta):
        mean_delta = np.mean(days_delta)
        if 25 <= mean_delta <= 35:
            return "monthly"
        if 6 <= mean_delta <= 8:
            return "weekly"
        if 13 <= mean_delta <= 16:
            return "biweekly"
        if 85 <= mean_delta <= 95:
            return "quarterly"
        if 355 <= mean_delta <= 370:
            return "annual"
        if mean_delta <= 5:
            return "daily/workdays"
        return f"approximately every {int(round(mean_delta))} days"

    # Withdrawals with enhanced detection
    w_df = df[df["Withdrawal (INR)"] > 0].copy()

    # First look for exact amount matches (traditional approach)
    grouped_w = w_df.groupby(["Narration", "Withdrawal (INR)"]).agg(
        dates=("parsed_date", list),
        count=("parsed_date", "count"),
        category=("category", "first"),
    )
    for (narration, amount), row in grouped_w.iterrows():
        if row["count"] > 1:
            dates = sorted(pd.to_datetime(row["dates"]))
            deltas = np.diff(dates).astype("timedelta64[D]").astype(int)

            if len(deltas) > 0 and np.std(deltas) < 5:
                recurring.append({
                    "narration": narration,
                    "amount": amount,
                    "frequency": determine_frequency(deltas),
                    "category": row["category"],
                    "type": "withdrawal",
                    "match_type": "exact_amount",
                    "regularity": "high" if np.std(deltas) < 3 else "medium",
                    "first_date": dates[0].strftime("%Y-%m-%d"),
                    "last_date": dates[-1].strftime("%Y-%m-%d"),
                    "occurrence_count": row["count"],
                })

    # Now look for similar amounts with same description (looser match)
    for narration in w_df["Narration"].unique():
        narr_df = w_df[w_df["Narration"] == narration]
        if len(narr_df) >= 3:  # Need at least 3 to establish a pattern
            amount_mean = narr_df["Withdrawal (INR)"].mean()
            amount_std = narr_df["Withdrawal (INR)"].std()

            # If amounts are similar but not identical (e.g., utility bills)
            if 0 < amount_std < amount_mean * 0.15:  # Allow 15% variation
                dates = sorted(narr_df["parsed_date"].tolist())
                deltas = np.diff(dates).astype("timedelta64[D]").astype(int)

                if len(deltas) > 0 and np.std(deltas) < 10:  # More lenient on timing
                    recurring.append({
                        "narration": narration,
                        "amount": f"{amount_mean:.2f} (±{amount_std:.2f})",
                        "frequency": determine_frequency(deltas),
                        "category": narr_df["category"].iloc[0],
                        "type": "withdrawal",
                        "match_type": "similar_amounts",
                        "regularity": "medium" if np.std(deltas) < 5 else "low",
                        "first_date": dates[0].strftime("%Y-%m-%d"),
                        "last_date": dates[-1].strftime("%Y-%m-%d"),
                        "occurrence_count": len(narr_df),
                    })

    # Deposits with same enhancements
    d_df = df[df["Deposit (INR)"] > 0].copy()

    # Exact amount matches
    grouped_d = d_df.groupby(["Narration", "Deposit (INR)"]).agg(
        dates=("parsed_date", list),
        count=("parsed_date", "count"),
        category=("category", "first"),
    )
    for (narration, amount), row in grouped_d.iterrows():
        if row["count"] > 1:
            dates = sorted(pd.to_datetime(row["dates"]))
            deltas = np.diff(dates).astype("timedelta64[D]").astype(int)

            if len(deltas) > 0 and np.std(deltas) < 5:
                recurring.append({
                    "narration": narration,
                    "amount": amount,
                    "frequency": determine_frequency(deltas),
                    "category": row["category"],
                    "type": "deposit",
                    "match_type": "exact_amount",
                    "regularity": "high" if np.std(deltas) < 3 else "medium",
                    "first_date": dates[0].strftime("%Y-%m-%d"),
                    "last_date": dates[-1].strftime("%Y-%m-%d"),
                    "occurrence_count": row["count"],
                })

    # Similar amount matches
    for narration in d_df["Narration"].unique():
        narr_df = d_df[d_df["Narration"] == narration]
        if len(narr_df) >= 3:
            amount_mean = narr_df["Deposit (INR)"].mean()
            amount_std = narr_df["Deposit (INR)"].std()

            if 0 < amount_std < amount_mean * 0.15:
                dates = sorted(narr_df["parsed_date"].tolist())
                deltas = np.diff(dates).astype("timedelta64[D]").astype(int)

                if len(deltas) > 0 and np.std(deltas) < 10:
                    recurring.append({
                        "narration": narration,
                        "amount": f"{amount_mean:.2f} (±{amount_std:.2f})",
                        "frequency": determine_frequency(deltas),
                        "category": narr_df["category"].iloc[0],
                        "type": "deposit",
                        "match_type": "similar_amounts",
                        "regularity": "medium" if np.std(deltas) < 5 else "low",
                        "first_date": dates[0].strftime("%Y-%m-%d"),
                        "last_date": dates[-1].strftime("%Y-%m-%d"),
                        "occurrence_count": len(narr_df),
                    })

    return recurring

def detect_anomalies(df: pd.DataFrame) -> list[dict]:
    """Flag unusual transactions by category with enhanced detection."""
    anomalies = []

    # Process each category separately
    for category in df["category"].unique():
        cat_df = df[df["category"] == category].copy()

        # Withdrawals - standard method plus enhanced detection
        w_df = cat_df[cat_df["Withdrawal (INR)"] > 0]
        if not w_df.empty:
            mean_w = w_df["Withdrawal (INR)"].mean()
            std_w = w_df["Withdrawal (INR)"].std()

            # Standard statistical outlier detection (3 std)
            threshold_w = mean_w + 3 * std_w if std_w > 0 else mean_w * 2
            w_anomalies = w_df[w_df["Withdrawal (INR)"] > threshold_w][
                ["parsed_date", "Narration", "Withdrawal (INR)", "category"]
            ].rename(columns={"Withdrawal (INR)": "amount"})

            # Add anomaly score and type
            if not w_anomalies.empty:
                for idx, row in w_anomalies.iterrows():
                    z_score = (row["amount"] - mean_w) / (std_w if std_w > 0 else mean_w)
                    w_anomalies.at[idx, "z_score"] = z_score
                    w_anomalies.at[idx, "severity"] = "extreme" if z_score > 5 else "high" if z_score > 3 else "moderate"

                w_anomalies["type"] = "withdrawal"
                w_anomalies["detection_method"] = "statistical"
                anomalies.extend(w_anomalies.to_dict("records"))

            # Time-based anomaly detection (sudden increases)
            if len(w_df) >= 5:  # Need enough data points
                w_df = w_df.sort_values("parsed_date")
                w_df["prev_amount"] = w_df["Withdrawal (INR)"].shift(1)
                w_df["amount_change_ratio"] = w_df["Withdrawal (INR)"] / w_df["prev_amount"]

                # Find sudden spikes (more than 5x previous transaction)
                spikes = w_df[(w_df["amount_change_ratio"] > 5) & (~w_df["prev_amount"].isna())][
                    ["parsed_date", "Narration", "Withdrawal (INR)", "prev_amount", "amount_change_ratio", "category"]
                ].rename(columns={"Withdrawal (INR)": "amount"})

                if not spikes.empty:
                    for idx, row in spikes.iterrows():
                        spikes.at[idx, "severity"] = "extreme" if row["amount_change_ratio"] > 10 else "high"

                    spikes["type"] = "withdrawal"
                    spikes["detection_method"] = "sudden_increase"
                    spike_records = spikes[["parsed_date", "Narration", "amount", "type", "category", "severity", "detection_method"]].to_dict("records")
                    anomalies.extend(spike_records)

        # Deposits - with similar enhancements
        d_df = cat_df[cat_df["Deposit (INR)"] > 0]
        if not d_df.empty:
            mean_d = d_df["Deposit (INR)"].mean()
            std_d = d_df["Deposit (INR)"].std()

            # Standard detection
            threshold_d = mean_d + 3 * std_d if std_d > 0 else mean_d * 2
            d_anomalies = d_df[d_df["Deposit (INR)"] > threshold_d][
                ["parsed_date", "Narration", "Deposit (INR)", "category"]
            ].rename(columns={"Deposit (INR)": "amount"})

            # Add anomaly score and type
            if not d_anomalies.empty:
                for idx, row in d_anomalies.iterrows():
                    z_score = (row["amount"] - mean_d) / (std_d if std_d > 0 else mean_d)
                    d_anomalies.at[idx, "z_score"] = z_score
                    d_anomalies.at[idx, "severity"] = "extreme" if z_score > 5 else "high" if z_score > 3 else "moderate"

                d_anomalies["type"] = "deposit"
                d_anomalies["detection_method"] = "statistical"
                anomalies.extend(d_anomalies.to_dict("records"))

            # Time-based deposit anomalies
            if len(d_df) >= 5:
                d_df = d_df.sort_values("parsed_date")
                d_df["prev_amount"] = d_df["Deposit (INR)"].shift(1)
                d_df["amount_change_ratio"] = d_df["Deposit (INR)"] / d_df["prev_amount"]

                # Find sudden spikes
                spikes = d_df[(d_df["amount_change_ratio"] > 5) & (~d_df["prev_amount"].isna())][
                    ["parsed_date", "Narration", "Deposit (INR)", "prev_amount", "amount_change_ratio", "category"]
                ].rename(columns={"Deposit (INR)": "amount"})

                if not spikes.empty:
                    for idx, row in spikes.iterrows():
                        spikes.at[idx, "severity"] = "extreme" if row["amount_change_ratio"] > 10 else "high"

                    spikes["type"] = "deposit"
                    spikes["detection_method"] = "sudden_increase"
                    spike_records = spikes[["parsed_date", "Narration", "amount", "type", "category", "severity", "detection_method"]].to_dict("records")
                    anomalies.extend(spike_records)

    # Frequency anomalies - unusual transaction frequency
    if not df.empty:
        df_sorted = df.sort_values("parsed_date")
        df_sorted["next_trans_days"] = (df_sorted["parsed_date"].shift(-1) - df_sorted["parsed_date"]).dt.days

        # Very long gaps (possible missed transactions)
        mean_gap = df_sorted["next_trans_days"].mean()
        std_gap = df_sorted["next_trans_days"].std()
        large_gaps = df_sorted[df_sorted["next_trans_days"] > mean_gap + 3*std_gap]

        if not large_gaps.empty:
            for idx, row in large_gaps.iterrows():
                if row["next_trans_days"] is not None and not np.isnan(row["next_trans_days"]):
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
    """Analyze cash flow with simplified output."""
    cash_flow_analysis = []

    if df.empty or "parsed_date" not in df.columns:
        return cash_flow_analysis

    df = df.copy()
    df["parsed_date"] = pd.to_datetime(df["parsed_date"], errors="coerce")
    df["month"] = df["parsed_date"].dt.to_period("M")

    # Monthly cash flow
    monthly_cf = df.groupby("month").agg({
        "Deposit (INR)": "sum",
        "Withdrawal (INR)": "sum",
    }).reset_index()

    monthly_cf["net_cash_flow"] = monthly_cf["Deposit (INR)"] - monthly_cf["Withdrawal (INR)"]
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
