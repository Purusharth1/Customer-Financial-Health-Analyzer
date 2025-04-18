
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# FastAPI backend URL
API_URL = "http://localhost:8000"

# Streamlit page config
st.set_page_config(page_title="Financial Health Analyzer", layout="wide")

# Title
st.title("Customer Financial Health Analyzer")

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Upload Statements", "Transactions", "Analysis", "Visualizations", "Stories", "Ask a Question"])

# Upload Statements
if page == "Upload Statements":
    st.header("Upload Bank Statements")
    st.write("Upload up to 10 PDF bank statements to analyze your financial health.")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        if len(uploaded_files) > 10:
            st.error("Maximum 10 PDFs allowed.")
        else:
            files = [("files", (file.name, file, "application/pdf")) for file in uploaded_files]
            try:
                with st.spinner("Processing PDFs..."):
                    response = requests.post(f"{API_URL}/upload-pdfs/", files=files)
                    if response.status_code == 202:
                        st.success(f"Successfully processed {response.json()['files_uploaded']} file(s)!")
                    else:
                        st.error(f"Error: {response.json()['detail']}")
            except Exception as e:
                st.error(f"Upload failed: {e!s}")

# Transactions
elif page == "Transactions":
    st.header("Recent Transactions")
    try:
        response = requests.get(f"{API_URL}/transactions/")
        if response.status_code == 200:
            transactions = response.json()
            if transactions:
                df = pd.DataFrame(transactions)
                df["Amount (INR)"] = df.apply(
                    lambda row: -row["Withdrawal (INR)"] if row["Withdrawal (INR)"] else row["Deposit (INR)"], axis=1,
                )
                st.dataframe(
                    df[["parsed_date", "Narration", "Amount (INR)", "category"]],
                    column_config={
                        "parsed_date": "Date",
                        "Narration": "Description",
                        "Amount (INR)": st.column_config.NumberColumn("Amount (INR)", format="₹%.2f"),
                        "category": "Category",
                    },
                    use_container_width=True,
                )
            else:
                st.info("No transactions found. Please upload bank statements.")
        else:
            st.error(f"Error: {response.json()['detail']}")
    except Exception as e:
        st.error(f"Failed to load transactions: {e!s}")

# Analysis
elif page == "Analysis":
    st.header("Financial Analysis")
    try:
        response = requests.get(f"{API_URL}/analysis/")
        if response.status_code == 200:
            analysis = response.json()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Balance", f"₹{analysis['account_overview']['total_balance']:,.2f}",
                          f"{analysis['account_overview']['balance_percentage']}%")
            with col2:
                st.metric("Monthly Income", f"₹{analysis['account_overview']['monthly_income']:,.2f}",
                          f"{analysis['account_overview']['income_percentage']}%")
            with col3:
                st.metric("Monthly Expense", f"₹{analysis['account_overview']['monthly_expense']:,.2f}",
                          f"{analysis['account_overview']['expense_percentage']}%")

            st.subheader("Spending Patterns")
            for pattern in analysis.get("patterns", []):
                st.write(f"- {pattern}")

            st.subheader("Recurring Payments")
            for recurring in analysis.get("recurring", []):
                st.write(f"- {recurring['narration']}: ₹{recurring['amount']} ({recurring['frequency']})")

            st.subheader("Anomalies")
            for anomaly in analysis.get("anomalies", []):
                st.write(f"- {anomaly['Narration']}: ₹{anomaly['amount']} ({anomaly['severity']})")
        else:
            st.error(f"Error: {response.json()['detail']}")
    except Exception as e:
        st.error(f"Failed to load analysis: {e!s}")

# Visualizations
elif page == "Visualizations":
    st.header("Spending Visualizations")
    try:
        response = requests.get(f"{API_URL}/visualizations/")
        if response.status_code == 200:
            viz_data = response.json()
            col1, col2 = st.columns(2)

            # Spending Trends (Line Chart)
            with col1:
                trends = viz_data.get("spending_trends", {"labels": [], "expenses": [], "budget": []})
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=trends["labels"], y=trends["expenses"], name="Expenses", line=dict(color="#3b82f6")))
                fig.add_trace(go.Scatter(x=trends["labels"], y=trends["budget"], name="Budget", line=dict(color="#f59e0b", dash="dash")))
                fig.update_layout(title="Spending Trends", xaxis_title="Month", yaxis_title="Amount (INR)", height=400)
                st.plotly_chart(fig, use_container_width=True)

            # Expense Breakdown (Pie Chart)
            with col2:
                breakdown = viz_data.get("expense_breakdown", {"categories": [], "percentages": []})
                fig = px.pie(
                    names=breakdown["categories"],
                    values=breakdown["percentages"],
                    title="Expense Breakdown",
                    height=400,
                )
                fig.update_traces(textinfo="percent+label")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Error: {response.json()['detail']}")
    except Exception as e:
        st.error(f"Failed to load visualizations: {e!s}")

# Stories
elif page == "Stories":
    st.header("Financial Stories")
    try:
        response = requests.get(f"{API_URL}/stories/")
        if response.status_code == 200:
            stories = response.json()
            if stories:
                for story in stories:
                    st.write(f"- {story}")
            else:
                st.info("No stories available. Please upload bank statements.")
        else:
            st.error(f"Error: {response.json()['detail']}")
    except Exception as e:
        st.error(f"Failed to load stories: {e!s}")

# Ask a Question
elif page == "Ask a Question":
    st.header("Ask a Financial Question")
    query = st.text_input("Enter your question (e.g., 'What are my top expenses?')")
    if st.button("Submit"):
        if query:
            try:
                with st.spinner("Processing query..."):
                    response = requests.post(f"{API_URL}/query/", json={"query": query})
                    if response.status_code == 200:
                        result = response.json()
                        st.write("**Answer:**")
                        st.write(result["response"])
                        # In the "Ask a Question" section where you handle query response visualization:
                        if result.get("visualization"):
                            st.write("**Visualization:**")
                            viz_data = result["visualization"]

                            # Create DataFrame from the visualization data
                            if "data" in viz_data and "columns" in viz_data:
                                # Handle the structure in your JSON
                                df = pd.DataFrame(viz_data["data"], columns=viz_data["columns"])

                                # Create the chart based on type
                                if viz_data["type"] == "bar":
                                    fig = px.bar(
                                        df,
                                        x=viz_data["columns"][0],  # First column (Date)
                                        y=viz_data["columns"][1],  # Second column (Amount)
                                        title=viz_data.get("title", "Query Result"),
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                # Fallback for other visualization formats
                                try:
                                    df = pd.DataFrame({
                                        "labels": viz_data.get("labels", []),
                                        "values": viz_data.get("values", []),
                                    })
                                    fig = px.bar(
                                        df,
                                        x="labels",
                                        y="values",
                                        title=viz_data.get("title", "Query Result"),
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Could not render visualization: {e!s}")
            except Exception as e:
                st.error(f"Query failed: {e!s}")
        else:
            st.warning("Please enter a question.")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Built with Streamlit and FastAPI")
