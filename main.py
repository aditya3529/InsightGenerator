import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re
import together
import time

# Set Together API key
together.api_key = st.secrets["TOGETHER_API_KEY"]

# Retry wrapper for Together API call
def safe_generate(prompt, retries=3, wait=3):
    for attempt in range(retries):
        try:
            return together.Complete.create(
                prompt=prompt,
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                max_tokens=512,
                temperature=0.7,
                stop=["```", "</json>"]
            )
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                raise e

# Generate AI churn insight
def generate_churn_insight(df: pd.DataFrame):
    sampled_df = df[['Age', 'CreditScore', 'Geography', 'Exited']].sample(
        n=min(30, len(df)), random_state=1
    )
    data_sample = sampled_df.to_csv(index=False)

    prompt = f"""
Analyze this CSV dataset of customer churn. Return ONLY a JSON with:
- title: a short insight title
- summary: a short analysis summary
- actionItems: a list of actionable suggestions

CSV Data:
{data_sample}
"""

    response = safe_generate(prompt)
    st.code(response, language="json")  # Debug display

    output_text = response.get("output", {}).get("choices", [{}])[0].get("text", "").strip()
    if not output_text:
        raise ValueError("Empty response from model")

    try:
        json_text = re.search(r"\{.*\}", output_text, re.DOTALL).group()
        return json.loads(json_text)
    except Exception:
        raise ValueError(f"Could not parse JSON. Model response:\n\n{output_text[:300]}")

# Generate dummy data
def generate_dummy_data():
    return pd.DataFrame({
        'CustomerId': range(1001, 1021),
        'Surname': [f'User{i}' for i in range(20)],
        'CreditScore': [650 + i % 50 for i in range(20)],
        'Geography': ['France', 'Spain', 'Germany', 'France'] * 5,
        'Gender': ['Male', 'Female'] * 10,
        'Age': [30 + i % 10 for i in range(20)],
        'Tenure': [i % 5 for i in range(20)],
        'Balance': [10000 + i * 1000 for i in range(20)],
        'NumOfProducts': [1, 2] * 10,
        'HasCrCard': [1, 0] * 10,
        'IsActiveMember': [1, 0] * 10,
        'EstimatedSalary': [50000 + i * 1500 for i in range(20)],
        'Exited': [0, 1] * 10
    })

# App UI
st.title("🧭 InsightPilot")
st.caption("Navigate churn with product-led transformation")
st.markdown("Upload a customer CSV file or generate sample data to explore churn patterns and insights.")
st.markdown("Click on 'Generate Dummy data' to explore the app.")

# Action buttons
col_btn1, col_btn2 = st.columns([1, 1])
with col_btn1:
    if st.button("🧪 Generate Dummy Data", use_container_width=True):
        st.session_state["dummy_data"] = generate_dummy_data()
        st.rerun()
with col_btn2:
    if st.button("🔄 Reset App", use_container_width=True):
        st.session_state.clear()
        st.rerun()

uploaded_file = st.file_uploader("📤 Upload CSV", type=["csv", "txt"])

if "dummy_data" in st.session_state:
    df = st.session_state["dummy_data"]
elif uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = None

if df is not None:
    try:
        st.subheader("📄 Data Preview")
        st.dataframe(df)

        if 'Exited' not in df.columns:
            raise ValueError("Missing 'Exited' column for churn analysis.")

        # KPIs
        st.subheader("📌 Key KPIs")
        total_customers = len(df)
        churn_rate = df['Exited'].mean()
        avg_credit = df['CreditScore'].mean()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", total_customers)
        col2.metric("Churn Rate", f"{churn_rate * 100:.2f}%")
        col3.metric("Avg. Credit Score", f"{avg_credit:.0f}")

        # Visuals
        st.subheader("📊 Churn Breakdown")
        fig1, ax1 = plt.subplots()
        df['Exited'].value_counts().plot(kind='bar', ax=ax1, color=['green', 'red'])
        ax1.set_xticklabels(['Retained', 'Churned'], rotation=0)
        ax1.set_ylabel("Customers")
        ax1.set_title("Churn Distribution")
        st.pyplot(fig1)

        st.subheader("🌍 Churn by Geography")
        fig2, ax2 = plt.subplots()
        sns.barplot(data=df, x='Geography', y='Exited', ci=None, ax=ax2)
        ax2.set_title("Churn Rate by Geography")
        st.pyplot(fig2)

        st.subheader("🎯 Age vs. Churn")
        fig3, ax3 = plt.subplots()
        sns.histplot(data=df, x="Age", hue="Exited", bins=20, multiple="stack", ax=ax3)
        ax3.set_title("Age Distribution by Churn Status")
        st.pyplot(fig3)

        st.subheader("💳 Credit Score vs Churn")
        fig4, ax4 = plt.subplots()
        sns.boxplot(data=df, x="Exited", y="CreditScore", ax=ax4)
        ax4.set_xticklabels(['Retained', 'Churned'])
        ax4.set_title("Credit Score Distribution by Churn")
        st.pyplot(fig4)

        # LLM Insight
        st.subheader("🧠 AI-Generated Churn Insight")
        with st.spinner("Analyzing data..."):
            try:
                insight = generate_churn_insight(df)
                st.success(insight["title"])
                st.markdown(f"**Summary:** {insight['summary']}")
                st.markdown("**Action Items:**")
                for item in insight["actionItems"]:
                    st.markdown(f"- {item}")
            except Exception as e:
                st.error(f"Failed to generate insight: {e}")

    except Exception as e:
        st.error(f"❌ Error processing data: {e}")

