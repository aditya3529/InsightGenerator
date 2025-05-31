import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import together
from pydantic import BaseModel, Field

# ✅ Initialize Together client without manually passing the API key
client = together.Together()


# Define structured response schema
class ChurnInsight(BaseModel):
    title: str = Field(description="Insight title")
    summary: str = Field(description="Brief summary of the insight")
    actionItems: list[str] = Field(description="List of recommended actions")


# Generate AI churn insight
def generate_churn_insight(df: pd.DataFrame):
    data_sample = df.sample(n=min(50, len(df)), random_state=1).to_csv(index=False)
    prompt = f"""
    You are a data analyst. Analyze the following customer churn dataset (CSV format).
    Provide a JSON response with:
    {{
      title: string,
      summary: string,
      actionItems: [string]
    }}

    Data:
    {data_sample}
    """

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[
            {"role": "system", "content": "You are a product analyst. Reply ONLY in JSON."},
            {"role": "user", "content": prompt}
        ],
        response_format={
            "type": "json_object",
            "schema": ChurnInsight.model_json_schema()
        },
    )

    output = json.loads(response.choices[0].message.content)
    return output


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


# App Title and Branding
st.title("🧭 InsightPilot")
st.caption("Navigate churn with product-led transformation")
st.markdown("Upload a customer CSV file or generate sample data to explore churn patterns and insights.")
st.markdown("Click on 'Generate Dummy data' to explore the app.")

# Action Buttons
col_btn1, col_btn2 = st.columns([1, 1])
with col_btn1:
    if st.button("🧪 Generate Dummy Data", use_container_width=True):
        st.session_state["dummy_data"] = generate_dummy_data()
        st.rerun()
with col_btn2:
    if st.button("🔄 Reset App", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# File Upload
uploaded_file = st.file_uploader("📤 Upload CSV", type=["csv", "txt"])

# Load Data
if "dummy_data" in st.session_state:
    df = st.session_state["dummy_data"]
elif uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = None

# Main Logic
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
        avg_age = df['Age'].mean()
        avg_credit = df['CreditScore'].mean()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", total_customers)
        col2.metric("Churn Rate", f"{churn_rate * 100:.2f}%")
        col3.metric("Avg. Credit Score", f"{avg_credit:.0f}")

        # Visualizations
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

        # AI Insight
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


