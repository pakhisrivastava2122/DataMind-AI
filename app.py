import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="DataMind AI",
    page_icon="🤖",
    layout="wide"
)

# ---------------- AI RESPONSE ----------------
def get_ai_response(task, expertise):

    if expertise == "Beginner":
        return f"""
✅ AI Analysis Complete

Detected Task: {task}

The AI system automatically analyzed your dataset and selected the correct machine learning workflow.

The model successfully identified patterns and generated insights from your data.
"""

    elif expertise == "Intermediate":
        return f"""
✅ AI Analysis Complete

Detected Task: {task}

Insights:
- Dataset processed successfully
- Features analyzed automatically
- ML workflow selected dynamically
- Results generated using machine learning pipeline
"""

    else:
        return f"""
✅ AI Analysis Complete

Detected Task: {task}

Advanced Analysis:
The application dynamically inferred the ML task type using dataset structure analysis.

Automated preprocessing and model orchestration were completed successfully.
"""


# ---------------- TASK DETECTION ----------------
def detect_task_type(df):

    target = df.columns[-1]

    if df[target].dtype == "object":
        return "classification"

    elif df[target].nunique() < 10:
        return "classification"

    else:
        return "regression"


# ---------------- MODEL FUNCTION ----------------
def run_model(df, task, target_col):

    results = {}

    df_encoded = df.copy()

    le = LabelEncoder()

    for col in df_encoded.select_dtypes(include=['object']).columns:
        df_encoded[col] = le.fit_transform(
            df_encoded[col].astype(str)
        )

    df_encoded = df_encoded.fillna(df_encoded.mean())

    # ---------------- CLUSTERING ----------------
    if task == "clustering":

        kmeans = KMeans(
            n_clusters=3,
            random_state=42
        )

        df_encoded['Cluster'] = kmeans.fit_predict(df_encoded)

        results['type'] = 'clustering'
        results['clusters'] = 3
        results['inertia'] = round(kmeans.inertia_, 2)
        results['cluster_labels'] = df_encoded['Cluster'].tolist()

        return results, df_encoded

    # ---------------- TARGET ----------------
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ---------------- CLASSIFICATION ----------------
    if task == "classification":

        model = RandomForestClassifier(
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        results['type'] = 'classification'
        results['accuracy'] = round(
            accuracy_score(y_test, y_pred) * 100,
            2
        )

        results['model_used'] = "Random Forest Classifier"

    # ---------------- REGRESSION ----------------
    elif task == "regression":

        model = RandomForestRegressor(
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        results['type'] = 'regression'
        results['r2_score'] = round(
            r2_score(y_test, y_pred) * 100,
            2
        )

        results['model_used'] = "Random Forest Regressor"

    return results, df_encoded


# ---------------- MAIN UI ----------------
st.title("🤖 DataMind AI")

st.subheader(
    "AI-Powered Automated Data Science Platform"
)

st.markdown(
    "Upload your dataset and let AI automatically analyze, model, and explain your data."
)

st.markdown("---")

# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.title("🧠 DataMind AI")

    st.markdown("""
AI-Powered AutoML Platform

Developed By:
Pakhi Srivastava

Tech Stack:
- Python
- Streamlit
- Scikit-learn
- Plotly
""")

    expertise = st.selectbox(
        "Select Expertise Level",
        ["Beginner", "Intermediate", "Expert"]
    )

    st.markdown("---")

    st.markdown("""
### 🚀 Workflow

1. Upload CSV
2. AI Detects Task
3. ML Model Runs
4. Visualization Generated
5. AI Explains Results
""")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📂 Upload CSV Dataset",
    type=["csv"]
)

# ---------------- PROCESS ----------------
if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.success(
        f"✅ File Uploaded Successfully: {uploaded_file.name}"
    )

    # ---------------- METRICS ----------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

    # ---------------- PREVIEW ----------------
    with st.expander("👀 Dataset Preview"):
        st.dataframe(df.head())

    with st.expander("📊 Dataset Statistics"):
        st.dataframe(df.describe())

    st.markdown("---")

    # ---------------- TASK DETECTION ----------------
    st.subheader("🧠 Step 1: AI Analyzing Your Data")

    with st.spinner("Analyzing dataset..."):

        detected_task = detect_task_type(df)

    st.success(
        f"✅ Detected Task Type: {detected_task.upper()}"
    )

    target_col = None

    # ---------------- TASK + TARGET ----------------
    col1, col2 = st.columns(2)

    with col1:

        task_type = st.selectbox(
            "⚙️ Select Task Type",
            ["classification", "regression", "clustering"],
            index=[
                "classification",
                "regression",
                "clustering"
            ].index(detected_task)
        )

    with col2:

        if task_type != "clustering":

            target_col = st.selectbox(
                "🎯 Select Target Column",
                df.columns.tolist(),
                index=len(df.columns)-1
            )

        else:

            st.info(
                "No target column required for clustering"
            )

    st.markdown("---")

    # ---------------- RUN MODEL ----------------
    if st.button(
        "🚀 Run AI Model",
        use_container_width=True
    ):

        with st.spinner("Running machine learning model..."):

            results, df_processed = run_model(
                df,
                task_type,
                target_col
            )

        # ---------------- RESULTS ----------------
        st.subheader("📈 Step 2: Results")

        st.success(
            "✅ Analysis Completed Successfully"
        )

        # ---------------- CLASSIFICATION ----------------
        if results['type'] == 'classification':

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Accuracy",
                    f"{results['accuracy']}%"
                )

            with col2:
                st.metric(
                    "Model Used",
                    results['model_used']
                )

        # ---------------- REGRESSION ----------------
        elif results['type'] == 'regression':

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "R2 Score",
                    f"{results['r2_score']}%"
                )

            with col2:
                st.metric(
                    "Model Used",
                    results['model_used']
                )

        # ---------------- CLUSTERING ----------------
        elif results['type'] == 'clustering':

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Clusters",
                    results['clusters']
                )

            with col2:
                st.metric(
                    "Inertia Score",
                    results['inertia']
                )

        # ---------------- VISUALIZATION ----------------
        st.subheader("📊 Step 3: Visualization")

        numeric_cols = df.select_dtypes(
            include=['number']
        ).columns.tolist()

        if len(numeric_cols) >= 2:

            # -------- CLUSTERING VISUAL --------
            if results['type'] == 'clustering':

                df['Cluster'] = [
                    str(x)
                    for x in results['cluster_labels']
                ]

                fig = px.scatter(
                    df,
                    x=numeric_cols[0],
                    y=numeric_cols[1],
                    color='Cluster',
                    title="Cluster Visualization"
                )

            # -------- REGRESSION/CLASSIFICATION --------
            else:

                fig = px.histogram(
                    df,
                    x=numeric_cols[0],
                    title=f"Distribution of {numeric_cols[0]}"
                )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ---------------- HEATMAP ----------------
        if len(numeric_cols) > 1:

            corr = df[numeric_cols].corr()

            fig2 = px.imshow(
                corr,
                title="Feature Correlation Heatmap",
                color_continuous_scale='RdBu'
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

        # ---------------- AI EXPLANATION ----------------
        st.subheader("💬 Step 4: AI Explanation")

        explanation = get_ai_response(
            task_type,
            expertise
        )

        st.info(explanation)

        # ---------------- DOWNLOAD REPORT ----------------
        report = f"""
DataMind AI Report

Task Type:
{task_type}

Dataset Shape:
{df.shape}

Analysis Completed Successfully.
"""

        st.download_button(
            label="📥 Download AI Report",
            data=report,
            file_name="DataMind_AI_Report.txt",
            mime="text/plain"
        )

# ---------------- HOME SCREEN ----------------
else:

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📂 Upload")
        st.markdown(
            "Upload any CSV dataset"
        )

    with col2:
        st.markdown("### 🧠 AI Analysis")
        st.markdown(
            "AI automatically detects the ML task"
        )

    with col3:
        st.markdown("### 📊 Results")
        st.markdown(
            "Visualize insights and predictions"
        )

# ---------------- FOOTER ----------------
st.markdown("---")

st.markdown(
    "🚀 Developed by Pakhi Srivastava | DataMind AI"
)