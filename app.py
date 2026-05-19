import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.figure_factory as ff
import warnings

warnings.filterwarnings('ignore')

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="DataMind AI",
    page_icon="🧠",
    layout="wide"
)

# ---------------- CUSTOM HEADER ----------------
st.markdown("""
# 🧠 DataMind AI

### AI-Powered Automated Data Science Platform

Upload datasets, detect ML tasks automatically, visualize insights, and generate predictions instantly.
""")

st.markdown("---")

# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.title("🚀 DataMind AI")

    st.markdown("""
### Smart AutoML Dashboard

Developed By:
**Pakhi Srivastava**

### Tech Stack
- Python
- Streamlit
- Scikit-learn
- Plotly
""")

    expertise = st.selectbox(
        "🎓 Expertise Level",
        ["Beginner", "Intermediate", "Expert"]
    )

    st.markdown("---")

    st.markdown("""
## ⚡ Workflow

✅ Upload Dataset  
✅ Detect Task  
✅ Train ML Model  
✅ Generate Insights  
✅ Visualize Results  
✅ Download Report  
""")

# ---------------- AI EXPLANATION ----------------
def get_ai_response(task, expertise):

    if expertise == "Beginner":

        return f"""
✅ AI Analysis Complete

Detected Task: {task}

The AI system automatically analyzed your dataset and selected the correct machine learning workflow.

The model successfully identified patterns and generated predictions from your data.
"""

    elif expertise == "Intermediate":

        return f"""
✅ AI Analysis Complete

Insights:
- Dataset processed successfully
- ML pipeline executed automatically
- Features analyzed dynamically
- Results generated successfully
"""

    else:

        return f"""
✅ AI Analysis Complete

Advanced Insights:
- Automated preprocessing completed
- Dynamic ML task inference executed
- Model orchestration successful
- Feature engineering pipeline optimized
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

        return results, df_encoded, None

    # ---------------- FEATURES ----------------
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

    return results, df_encoded, model

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📂 Upload CSV Dataset",
    type=["csv"]
)

# ---------------- MAIN PROCESS ----------------
if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.success(
        f"✅ File Uploaded Successfully: {uploaded_file.name}"
    )

    # ---------------- DASHBOARD METRICS ----------------
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))
    col4.metric("Features", len(df.columns))

    st.markdown("---")

    # ---------------- DATA QUALITY ----------------
    missing = df.isnull().sum().sum()

    if missing == 0:
        st.success("✅ Clean Dataset")
    else:
        st.warning(f"⚠️ Missing Values Found: {missing}")

    # ---------------- TABS ----------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "📂 Dataset",
        "🧠 Analysis",
        "📈 ML Results",
        "📊 Visualizations"
    ])

    # ---------------- TAB 1 ----------------
    with tab1:

        st.subheader("👀 Dataset Preview")
        st.dataframe(df.head())

        st.subheader("📊 Dataset Statistics")
        st.dataframe(df.describe())

    # ---------------- TASK DETECTION ----------------
    detected_task = detect_task_type(df)

    target_col = None

    # ---------------- TAB 2 ----------------
    with tab2:

        st.subheader("🧠 AI Dataset Analysis")

        st.success(
            f"✅ Detected Task: {detected_task.upper()}"
        )

        col1, col2 = st.columns(2)

        with col1:

            task_type = st.selectbox(
                "⚙️ Select Task Type",
                [
                    "classification",
                    "regression",
                    "clustering"
                ],
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

        # ---------------- MODEL RECOMMENDATION ----------------
        if task_type == "classification":

            st.info(
                "🤖 Recommended Model: Random Forest Classifier"
            )

        elif task_type == "regression":

            st.info(
                "🤖 Recommended Model: Random Forest Regressor"
            )

        elif task_type == "clustering":

            st.info(
                "🤖 Recommended Model: KMeans Clustering"
            )

    # ---------------- RUN MODEL ----------------
    if st.button(
        "🚀 Run AI Model",
        use_container_width=True
    ):

        results, df_processed, model = run_model(
            df,
            task_type,
            target_col
        )

        # ---------------- TAB 3 ----------------
        with tab3:

            st.subheader("📈 Machine Learning Results")

            st.success(
                "✅ Analysis Completed Successfully"
            )

            # ---------------- CLASSIFICATION ----------------
            if results['type'] == 'classification':

                col1, col2 = st.columns(2)

                col1.metric(
                    "Accuracy",
                    f"{results['accuracy']}%"
                )

                col2.metric(
                    "Model",
                    results['model_used']
                )

            # ---------------- REGRESSION ----------------
            elif results['type'] == 'regression':

                col1, col2 = st.columns(2)

                col1.metric(
                    "R2 Score",
                    f"{results['r2_score']}%"
                )

                col2.metric(
                    "Model",
                    results['model_used']
                )

            # ---------------- CLUSTERING ----------------
            elif results['type'] == 'clustering':

                col1, col2 = st.columns(2)

                col1.metric(
                    "Clusters",
                    results['clusters']
                )

                col2.metric(
                    "Inertia",
                    results['inertia']
                )

            # ---------------- WHY MODEL ----------------
            st.subheader("🧠 Why This Model?")

            if task_type == "classification":

                st.write(
                    "Random Forest performs well for structured classification datasets."
                )

            elif task_type == "regression":

                st.write(
                    "Random Forest Regressor handles nonlinear relationships effectively."
                )

            elif task_type == "clustering":

                st.write(
                    "KMeans automatically groups similar datapoints."
                )

            # ---------------- AI EXPLANATION ----------------
            st.subheader("💬 AI Explanation")

            explanation = get_ai_response(
                task_type,
                expertise
            )

            st.info(explanation)

        # ---------------- TAB 4 ----------------
        with tab4:

            st.subheader("📊 Interactive Visualizations")

            numeric_cols = df.select_dtypes(
                include=['number']
            ).columns.tolist()

            # ---------------- MAIN CHART ----------------
            if len(numeric_cols) >= 2:

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

                else:

                    fig = px.scatter(
                        df,
                        x=numeric_cols[0],
                        y=numeric_cols[1],
                        title="Dataset Visualization"
                    )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            # ---------------- CORRELATION HEATMAP ----------------
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

            # ---------------- FEATURE IMPORTANCE ----------------
            if task_type != "clustering":

                importance = model.feature_importances_

                importance_df = pd.DataFrame({
                    "Feature": df.drop(columns=[target_col]).columns,
                    "Importance": importance
                })

                fig3 = px.bar(
                    importance_df,
                    x="Feature",
                    y="Importance",
                    title="Feature Importance"
                )

                st.plotly_chart(
                    fig3,
                    use_container_width=True
                )

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
        st.markdown("## 📂 Upload")
        st.write(
            "Upload any CSV dataset for AI-powered analysis."
        )

    with col2:
        st.markdown("## 🧠 Analyze")
        st.write(
            "AI automatically detects the ML workflow."
        )

    with col3:
        st.markdown("## 📈 Predict")
        st.write(
            "Generate insights, charts, and predictions."
        )

# ---------------- FOOTER ----------------
st.markdown("---")

st.markdown(
    "🚀 Developed by Pakhi Srivastava | DataMind AI"
)