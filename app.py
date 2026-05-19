import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# ---- CONFIG ----
st.set_page_config(
    page_title="DataMind AI",
    page_icon="🤖",
    layout="wide"
)

# ---- GEMINI SETUP ----
# genai.configure(api_key="AIzaSyBepf_uGWiU1P6GnPGROytTz5JnGn1njxg")
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model_ai = genai.GenerativeModel("gemini-1.5-pro")

# ---- HELPER FUNCTIONS ----
def get_ai_response(prompt):
    try:
        response = model_ai.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"


def explain_results(task, metrics, df_info, expertise):
    prompt = f"""
    A student ran a {task} model on a dataset with {df_info['rows']} rows and {df_info['cols']} columns.
    Results: {metrics}
    
    Explain these results to a {expertise} level user in simple, friendly language.
    - What do these numbers mean?
    - Is this a good result?
    - What could be improved?
    Keep it under 150 words.
    """
    return get_ai_response(prompt)

def detect_task_type(df):
    cols = df.columns.tolist()

    prompt = f"""
    Analyze these dataset columns:
    {cols}

    Decide:
    - classification
    - regression
    - clustering

    Return ONLY:

    TASK: <task>
    TARGET: <target column>
    """

    return get_ai_response(prompt)

def run_model(df, task, target_col):
    results = {}
    
    # Encode categorical columns
    df_encoded = df.copy()
    le = LabelEncoder()
    for col in df_encoded.select_dtypes(include=['object']).columns:
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    
    df_encoded = df_encoded.fillna(df_encoded.mean())
    
    if task == "clustering":
        kmeans = KMeans(n_clusters=3, random_state=42)
        df_encoded['Cluster'] = kmeans.fit_predict(df_encoded)
        results['type'] = 'clustering'
        results['clusters'] = 3
        results['inertia'] = round(kmeans.inertia_, 2)
        results['cluster_labels'] = df_encoded['Cluster'].tolist()
        return results, df_encoded
    
    if target_col not in df_encoded.columns:
        return None, df_encoded
    
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    if task == "classification":
        ml_model = RandomForestClassifier(random_state=42)
        ml_model.fit(X_train, y_train)
        y_pred = ml_model.predict(X_test)
        results['type'] = 'classification'
        results['accuracy'] = round(accuracy_score(y_test, y_pred) * 100, 2)
        results['model_used'] = 'Random Forest Classifier'
        
    elif task == "regression":
        ml_model = RandomForestRegressor(random_state=42)
        ml_model.fit(X_train, y_train)
        y_pred = ml_model.predict(X_test)
        results['type'] = 'regression'
        results['r2_score'] = round(r2_score(y_test, y_pred) * 100, 2)
        results['model_used'] = 'Random Forest Regressor'
    
    return results, df_encoded

# ---- MAIN APP ----
st.title("🤖 DataMind AI")
st.markdown("### Upload your data — AI will analyze, model, and explain it for you!")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    expertise = st.selectbox(
        "Your expertise level:",
        ["Beginner", "Intermediate", "Expert"]
    )
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. Upload CSV file")
    st.markdown("2. AI detects task type")
    st.markdown("3. Model runs automatically")
    st.markdown("4. AI explains results")

# File Upload
uploaded_file = st.file_uploader(
    "📂 Upload your CSV file here",
    type=['csv'],
    help="Upload any CSV dataset"
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.success(f"✅ File uploaded: {uploaded_file.name}")
    
    # Show data
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    
    with st.expander("👀 Preview your data"):
        st.dataframe(df.head(10))
    
    with st.expander("📊 Data Statistics"):
        st.dataframe(df.describe())
    
    st.markdown("---")
    
    # AI Task Detection
    st.subheader("🧠 Step 1: AI Analyzing Your Data...")
    
    with st.spinner("AI is reading your dataset..."):
        ai_analysis = detect_task_type(df)
    
    st.markdown("**AI says:**")
    st.info(ai_analysis)
    
    # Parse AI response
    task_type = "clustering"
    target_col = None
    
    for line in ai_analysis.split('\n'):
        if line.startswith("TASK:"):
            task_type = line.replace("TASK:", "").strip().lower()
        if line.startswith("TARGET:"):
            target_col = line.replace("TARGET:", "").strip()
    
    st.markdown("---")
    
    # Manual override
    st.subheader("⚙️ Step 2: Confirm or Change Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        task_type = st.selectbox(
            "Task Type:",
            ["classification", "regression", "clustering"],
            index=["classification", "regression", "clustering"].index(
                task_type if task_type in ["classification", "regression", "clustering"] else "clustering"
            )
        )
    with col2:
        if task_type != "clustering":
            target_col = st.selectbox(
                "Target Column:",
                df.columns.tolist(),
                index=df.columns.tolist().index(target_col) 
                if target_col in df.columns.tolist() else 0
            )
    
    # Run Model Button
    st.markdown("---")
    if st.button("🚀 Run AI Model", type="primary", use_container_width=True):
        
        with st.spinner("Running ML model..."):
            results, df_processed = run_model(df, task_type, target_col)
        
        st.subheader("📈 Step 3: Results")
        
        if results:
            # Show metrics
            if results['type'] == 'classification':
                st.metric("✅ Model Accuracy", f"{results['accuracy']}%")
                st.metric("🤖 Model Used", results['model_used'])
                
            elif results['type'] == 'regression':
                st.metric("✅ R2 Score", f"{results['r2_score']}%")
                st.metric("🤖 Model Used", results['model_used'])
                
            elif results['type'] == 'clustering':
                st.metric("✅ Clusters Found", results['clusters'])
                st.metric("📊 Inertia Score", results['inertia'])
            
            # Visualization
            st.subheader("📊 Step 4: Visualization")
            
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) >= 2:
                if results['type'] == 'clustering':
                    df['Cluster'] = [str(x) for x in results['cluster_labels']]
                    fig = px.scatter(
                        df, x=numeric_cols[0], y=numeric_cols[1],
                        color='Cluster',
                        title="Cluster Visualization"
                    )
                else:
                    fig = px.histogram(
                        df, x=numeric_cols[0],
                        title=f"Distribution of {numeric_cols[0]}"
                    )
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation heatmap
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                fig2 = px.imshow(
                    corr,
                    title="Feature Correlation Heatmap",
                    color_continuous_scale='RdBu'
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # AI Explanation
            st.subheader("💬 Step 5: AI Explains Your Results")
            
            df_info = {'rows': df.shape[0], 'cols': df.shape[1]}
            
            with st.spinner("AI is writing explanation..."):
                explanation = explain_results(
                    task_type, results, df_info, expertise
                )
            
            st.success(explanation)
            
        else:
            st.error("Something went wrong. Please check your target column.")

else:
    # Show instructions when no file uploaded
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📂 Upload")
        st.markdown("Upload any CSV dataset from your computer")
    
    with col2:
        st.markdown("### 🧠 AI Analyzes")
        st.markdown("AI automatically detects what kind of analysis to do")
    
    with col3:
        st.markdown("### 📊 Get Results")
        st.markdown("See results explained in your language level")