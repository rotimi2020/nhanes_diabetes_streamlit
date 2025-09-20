import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import joblib
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Assessment Suite",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base directory (where app.py lives) - safe relative paths for Streamlit Cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------- Custom CSS for updated styling -----------------
st.markdown("""
<style>
    body {
        background-color: #f0f2f6;
        color: #31333F;
    }
    .stApp {
        background: #ffffff;
    }
    .main-header {
        font-size: 2.2rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #FF4B4B;
    }
    .section-header {
        font-size: 1.6rem;
        color: #FF4B4B;
        margin-top: 1.8rem;
        margin-bottom: 1rem;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #ddd;
    }
    .subsection-header {
        font-size: 1.3rem;
        color: #FF4B4B;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .info-box {
        background-color: #FFF5F5;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #FF4B4B;
        font-size: 0.95rem;
        color: #31333F;
    }
    .metric-box {
        background-color: #FFF5F5;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin: 10px 0;
        border: 1px solid #FFCCCB;
        color: #31333F;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #FF4B4B;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666666;
    }
    .stDataFrame {
        background-color: #ffffff;
        color: #31333F;
    }
    .stSlider, .stNumberInput, .stSelectbox, .stRadio {
        font-size: 0.9rem;
        color: #31333F;
    }
    /* Sidebar styling */
    .css-1d391kg, .css-1d391kg p {
        background-color: #ffffff;
        color: #31333F;
    }
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
        color: #FF4B4B;
    }
    /* Hide empty text elements */
    .stMarkdown:has(> div > div > p:empty) {
        display: none;
    }
    /* Prediction app styles */
    .sub-header {
        font-size: 1.2rem;
        color: #2c3e50;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }
    .risk-high {
        background-color: #ffcccc;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ff4b4b;
        margin: 8px 0;
        font-size: 0.9rem;
    }
    .risk-medium {
        background-color: #fff4cc;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 8px 0;
        font-size: 0.9rem;
    }
    .risk-low {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 8px 0;
        font-size: 0.9rem;
    }
    .result-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border: 1px solid #dee2e6;
        font-size: 0.9rem;
    }
    .screening-box {
        border-color: #ffc107;
        background-color: #fff3cd;
    }
    .diagnostic-box {
        border-color: #dc3545;
        background-color: #f8d7da;
    }
    .positive-result {
        color: #dc3545;
        font-weight: bold;
        font-size: 1rem;
    }
    .negative-result {
        color: #28a745;
        font-weight: bold;
        font-size: 1rem;
    }
    .prediction-info-box {
        background-color: #e8f4f8;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 4px solid #17a2b8;
        font-size: 0.9rem;
    }
    .summary-box {
        background-color: #e9ecef;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        border-left: 4px solid #6c757d;
        font-size: 0.9rem;
    }
    .probability-display {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin: 15px 0;
        padding: 12px;
        background-color: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Helper: allowed-file checks -----------------
def _is_csv(filename: str) -> bool:
    return filename.lower().endswith('.csv')

def _is_pkl(filename: str) -> bool:
    return filename.lower().endswith('.pkl')

# ----------------- Load data for visualization (repo or upload) -----------------
@st.cache_data
def load_data_from_path(path: str):
    df = pd.read_csv(path)
    return df

@st.cache_data
def load_data_from_bytes(bytes_io: bytes):
    return pd.read_csv(io.BytesIO(bytes_io))

def get_dataset():
    """
    Priority:
    1. Sidebar uploaded CSV (user)
    2. CSV in same folder as app (repo)
    Returns: DataFrame or None
    """
    uploaded = st.sidebar.file_uploader("Upload dataset (.csv)", type=['csv'], help="Optional: upload nhanes_analysis.csv")
    if uploaded is not None:
        try:
            df = load_data_from_bytes(uploaded.read())
            st.sidebar.success("Dataset uploaded.")
            return df
        except Exception as e:
            st.sidebar.error(f"Uploaded CSV could not be read: {e}")
            return None
    # fallback to repo file
    csv_path = os.path.join(BASE_DIR, "nhanes_analysis.csv")
    if os.path.exists(csv_path):
        try:
            return load_data_from_path(csv_path)
        except Exception as e:
            st.sidebar.error(f"Failed to load CSV from repo: {e}")
            return None
    else:
        st.sidebar.info("No CSV found in repo and no uploaded CSV. Upload one or add 'nhanes_analysis.csv' to the app folder.")
        return None

# ----------------- Load model for prediction (repo or upload) -----------------
@st.cache_resource
def load_model_from_path(model_path: str, scaler_path: str, features_path: str):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)
    return model, scaler, feature_names

def load_model():
    """
    Priority:
    1. User-uploaded PKL files via sidebar (model, scaler, feature names)
    2. Files in repo next to app.py
    Returns: (model, scaler, feature_names) or (None, None, None)
    """
    st.sidebar.markdown("## Model files (optional upload)")
    uploaded_model = st.sidebar.file_uploader("Upload model (.pkl)", type=['pkl'], help="RandomForest or similar .pkl")
    uploaded_scaler = st.sidebar.file_uploader("Upload scaler (.pkl)", type=['pkl'], help="scaler.pkl")
    uploaded_features = st.sidebar.file_uploader("Upload feature names (.pkl)", type=['pkl'], help="feature_names.pkl (list)")

    # If user provided all files via uploader
    if uploaded_model and uploaded_scaler and uploaded_features:
        try:
            model = joblib.load(io.BytesIO(uploaded_model.read()))
            scaler = joblib.load(io.BytesIO(uploaded_scaler.read()))
            feature_names = joblib.load(io.BytesIO(uploaded_features.read()))
            st.sidebar.success("Model, scaler and feature names uploaded.")
            return model, scaler, feature_names
        except Exception as e:
            st.sidebar.error(f"Uploaded PKL files couldn't be loaded: {e}")
            return None, None, None

    # Fallback to files in repo
    model_path = os.path.join(BASE_DIR, "diabetes_model.pkl")
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
    features_path = os.path.join(BASE_DIR, "feature_names.pkl")

    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path):
        try:
            return load_model_from_path(model_path, scaler_path, features_path)
        except Exception as e:
            st.sidebar.error(f"Failed to load PKL files from repo: {e}")
            return None, None, None

    st.sidebar.info("No model files found in repo and not all uploaded. Add 'diabetes_model.pkl', 'scaler.pkl', and 'feature_names.pkl' to the app folder or upload via sidebar.")
    return None, None, None

# ----------------- Palettes and style -----------------
red_palette = ["#FF4B4B", "#FF6B6B", "#FF8E8E", "#FFA8A8", "#FFC2C2", "#FFDCDC"]
red_cmap = LinearSegmentedColormap.from_list("red", ["#FFFFFF", "#FFDCDC", "#FFA8A8", "#FF6B6B", "#FF4B4B"])

plt.style.use('default')
sns.set_palette(red_palette)

# ----------------- Mapping dictionaries for prediction -----------------
Gender_Code = {1: 'Male', 2: 'Female'}
Race_Code = {
    1: 'Mexican American',
    2: 'Other Hispanic',
    3: 'Non-Hispanic White',
    4: 'Non-Hispanic Black',
    6: 'Non-Hispanic Asian',
    7: 'Other Race'
}
Education_Code_Imputed = {
    1: 'Less than 9th grade',
    2: '9-11th grade',
    3: 'High school graduate',
    4: 'Some college or AA degree',
    5: 'College graduate'
}
Family_Diabetes_Code_Imputed = {1: 'Yes', 2: 'No'}
Risk_Level = {0: 'High Risk', 1: 'Low Risk'}
Obesity_Status = {0: 'Non-Obese', 1: 'Obese', 2: 'Overweight'}

# Thresholds
SCREENING_THRESHOLD = 0.7
DIAGNOSTIC_THRESHOLD = 0.9

# ----------------- Visualization app -----------------
def visualization_app():
    st.markdown('<h1 class="main-header">Diabetes Risk Assessment Data Visualization</h1>', unsafe_allow_html=True)

    df = get_dataset()
    if df is None:
        st.warning("No data available to visualize. Upload a CSV or add 'nhanes_analysis.csv' to the app folder.")
        return

    # Ensure expected columns exist - minimal check
    required_cols = ['Age_Imputed', 'Diabetes_Status', 'Gender', 'BMI_Imputed', 'Waist_Circumference_Imputed']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Dataset missing required columns: {missing}")
        st.write("Dataset columns:", list(df.columns))
        return

    # Sidebar filters
    st.sidebar.markdown("## Data Filters")

    # Age filter
    try:
        min_age, max_age = int(df['Age_Imputed'].min()), int(df['Age_Imputed'].max())
    except Exception:
        st.error("Column 'Age_Imputed' must be numeric.")
        return

    age_range = st.sidebar.slider("Age Range", min_age, max_age, (min_age, max_age))

    # Diabetes status filter
    diabetes_options = list(df['Diabetes_Status'].unique())
    selected_diabetes = st.sidebar.multiselect("Diabetes Status", diabetes_options, diabetes_options)

    # Gender filter
    gender_options = list(df['Gender'].unique())
    selected_gender = st.sidebar.multiselect("Gender", gender_options, gender_options)

    # Apply filters
    filtered_df = df[
        (df['Age_Imputed'] >= age_range[0]) &
        (df['Age_Imputed'] <= age_range[1]) &
        (df['Diabetes_Status'].isin(selected_diabetes)) &
        (df['Gender'].isin(selected_gender))
    ]

    # Display dataset info
    st.markdown('<h2 class="section-header">Dataset Overview</h2>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f'''
            <div class="metric-box">
                <div class="metric-value">{filtered_df.shape[0]}</div>
                <div class="metric-label">Total Records</div>
            </div>
        ''', unsafe_allow_html=True)

    with col2:
        diabetes_count = filtered_df[filtered_df['Diabetes_Status'] == 'Diabetes'].shape[0]
        st.markdown(f'''
            <div class="metric-box">
                <div class="metric-value">{diabetes_count}</div>
                <div class="metric-label">Diabetes Cases</div>
            </div>
        ''', unsafe_allow_html=True)

    with col3:
        avg_age = filtered_df['Age_Imputed'].mean()
        st.markdown(f'''
            <div class="metric-box">
                <div class="metric-value">{avg_age:.1f}</div>
                <div class="metric-label">Average Age</div>
            </div>
        ''', unsafe_allow_html=True)

    with col4:
        avg_bmi = filtered_df['BMI_Imputed'].mean()
        st.markdown(f'''
            <div class="metric-box">
                <div class="metric-value">{avg_bmi:.1f}</div>
                <div class="metric-label">Average BMI</div>
            </div>
        ''', unsafe_allow_html=True)

    # Show filtered data
    if st.checkbox("Show Filtered Data"):
        st.dataframe(filtered_df)

    # Distribution analysis
    st.markdown('<h2 class="section-header">Distribution Analysis</h2>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<h3 class="subsection-header">Diabetes Status</h3>', unsafe_allow_html=True)
        diabetes_counts = filtered_df['Diabetes_Status'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 5))
        wedges, texts, autotexts = ax.pie(diabetes_counts.values, labels=diabetes_counts.index, autopct='%1.1f%%',
                                         colors=red_palette[:len(diabetes_counts)])
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax.set_title('Diabetes Status Distribution')
        st.pyplot(fig)

    with col2:
        st.markdown('<h3 class="subsection-header">Risk Level</h3>', unsafe_allow_html=True)
        if 'Risk_Level' in filtered_df.columns:
            risk_counts = filtered_df['Risk_Level'].value_counts()
        else:
            risk_counts = pd.Series([], dtype=int)
        fig, ax = plt.subplots(figsize=(6, 5))
        if not risk_counts.empty:
            bars = ax.bar(risk_counts.index, risk_counts.values, color=red_palette[:len(risk_counts)])
            ax.set_title('Risk Level Distribution')
            ax.set_ylabel('Count')
            plt.xticks(rotation=45)
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No Risk_Level column', ha='center')
        st.pyplot(fig)

    with col3:
        st.markdown('<h3 class="subsection-header">Obesity Status</h3>', unsafe_allow_html=True)
        if 'Obesity_Status' in filtered_df.columns:
            obesity_counts = filtered_df['Obesity_Status'].value_counts()
            fig, ax = plt.subplots(figsize=(6, 5))
            bars = ax.bar(obesity_counts.index, obesity_counts.values, color=red_palette[:len(obesity_counts)])
            ax.set_title('Obesity Status Distribution')
            ax.set_ylabel('Count')
            plt.xticks(rotation=45)
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom')
            st.pyplot(fig)
        else:
            st.info("No 'Obesity_Status' column in dataset.")

    # Health metrics comparison
    st.markdown('<h2 class="section-header">Health Metrics by Diabetes Status</h2>', unsafe_allow_html=True)
    cols_for_compare = [c for c in ["BMI_Imputed", "Waist_Circumference_Imputed", "Glucose_Imputed", "Triglycerides_Imputed"] if c in filtered_df.columns]
    if cols_for_compare:
        diabetes_comparison = filtered_df.groupby('Diabetes_Status')[cols_for_compare].mean().round(2)
        st.dataframe(diabetes_comparison.style.background_gradient(cmap=red_cmap))
    else:
        st.info("Not enough numeric columns to show health metric comparison.")

    # BMI Comparison by Gender and Diabetes Status
    st.markdown('<h2 class="section-header">BMI Comparison by Gender and Diabetes Status</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h3 class="subsection-header">Males</h3>', unsafe_allow_html=True)
        if 'Gender' in filtered_df.columns and 'BMI_Imputed' in filtered_df.columns:
            male_bmi = filtered_df.query("Gender == 'Male'").groupby("Diabetes_Status")["BMI_Imputed"].mean().round(2)
            fig, ax = plt.subplots(figsize=(6, 5))
            bars = ax.bar(male_bmi.index, male_bmi.values, color=red_palette[:len(male_bmi)])
            ax.set_title('Average BMI - Males')
            ax.set_ylabel('BMI')
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                        f'{height}', ha='center', va='bottom')
            st.pyplot(fig)
        else:
            st.info("Missing 'Gender' or 'BMI_Imputed' for male BMI plot.")

    with col2:
        st.markdown('<h3 class="subsection-header">Females</h3>', unsafe_allow_html=True)
        if 'Gender' in filtered_df.columns and 'BMI_Imputed' in filtered_df.columns:
            female_bmi = filtered_df.query("Gender == 'Female'").groupby("Diabetes_Status")["BMI_Imputed"].mean().round(2)
            fig, ax = plt.subplots(figsize=(6, 5))
            bars = ax.bar(female_bmi.index, female_bmi.values, color=red_palette[:len(female_bmi)])
            ax.set_title('Average BMI - Females')
            ax.set_ylabel('BMI')
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                        f'{height}', ha='center', va='bottom')
            st.pyplot(fig)
        else:
            st.info("Missing 'Gender' or 'BMI_Imputed' for female BMI plot.")

    # Income analysis
    st.markdown('<h2 class="section-header">Income Analysis</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h3 class="subsection-header">Income by Obesity Status</h3>', unsafe_allow_html=True)
        if 'Obesity_Status' in filtered_df.columns and 'Income' in filtered_df.columns:
            income_obesity = filtered_df.groupby("Obesity_Status")["Income"].mean().round(2)
            fig, ax = plt.subplots(figsize=(6, 5))
            bars = ax.bar(income_obesity.index, income_obesity.values, color=red_palette[:len(income_obesity)])
            ax.set_title('Average Income by Obesity Status')
            ax.set_ylabel('Income ($)')
            plt.xticks(rotation=45)
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                        f'${height:,.0f}', ha='center', va='bottom')
            st.pyplot(fig)
        else:
            st.info("Missing 'Obesity_Status' or 'Income' for income analysis.")

    with col2:
        st.markdown('<h3 class="subsection-header">Income by Education Level (Diabetes Patients)</h3>', unsafe_allow_html=True)
        if 'Education' in filtered_df.columns and 'Income' in filtered_df.columns and 'Diabetes_Status' in filtered_df.columns:
            income_education = filtered_df.query("Diabetes_Status == 'Diabetes'").groupby('Education')['Income'].mean()
            fig, ax = plt.subplots(figsize=(6, 5))
            bars = ax.bar(income_education.index, income_education.values, color=red_palette[:len(income_education)])
            ax.set_title('Income by Education (Diabetes Patients)')
            ax.set_ylabel('Income ($)')
            plt.xticks(rotation=45)
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                        f'${height:,.0f}', ha='center', va='bottom')
            st.pyplot(fig)
        else:
            st.info("Missing 'Education', 'Income' or 'Diabetes_Status' for income-by-education plot.")

    # Detailed visualizations (age, BMI, counts)
    st.markdown('<h2 class="section-header">Detailed Visualizations</h2>', unsafe_allow_html=True)

    # Age Distribution
    st.markdown('<h3 class="subsection-header">Age Distribution</h3>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(filtered_df['Age_Imputed'], bins=20, kde=True, color=red_palette[0])
    plt.title('Age Distribution (Imputed)', fontsize=14, fontweight='bold')
    plt.xlabel('Age (years)')
    plt.ylabel('Frequency')
    st.pyplot(fig)

    # BMI Distribution
    st.markdown('<h3 class="subsection-header">BMI Distribution</h3>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(filtered_df['BMI_Imputed'], bins=20, kde=True, color=red_palette[1])
    plt.title('BMI Distribution (Imputed)', fontsize=14, fontweight='bold')
    plt.xlabel('BMI')
    plt.ylabel('Frequency')
    st.pyplot(fig)

    # Diabetes by Gender
    st.markdown('<h3 class="subsection-header">Diabetes Prevalence by Gender</h3>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=filtered_df, x='Gender', hue='Diabetes_Status', palette=red_palette[:2])
    plt.title('Diabetes Prevalence by Gender', fontsize=14, fontweight='bold')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.legend(title='Diabetes Status')
    st.pyplot(fig)

    # Diabetes by Race
    st.markdown('<h3 class="subsection-header">Diabetes Prevalence by Race/Ethnicity</h3>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'Race' in filtered_df.columns:
        sns.countplot(data=filtered_df, y='Race', hue='Diabetes_Status', palette=red_palette[:2])
        plt.title('Diabetes Prevalence by Race/Ethnicity', fontsize=14, fontweight='bold')
        plt.xlabel('Count')
        plt.ylabel('Race/Ethnicity')
        plt.legend(title='Diabetes Status')
        st.pyplot(fig)
    else:
        st.info("No 'Race' column to plot diabetes by race.")

    # Correlation Heatmap (imputed numeric features)
    st.markdown('<h3 class="subsection-header">Feature Correlation Matrix</h3>', unsafe_allow_html=True)
    imputed_numeric_features = [col for col in filtered_df.columns if col.endswith("_Imputed") and col != 'Income_Code_Imputed']
    if len(imputed_numeric_features) >= 2:
        correlation_matrix = filtered_df[imputed_numeric_features].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap=red_cmap, center=0,
                    square=True, cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Matrix (Imputed Values)', fontsize=14, fontweight='bold')
        st.pyplot(fig)
    else:
        st.info("Not enough imputed numeric features to compute correlation matrix.")

    # BMI by Diabetes Status
    st.markdown('<h3 class="subsection-header">BMI Distribution by Diabetes Status</h3>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='Diabetes_Status', y='BMI_Imputed', data=filtered_df, palette=red_palette[:2])
    plt.title('BMI Distribution by Diabetes Status', fontsize=14, fontweight='bold')
    plt.xlabel('Diabetes Status')
    plt.ylabel('BMI')
    st.pyplot(fig)

# ----------------- Prediction app -----------------
def prediction_app():
    st.markdown('<h1 class="main-header">Diabetes Risk Assessment</h1>', unsafe_allow_html=True)

    model, scaler, feature_names = load_model()
    if model is None or scaler is None or feature_names is None:
        st.warning("Model not available. Upload model files in the sidebar or add them to the app folder.")
        return

    # if feature_names is a numpy array, convert to list
    if isinstance(feature_names, (np.ndarray,)):
        feature_names = feature_names.tolist()

    # Basic sanity check
    if not isinstance(feature_names, (list, tuple)) or len(feature_names) == 0:
        st.error("Loaded 'feature_names' must be a non-empty list of column names used in model training.")
        return

    st.markdown("""
    <div class="prediction-info-box">
        <strong>Clinical Guidance:</strong> This tool uses a dual-threshold approach:
        <ul>
            <li><strong>Screening (70%):</strong> High recall to identify potential cases</li>
            <li><strong>Diagnostic (90%):</strong> High precision to confirm diagnosis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Create input form
    st.markdown('<h2 class="sub-header">Patient Information</h2>', unsafe_allow_html=True)

    # Demographics
    st.markdown("**Demographic Information**")
    age = st.slider("Age", min_value=1, max_value=100, value=45)
    gender = st.radio("Gender", options=list(Gender_Code.values()), horizontal=True)
    race = st.selectbox("Race/Ethnicity", options=list(Race_Code.values()))
    education = st.selectbox("Education Level", options=list(Education_Code_Imputed.values()))
    income = st.number_input("Annual Income ($)", min_value=2500, max_value=100000, value=50000, step=1000)

    # Clinical measurements
    st.markdown("**Clinical Measurements**")
    col1, col2 = st.columns(2)
    with col1:
        bmi = st.slider("BMI", min_value=10.0, max_value=85.0, value=25.0, step=0.1)
        waist_circumference = st.slider("Waist Circumference (cm)", min_value=40.0, max_value=180.0, value=90.0)
        systolic_bp = st.slider("Systolic BP (mmHg)", min_value=65, max_value=230, value=120)
    with col2:
        diastolic_bp = st.slider("Diastolic BP (mmHg)", min_value=0, max_value=130, value=80)
        glucose = st.slider("Glucose Level (mg/dL)", min_value=40, max_value=610, value=100)
        hdl = st.slider("HDL Level (mg/dL)", min_value=10, max_value=125, value=50)
        triglycerides = st.slider("Triglycerides Level (mg/dL)", min_value=10, max_value=4250, value=150)

    # Risk factors
    st.markdown("**Risk Factors**")
    family_diabetes = st.radio("Family History of Diabetes", options=list(Family_Diabetes_Code_Imputed.values()), horizontal=True)
    risk_level = st.selectbox("Clinical Risk Level", options=list(Risk_Level.values()))
    obesity_status = st.selectbox("Obesity Status", options=list(Obesity_Status.values()))

    # Convert categorical inputs back to numerical codes
    gender_code = [k for k, v in Gender_Code.items() if v == gender][0]
    race_code = [k for k, v in Race_Code.items() if v == race][0]
    education_code = [k for k, v in Education_Code_Imputed.items() if v == education][0]
    family_diabetes_code = [k for k, v in Family_Diabetes_Code_Imputed.items() if v == family_diabetes][0]
    risk_level_code = [k for k, v in Risk_Level.items() if v == risk_level][0]
    obesity_status_code = [k for k, v in Obesity_Status.items() if v == obesity_status][0]

    # Create feature vector in the same order as training
    # We'll attempt to build a dict of available named features, then align to feature_names
    candidate = {
        'Age': age,
        'Gender_Code': gender_code,
        'Race_Code': race_code,
        'BMI_Imputed': bmi,
        'Waist_Circumference_Imputed': waist_circumference,
        'Systolic_BP_Imputed': systolic_bp,
        'Diastolic_BP_Imputed': diastolic_bp,
        'Glucose_Imputed': glucose,
        'HDL_Imputed': hdl,
        'Triglycerides_Imputed': triglycerides,
        'Education_Code_Imputed': education_code,
        'Family_Diabetes_Code_Imputed': family_diabetes_code,
        'Income': income,
        'Risk_Level': risk_level_code,
        'Obesity_Status': obesity_status_code
    }

    # Build input_data with same columns order as feature_names
    try:
        input_row = [candidate.get(col, 0) for col in feature_names]  # default 0 if missing
        input_data = pd.DataFrame([input_row], columns=feature_names)
    except Exception as e:
        st.error(f"Failed to prepare input vector using feature_names: {e}")
        return

    # Scale and predict
    try:
        input_scaled = scaler.transform(input_data)
    except Exception as e:
        st.error(f"Scaler transform failed: {e}")
        st.write("Check that scaler matches feature order and types.")
        return

    if st.button("Assess Diabetes Risk", type="primary", use_container_width=True):
        try:
            probability = model.predict_proba(input_scaled)[0][1]
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            return

        probability_percent = probability * 100

        st.markdown("---")
        st.markdown('<h2 class="sub-header">Assessment Results</h2>', unsafe_allow_html=True)

        # Display the probability in percentage
        st.markdown(f'<div class="probability-display">Diabetes Probability: {probability_percent:.1f}%</div>', unsafe_allow_html=True)

        # Screening result
        st.markdown(f'<div class="result-box screening-box">', unsafe_allow_html=True)
        st.markdown("##### Screening Result (High Recall)")
        if probability >= SCREENING_THRESHOLD:
            st.markdown('<p class="positive-result">SCREENING POSITIVE</p>', unsafe_allow_html=True)
            st.markdown(f"*Probability ({probability_percent:.1f}%) ≥ Screening Threshold ({SCREENING_THRESHOLD*100:.0f}%)*")
            st.markdown("**Recommendation:** Refer for diagnostic testing")
        else:
            st.markdown('<p class="negative-result">SCREENING NEGATIVE</p>', unsafe_allow_html=True)
            st.markdown(f"*Probability ({probability_percent:.1f}%) < Screening Threshold ({SCREENING_THRESHOLD*100:.0f}%)*")
            st.markdown("**Recommendation:** No further testing needed")
        st.markdown('</div>', unsafe_allow_html=True)

        # Diagnostic result
        st.markdown(f'<div class="result-box diagnostic-box">', unsafe_allow_html=True)
        st.markdown("##### Diagnostic Result (High Precision)")
        if probability >= DIAGNOSTIC_THRESHOLD:
            st.markdown('<p class="positive-result">DIAGNOSTIC POSITIVE</p>', unsafe_allow_html=True)
            st.markdown(f"*Probability ({probability_percent:.1f}%) ≥ Diagnostic Threshold ({DIAGNOSTIC_THRESHOLD*100:.0f}%)*")
            st.markdown("**Recommendation:** High confidence of diabetes")
        else:
            st.markdown('<p class="negative-result">DIAGNOSTIC NEGATIVE</p>', unsafe_allow_html=True)
            st.markdown(f"*Probability ({probability_percent:.1f}%) < Diagnostic Threshold ({DIAGNOSTIC_THRESHOLD*100:.0f}%)*")
            st.markdown("**Recommendation:** Insufficient evidence for diagnosis")
        st.markdown('</div>', unsafe_allow_html=True)

        # Summary
        st.markdown('<div class="summary-box">', unsafe_allow_html=True)
        st.markdown("##### Clinical Summary")

        if probability >= DIAGNOSTIC_THRESHOLD:
            st.markdown("**HIGH CONFIDENCE OF DIABETES**")
            st.markdown("Immediate confirmatory testing and treatment planning recommended.")
        elif probability >= SCREENING_THRESHOLD:
            st.markdown("**ELEVATED RISK OF DIABETES**")
            st.markdown("Further diagnostic testing advised.")
        else:
            st.markdown("**LOW RISK OF DIABETES**")
            st.markdown("Continue routine preventive care.")

        st.markdown('</div>', unsafe_allow_html=True)

        # Risk factors visualization
        st.markdown("---")
        st.markdown('<h3 class="sub-header">Risk Factor Analysis</h3>', unsafe_allow_html=True)

        # Feature importance (if present)
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
            indices = np.argsort(feature_importance)[::-1]
            top = min(8, len(feature_importance))
            top_features = [feature_names[i] for i in indices[:top]]
            top_importance = [feature_importance[i] for i in indices[:top]]

            feature_name_map = {
                'Age': 'Age',
                'Gender_Code': 'Gender',
                'Race_Code': 'Race/Ethnicity',
                'BMI_Imputed': 'BMI',
                'Waist_Circumference_Imputed': 'Waist Circumference',
                'Systolic_BP_Imputed': 'Systolic BP',
                'Diastolic_BP_Imputed': 'Diastolic BP',
                'Glucose_Imputed': 'Glucose Level',
                'HDL_Imputed': 'HDL Level',
                'Triglycerides_Imputed': 'Triglycerides',
                'Education_Code_Imputed': 'Education Level',
                'Family_Diabetes_Code_Imputed': 'Family History',
                'Income': 'Income',
                'Risk_Level': 'Clinical Risk Level',
                'Obesity_Status': 'Obesity Status'
            }

            readable_features = [feature_name_map.get(f, f) for f in top_features]

            fig = go.Figure(go.Bar(
                x=top_importance,
                y=readable_features,
                orientation='h',
                marker_color='#3498db'
            ))

            fig.update_layout(
                title="Top Influential Risk Factors",
                xaxis_title="Importance",
                yaxis_title="Factor",
                height=300,
                margin=dict(l=10, r=10, t=30, b=10)
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Model does not expose feature_importances_.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="prediction-info-box">
    <strong>Disclaimer:</strong> This tool is for clinical decision support only and should not replace professional medical judgment. 
    </div>
    """, unsafe_allow_html=True)

    # Sidebar info repeated for context
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center;">
        <h3>Diabetes Risk Assessor</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Threshold Information")

        st.markdown("""
        <div class="result-box screening-box">
        <h4>Screening Threshold</h4>
        <h3>70%</h3>
        <p>High recall for community screening</p>
        </div>
        """ , unsafe_allow_html=True)

        st.markdown("""
        <div class="result-box diagnostic-box">
        <h4>Diagnostic Threshold</h4>
        <h3>90%</h3>
        <p>High precision for clinical diagnosis</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Interpretation Guide")

        st.markdown("""
        - **< 70%:** Screening Negative
        - **70-89%:** Screening Positive
        - **≥ 90%:** Diagnostic Positive
        """)

        st.markdown("---")
        st.caption("Clinical Decision Support Tool")

# ----------------- Main app -----------------
def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Select Mode", ["Data Visualization", "Risk Assessment"])

    if app_mode == "Data Visualization":
        visualization_app()
    else:
        prediction_app()

if __name__ == "__main__":
    main()
