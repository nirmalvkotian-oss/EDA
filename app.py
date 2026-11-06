import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Automated EDA & Engineering", layout="wide")

st.title("⚡ Automated EDA & Feature Engineering Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    # --- 1. DATA CLEANING SECTION ---
    st.subheader("🛠️ Automated Data Cleaning")
    
    colA, colB = st.columns(2)
    with colA:
        clean_missing = st.checkbox("Fill Missing Values")
        if clean_missing:
            num_cols = df.select_dtypes(include=np.number).columns
            cat_cols = df.select_dtypes(exclude=np.number).columns
            for col in num_cols:
                df[col] = df[col].fillna(df[col].median())
            for col in cat_cols:
                if not df[col].mode().empty:
                    df[col] = df[col].fillna(df[col].mode()[0])
            st.info("Missing values filled (Median for numbers, Mode for text).")

    with colB:
        cap_outliers = st.checkbox("Cap Outliers (IQR Method)")
        if cap_outliers:
            num_cols = df.select_dtypes(include=np.number).columns
            for col in num_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
                df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
            st.info("Outliers capped to prevent skewed graphs.")
            
    # --- 2. FEATURE ENGINEERING SECTION ---
    st.subheader("⚙️ Automated Feature Engineering")
    
    colC, colD = st.columns(2)
    with colC:
        encode_cats = st.checkbox("One-Hot Encode Categorical Columns")
        if encode_cats:
            # Find text columns with fewer than 15 unique values to avoid exploding the dataset size
            cat_cols = [col for col in df.select_dtypes(exclude=np.number).columns if df[col].nunique() < 15]
            if cat_cols:
                # drop_first=True prevents multicollinearity 
                df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
                st.info(f"One-Hot Encoded: {', '.join(cat_cols)}")
            else:
                st.warning("No low-cardinality categorical columns found to encode.")

    with colD:
        scale_nums = st.checkbox("Standardize Numerical Features")
        if scale_nums:
            # Applies StandardScaler to center data around 0 with a variance of 1
            num_cols = df.select_dtypes(include=np.number).columns
            scaler = StandardScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])
            st.info("Numerical features standardized successfully.")
    # ----------------------------------

    # Show basic info (This will dynamically update if checkboxes are ticked)
    st.subheader("1. Processed Dataset Preview")
    st.dataframe(df.head())

    st.subheader("2. Dataset Shape & Data Types")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Shape:", df.shape)
    with col2:
        st.write("Data Types:")
        st.write(df.dtypes.astype(str))

    st.subheader("3. Missing Values Count")
    st.write(df.isnull().sum())

    st.subheader("4. Descriptive Statistics")
    st.dataframe(df.describe())

    st.subheader("5. Correlation Heatmap")
    # Because one-hot encoding turns text into numbers, the heatmap will automatically include encoded categories
    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
        st.pyplot(fig)
    else:
        st.warning("No numerical features available for correlation.")

    st.subheader("6. Visualizations")
    num_cols_list = list(numeric_df.columns)
    
    if len(num_cols_list) > 0:
        col_x = st.selectbox("Select X-axis", options=num_cols_list)
        col_y = st.selectbox("Select Y-axis", options=num_cols_list, index=1 if len(num_cols_list) > 1 else 0)
        chart_type = st.selectbox("Chart Type", options=["Scatter", "Line", "Bar"])

        fig2, ax2 = plt.subplots()
        if chart_type == "Scatter":
            sns.scatterplot(data=df, x=col_x, y=col_y, ax=ax2)
        elif chart_type == "Line":
            sns.lineplot(data=df, x=col_x, y=col_y, ax=ax2)
        elif chart_type == "Bar":
            sns.barplot(data=df, x=col_x, y=col_y, ax=ax2)
        st.pyplot(fig2)
    else:
        st.warning("No numerical columns available for visualization.")

else:
    st.info("Upload a CSV file to begin EDA & Feature Engineering.")