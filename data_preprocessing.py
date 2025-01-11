import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Add a cache-clearing button
if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("Cache cleared successfully!")

# Streamlit App
st.title("Depression Data Preprocessing and EDA")

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    depression_data = pd.read_csv(uploaded_file)

    # 1. Show Raw Dataset
    st.subheader("1. Raw Dataset")
    st.write("This is the raw dataset as loaded from the source:")
    st.write(depression_data.head())

    # 2. Show Summary of the Dataset
    st.subheader("2. Summary of the Dataset")
    st.write("### Dataset Dimensions:")
    st.write(f"Rows: {depression_data.shape[0]}, Columns: {depression_data.shape[1]}")

    st.write("### Missing Values:")
    missing_values = depression_data.isnull().sum()
    columns_with_nulls = missing_values[missing_values > 0]
    st.write(columns_with_nulls)

    st.write("""
**How Missing Values Were Handled:**
- For students:
  - `Academic Pressure`, `CGPA`, `Study Satisfaction` → Filled with mean values.
  - `Work Pressure`, `Job Satisfaction` → Filled with `0` as not applicable.
- For working professionals:
  - `Work Pressure`, `Job Satisfaction` → Filled with mean values.
  - `Academic Pressure`, `CGPA`, `Study Satisfaction` → Filled with `0` as not applicable.
""")
    
    # 3. Handling Null Values
    st.subheader("3. Handling Null Values")
    def handle_missing_values(data):
        """Handle missing values in the dataset and ensure Arrow compatibility."""
        # Separate students and working professionals
        students = data[data['Working Professional or Student'] == 'Student'].copy()
        professionals = data[data['Working Professional or Student'] == 'Working Professional'].copy()

        # Handle missing values for students
        students['Academic Pressure'] = students['Academic Pressure'].fillna(students['Academic Pressure'].mean())
        students['CGPA'] = students['CGPA'].fillna(students['CGPA'].mean())
        students['Study Satisfaction'] = students['Study Satisfaction'].fillna(students['Study Satisfaction'].mean())
        students['Work Pressure'] = students['Work Pressure'].fillna(0)
        students['Job Satisfaction'] = students['Job Satisfaction'].fillna(0)

        # Handle missing values for professionals
        professionals['Work Pressure'] = professionals['Work Pressure'].fillna(professionals['Work Pressure'].mean())
        professionals['Job Satisfaction'] = professionals['Job Satisfaction'].fillna(professionals['Job Satisfaction'].mean())
        professionals['Academic Pressure'] = professionals['Academic Pressure'].fillna(0)
        professionals['CGPA'] = professionals['CGPA'].fillna(0)
        professionals['Study Satisfaction'] = professionals['Study Satisfaction'].fillna(0)

        # Combine the datasets back
        cleaned_data = pd.concat([students, professionals])

        return cleaned_data

    cleaned_data = handle_missing_values(depression_data)
    if st.checkbox("Show Cleaned Dataset After Handling Null Values"):
        st.write("### Cleaned Dataset")
        st.write("Dimensions of the Cleaned Dataset:")
        st.write(f"Rows: {cleaned_data.shape[0]}, Columns: {cleaned_data.shape[1]}")
        st.write(cleaned_data.head())
    
    st.write("### Missing Values Summary After Handling:")
    remaining_nulls = cleaned_data.isnull().sum()
    st.write(remaining_nulls[remaining_nulls > 0] if remaining_nulls.sum() > 0 else "No missing values remaining.")

    # 4. Process Sleep Duration
    st.subheader("4. Process Sleep Duration")

    # Function to process sleep duration
    def process_sleep_duration(value):
        if pd.isnull(value):  # Handle missing values
            return None
        value = value.lower()  # Convert to lowercase for consistency
        if '-' in value:  # If it's a range
            parts = value.replace('hours', '').strip().split('-')
            return (float(parts[0]) + float(parts[1])) / 2  # Take the average
        elif 'more than' in value:  # Handle "more than X hours"
            number = value.replace('more than', '').replace('hours', '').strip()
            return float(number) + 0.5  # Add 0.5 as an approximation
        elif 'less than' in value:  # Handle "less than X hours"
            number = value.replace('less than', '').replace('hours', '').strip()
            return float(number) - 0.5  # Subtract 0.5 as an approximation
        elif 'hours' in value:  # Handle single values like "8 hours"
            return float(value.replace('hours', '').strip())
        else:
            return None  # Handle invalid values (e.g., "unknown")

    # Apply the processing function and replace the original column
    cleaned_data['Sleep Duration'] = cleaned_data['Sleep Duration'].apply(process_sleep_duration)

    if st.checkbox("Show Dataset with Processed Sleep Duration"):
        st.write("### Dataset After Processing Sleep Duration:")
        st.write("Dimensions of the Dataset:")
        st.write(f"Rows: {cleaned_data.shape[0]}, Columns: {cleaned_data.shape[1]}")
        st.write(cleaned_data.head())
    
    # 5. Exploratory Data Analysis
    st.subheader("5. Exploratory Data Analysis")

    # Categorical Feature Distributions
    if st.checkbox("Show Categorical Feature Distributions"):
        categorical_columns = [
            "Gender",
            "Working Professional or Student",
            "Family History of Mental Illness",
            "Dietary Habits",
            "Depression",
            "Have you ever had suicidal thoughts ?",
        ]
        selected_column = st.selectbox("Select a Categorical Column", categorical_columns)
        plt.figure(figsize=(8, 5))
        sns.countplot(data=cleaned_data, x=selected_column, palette="viridis")
        plt.title(f"Distribution of {selected_column}")
        st.pyplot(plt)

    # Numerical Feature Distributions
    if st.checkbox("Show Numerical Feature Distributions"):
        numeric_columns = [
            "Age",
            "Work Pressure",
            "Job Satisfaction",
            "Work/Study Hours",
            "Financial Stress",
            "Sleep Duration",
        ]
        selected_column = st.selectbox("Select a Numerical Column", numeric_columns)
        plt.figure(figsize=(8, 5))
        sns.histplot(cleaned_data[selected_column].dropna(), kde=True, bins=20, color="blue")
        plt.title(f"Distribution of {selected_column}")
        st.pyplot(plt)

    # Correlation Matrix
    if st.checkbox("Show Correlation Matrix"):
        numeric_data = cleaned_data[[
            "Age", "Work Pressure", "Job Satisfaction", "Work/Study Hours", "Financial Stress", "Sleep Duration"
        ]]
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_data.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title("Correlation Matrix")
        st.pyplot(plt)

    # Feature-Target Relationships
    if st.checkbox("Show Feature-Target Relationships"):
        target = "Depression"
        relationship_columns = ["Work Pressure", "Job Satisfaction", "Work/Study Hours", "Financial Stress", "Sleep Duration"]
        selected_column = st.selectbox("Select a Feature", relationship_columns)
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=cleaned_data, x=target, y=selected_column, palette="viridis")
        plt.title(f"Relationship between {selected_column} and {target}")
        st.pyplot(plt)

    st.write("### Data Preprocessing and EDA Completed!")
else:
    st.write("Please upload a CSV file to proceed.")
