# pipeline.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# STEP 1: EXTRACT
def extract_data(file_path):
    logging.info("Extracting data from: %s", file_path)
    return pd.read_csv(file_path)

# STEP 2: TRANSFORM
def transform_data(df):
    logging.info("Starting transformation process...")

    # Handle missing values
    df.fillna(method='ffill', inplace=True)

    # Encode categorical variables
    if 'Sex' in df.columns:
        le = LabelEncoder()
        df['Sex'] = le.fit_transform(df['Sex'])

    # Scale numerical values
    numeric_cols = df.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    logging.info("Transformation complete.")
    return df

# STEP 3: LOAD
def load_data(df, output_path):
    logging.info("Saving transformed data to: %s", output_path)
    df.to_csv(output_path, index=False)

# MAIN FUNCTION
def main():
    input_path = r"C:\Users\Bandla\Desktop\CodeTech_intern\Task_1\data\Titanic_dataset.csv"
    output_path = r"C:\Users\Bandla\Desktop\CodeTech_intern\Task_1\cleaned_titanic.csv"

    df = extract_data(input_path)
    transformed_df = transform_data(df)
    load_data(transformed_df, output_path)
    logging.info("✅ Data pipeline executed successfully.")

if __name__ == "__main__":
    main()
