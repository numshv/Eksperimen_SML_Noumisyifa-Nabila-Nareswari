import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(raw_path):
    """Load raw dataset dari CSV."""
    df = pd.read_csv(raw_path)
    return df


def clean_data(df):
    """Drop kolom tidak informatif & handle missing value."""
    df_clean = df.drop(columns=['customerID'])
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].median())
    return df_clean


def encode_features(df):
    """Encoding target dan fitur kategorikal."""
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df_encoded


def split_and_scale(df_encoded, test_size=0.2, random_state=42):
    """Split train/test dan scaling fitur numerik."""
    X = df_encoded.drop(columns=['Churn'])
    y = df_encoded['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X_train[num_features] = scaler.fit_transform(X_train[num_features])
    X_test[num_features] = scaler.transform(X_test[num_features])

    return X_train, X_test, y_train, y_test


def save_output(X_train, X_test, y_train, y_test, output_dir):
    """Simpan hasil preprocessing ke folder output."""
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
    X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
    y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{output_dir}/y_test.csv', index=False)


def run_pipeline(raw_path, output_dir):
    """Jalankan seluruh pipeline preprocessing secara otomatis."""
    df = load_data(raw_path)
    df_clean = clean_data(df)
    df_encoded = encode_features(df_clean)
    X_train, X_test, y_train, y_test = split_and_scale(df_encoded)
    save_output(X_train, X_test, y_train, y_test, output_dir)
    print(f"Pipeline selesai. Output tersimpan di: {output_dir}")


if __name__ == "__main__":
    RAW_PATH = "Telco-Customer-Churn_raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    OUTPUT_DIR = "preprocessing/AUTOMATE_Telco-Customer-Churn_preprocessing"
    run_pipeline(RAW_PATH, OUTPUT_DIR)