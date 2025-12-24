import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import joblib

def perform_preprocessing(raw_data_path, output_dir, target_column='final_grade', save_scalers=True):
    # LOAD & CLEAN
    if not os.path.exists(raw_data_path):
        print(f"[ERROR] File tidak ditemukan: {raw_data_path}")
        return

    print("Loading Data...")
    df = pd.read_csv(raw_data_path)
    
    # Hapus Duplikat & Kolom ID
    df = df.drop_duplicates()
    if 'student_id' in df.columns:
        df = df.drop(columns=['student_id'])
            
    # String Normalization (Agar konsisten)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip().str.lower()
        
    # Drop Overall Score (Mencegah Target Leakage)
    if 'overall_score' in df.columns:
        df = df.drop(columns=['overall_score'])

    # SPLIT DATA
    print("Splitting Data...")
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ENCODING (Ordinal + OneHot)
    print("Encoding Data (Ordinal & Nominal)...")
    
    # Definisi Kolom
    ordinal_cols = ['parent_education', 'travel_time']
    nominal_cols = ['gender', 'school_type', 'internet_access', 'extra_activities', 'study_method']
    numeric_cols = [col for col in X_train.columns if col not in ordinal_cols + nominal_cols]

    # Definisi Urutan (Categories) untuk OrdinalEncoder
    ordinal_categories = [
        ['no formal', 'primary', 'high school', 'diploma', 'graduate', 'post graduate', 'phd'], # parent_education
        ['<15 min', '15-30 min', '30-60 min', '>60 min']                                      # travel_time
    ]

    # Ordinal: Ordinal Encoding
    ordinal_transformer = OrdinalEncoder(categories=ordinal_categories, handle_unknown='use_encoded_value', unknown_value=-1)
    
    # Nominal: One-Hot Encoding
    onehot_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=int)

    # Gabung dalam ColumnTransformer
    preprocessor_encode = ColumnTransformer(
        transformers=[
            ('ord', ordinal_transformer, ordinal_cols),
            ('cat', onehot_transformer, nominal_cols)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    
    # Fit & Transform
    X_train_encoded = preprocessor_encode.fit_transform(X_train)
    X_test_encoded = preprocessor_encode.transform(X_test)
    
    feature_names = preprocessor_encode.get_feature_names_out()
    
    X_train = pd.DataFrame(X_train_encoded, columns=feature_names, index=X_train.index)
    X_test = pd.DataFrame(X_test_encoded, columns=feature_names, index=X_test.index)

    # OUTLIER HANDLING (Manual Capping)
    print("Handling Outliers...")
    outlier_cols = ['study_hours', 'attendance_percentage']
    outlier_params = {}

    for col in outlier_cols:
        if col in X_train.columns:
            Q1 = X_train[col].quantile(0.25)
            Q3 = X_train[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            
            outlier_params[col] = {'lower': lower, 'upper': upper}
            
            X_train[col] = X_train[col].clip(lower, upper)
            X_test[col] = X_test[col].clip(lower, upper)

    # SCALING (StandardScaler)
    print("Scaling Features...")
    scaler = StandardScaler()
    
    # Scale semua fitur (hasil encode + numerik asli)
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    # Encode Target
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # SIMPAN HASIL
    os.makedirs(output_dir, exist_ok=True)
    
    train_final = pd.concat([X_train_scaled, pd.Series(y_train_enc, name=target_column, index=X_train.index)], axis=1)
    test_final = pd.concat([X_test_scaled, pd.Series(y_test_enc, name=target_column, index=X_test.index)], axis=1)
    
    train_final.to_csv(f"{output_dir}/train.csv", index=False)
    test_final.to_csv(f"{output_dir}/test.csv", index=False)
    
    print(f"Data Saved: {output_dir}/train.csv & test.csv")

    if save_scalers:
        artifacts = {
            'encoder_step': preprocessor_encode, 
            'scaler_step': scaler,               
            'label_encoder': le,
            'outlier_params': outlier_params,
            'feature_names': feature_names
        }
        joblib.dump(artifacts, f"{output_dir}/preprocessing_artifacts.joblib")
        print(f"Artifacts Saved: {output_dir}/preprocessing_artifacts.joblib")

if __name__ == "__main__":
    RAW_PATH = 'student_performance_raw/Student_Performance.csv'
    OUT_DIR = 'preprocessing/student_performance_preprocessing'
    
    perform_preprocessing(RAW_PATH, OUT_DIR)