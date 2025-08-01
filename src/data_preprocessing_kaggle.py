"""
Data preprocessing module for customer churn prediction project using Kaggle Telco dataset.
Handles data loading, cleaning, feature engineering, and dataset preparation.
"""

import pandas as pd
import numpy as np
import kagglehub
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class KaggleDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.dataset_path = None
        
    def download_kaggle_dataset(self):
        """Download the Kaggle Telco Customer Churn dataset"""
        print("Downloading Kaggle Telco Customer Churn dataset...")
        try:
            # Download latest version
            path = kagglehub.dataset_download("blastchar/telco-customer-churn")
            print(f"Path to dataset files: {path}")
            self.dataset_path = path
            return path
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please ensure you have Kaggle API credentials set up.")
            return None
    
    def load_kaggle_data(self):
        """Load the Kaggle Telco Customer Churn dataset"""
        if self.dataset_path is None:
            self.dataset_path = self.download_kaggle_dataset()
        
        if self.dataset_path is None:
            raise ValueError("Could not download dataset. Please check your Kaggle credentials.")
        
        # Find the CSV file in the downloaded path
        csv_files = [f for f in os.listdir(self.dataset_path) if f.endswith('.csv')]
        
        if not csv_files:
            raise ValueError("No CSV files found in the downloaded dataset.")
        
        # Load the first CSV file (should be WA_Fn-UseC_-Telco-Customer-Churn.csv)
        csv_file = csv_files[0]
        file_path = os.path.join(self.dataset_path, csv_file)
        
        print(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        
        print(f"Dataset loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        return df
    
    def clean_kaggle_data(self, df):
        """Clean and prepare the Kaggle dataset"""
        df_clean = df.copy()
        
        print("Cleaning Kaggle dataset...")
        
        # Handle the 'TotalCharges' column which might be stored as string
        if 'TotalCharges' in df_clean.columns:
            # Convert TotalCharges to numeric, handling empty strings
            df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
            
            # Fill missing TotalCharges with MonthlyCharges * tenure
            missing_mask = df_clean['TotalCharges'].isna()
            if missing_mask.any():
                print(f"Found {missing_mask.sum()} missing TotalCharges values, filling with estimated values...")
                df_clean.loc[missing_mask, 'TotalCharges'] = (
                    df_clean.loc[missing_mask, 'MonthlyCharges'] * df_clean.loc[missing_mask, 'tenure']
                )
        
        # Standardize column names to match our analysis
        column_mapping = {
            'customerID': 'customer_id',
            'gender': 'gender',
            'SeniorCitizen': 'senior_citizen',
            'Partner': 'partner',
            'Dependents': 'dependents',
            'tenure': 'tenure_months',
            'PhoneService': 'phone_service',
            'MultipleLines': 'multiple_lines',
            'InternetService': 'internet_service',
            'OnlineSecurity': 'online_security',
            'OnlineBackup': 'online_backup',
            'DeviceProtection': 'device_protection',
            'TechSupport': 'tech_support',
            'StreamingTV': 'streaming_tv',
            'StreamingMovies': 'streaming_movies',
            'Contract': 'contract_type',
            'PaperlessBilling': 'paperless_billing',
            'PaymentMethod': 'payment_method',
            'MonthlyCharges': 'monthly_charges',
            'TotalCharges': 'total_charges',
            'Churn': 'churned'
        }
        
        # Rename columns
        df_clean = df_clean.rename(columns=column_mapping)
        
        # Convert binary categorical variables to numeric
        binary_columns = ['partner', 'dependents', 'phone_service', 'paperless_billing']
        for col in binary_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].map({'Yes': 1, 'No': 0})
        
        # Convert senior citizen to binary if it's not already
        if 'senior_citizen' in df_clean.columns:
            df_clean['senior_citizen'] = df_clean['senior_citizen'].astype(int)
        
        # Convert churn target variable
        if 'churned' in df_clean.columns:
            df_clean['churned'] = df_clean['churned'].map({'Yes': 1, 'No': 0})
        
        # Handle 'No internet service' and 'No phone service' values
        internet_dependent_cols = ['online_security', 'online_backup', 'device_protection', 
                                 'tech_support', 'streaming_tv', 'streaming_movies']
        
        for col in internet_dependent_cols:
            if col in df_clean.columns:
                # Convert 'No internet service' to 'No' for consistency
                df_clean[col] = df_clean[col].replace('No internet service', 'No')
        
        if 'multiple_lines' in df_clean.columns:
            df_clean['multiple_lines'] = df_clean['multiple_lines'].replace('No phone service', 'No')
        
        print(f"Data cleaning completed. Final shape: {df_clean.shape}")
        return df_clean
    
    def engineer_kaggle_features(self, df):
        """Create additional features for better prediction using Kaggle data"""
        df_features = df.copy()
        
        print("Engineering features from Kaggle dataset...")
        
        # Financial features
        df_features['avg_monthly_spend'] = df_features['total_charges'] / df_features['tenure_months']
        df_features['avg_monthly_spend'] = df_features['avg_monthly_spend'].fillna(df_features['monthly_charges'])
        
        # Tenure categories
        df_features['tenure_category'] = pd.cut(df_features['tenure_months'], 
                                              bins=[0, 12, 24, 48, float('inf')],
                                              labels=['New', 'Medium', 'Long', 'Veteran'])
        
        # Service complexity score (count of additional services)
        service_features = ['online_security', 'online_backup', 'device_protection', 
                          'tech_support', 'streaming_tv', 'streaming_movies']
        df_features['service_complexity'] = 0
        
        for feature in service_features:
            if feature in df_features.columns:
                df_features['service_complexity'] += (df_features[feature] == 'Yes').astype(int)
        
        # Internet service type indicators
        if 'internet_service' in df_features.columns:
            df_features['has_fiber_optic'] = (df_features['internet_service'] == 'Fiber optic').astype(int)
            df_features['has_dsl'] = (df_features['internet_service'] == 'DSL').astype(int)
            df_features['no_internet'] = (df_features['internet_service'] == 'No').astype(int)
        
        # Contract type indicators
        if 'contract_type' in df_features.columns:
            df_features['month_to_month'] = (df_features['contract_type'] == 'Month-to-month').astype(int)
            df_features['one_year_contract'] = (df_features['contract_type'] == 'One year').astype(int)
            df_features['two_year_contract'] = (df_features['contract_type'] == 'Two year').astype(int)
        
        # Payment method risk indicators
        if 'payment_method' in df_features.columns:
            df_features['electronic_check'] = (df_features['payment_method'] == 'Electronic check').astype(int)
            df_features['automatic_payment'] = df_features['payment_method'].isin([
                'Bank transfer (automatic)', 'Credit card (automatic)'
            ]).astype(int)
        
        # Customer profile indicators
        df_features['family_customer'] = ((df_features.get('partner', 0) == 1) | 
                                        (df_features.get('dependents', 0) == 1)).astype(int)
        
        # High-risk profile
        df_features['high_risk_profile'] = (
            (df_features.get('month_to_month', 0) == 1) &
            (df_features.get('electronic_check', 0) == 1) &
            (df_features['tenure_months'] < 12)
        ).astype(int)
        
        # Customer value segments based on monthly charges
        df_features['customer_value'] = pd.qcut(df_features['monthly_charges'], 
                                              q=4, labels=['Low', 'Medium', 'High', 'Premium'],
                                              duplicates='drop')
        
        # Age-related features (if we have senior citizen info)
        if 'senior_citizen' in df_features.columns:
            df_features['is_senior'] = df_features['senior_citizen']
        
        print(f"Feature engineering completed. New shape: {df_features.shape}")
        return df_features
    
    def encode_categorical_features(self, df, fit=True):
        """Encode categorical variables"""
        df_encoded = df.copy()
        
        # Get categorical columns (excluding customer_id and binary encoded columns)
        categorical_columns = df.select_dtypes(include=['object']).columns
        categorical_columns = [col for col in categorical_columns 
                             if col not in ['customer_id'] and col in df.columns]
        
        print(f"Encoding categorical features: {categorical_columns}")
        
        for col in categorical_columns:
            if fit:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    le = self.label_encoders[col]
                    df_encoded[col] = df_encoded[col].astype(str)
                    mask = df_encoded[col].isin(le.classes_)
                    df_encoded.loc[mask, col] = le.transform(df_encoded.loc[mask, col])
                    df_encoded.loc[~mask, col] = -1  # Assign -1 to unseen categories
        
        return df_encoded
    
    def prepare_features(self, df, target_column='churned', test_size=0.2):
        """Prepare features for model training"""
        # Separate features and target
        feature_columns = [col for col in df.columns if col not in ['customer_id', target_column]]
        X = df[feature_columns]
        y = df[target_column]
        
        print(f"Preparing features: {len(feature_columns)} features, {len(y)} samples")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale numerical features
        numerical_columns = X.select_dtypes(include=[np.number]).columns
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        if len(numerical_columns) > 0:
            X_train_scaled[numerical_columns] = self.scaler.fit_transform(X_train[numerical_columns])
            X_test_scaled[numerical_columns] = self.scaler.transform(X_test[numerical_columns])
        
        print(f"Feature preparation completed:")
        print(f"  - Training set: {X_train_scaled.shape}")
        print(f"  - Test set: {X_test_scaled.shape}")
        print(f"  - Churn rate in training: {y_train.mean():.2%}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def process_kaggle_data(self):
        """Complete data processing pipeline for Kaggle dataset"""
        print("\n" + "="*60)
        print("KAGGLE TELCO DATASET PROCESSING PIPELINE")
        print("="*60)
        
        # Step 1: Load data
        df = self.load_kaggle_data()
        
        # Step 2: Clean data
        df_clean = self.clean_kaggle_data(df)
        
        # Step 3: Engineer features
        df_features = self.engineer_kaggle_features(df_clean)
        
        # Step 4: Encode categorical features
        df_encoded = self.encode_categorical_features(df_features, fit=True)
        
        # Step 5: Prepare train/test sets
        X_train, X_test, y_train, y_test = self.prepare_features(df_encoded)
        
        # Save processed data
        os.makedirs('data/processed', exist_ok=True)
        df_encoded.to_csv('data/processed/kaggle_processed_data.csv', index=False)
        
        print(f"\n✅ Kaggle data processing completed!")
        print(f"   - Original dataset shape: {df.shape}")
        print(f"   - Processed dataset shape: {df_encoded.shape}")
        print(f"   - Training samples: {X_train.shape[0]:,}")
        print(f"   - Test samples: {X_test.shape[0]:,}")
        print(f"   - Features: {X_train.shape[1]}")
        print(f"   - Churn rate: {y_train.mean():.2%}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'full_data': df_encoded,
            'feature_names': X_train.columns.tolist(),
            'original_data': df
        }

def main():
    """Test the Kaggle data preprocessing"""
    try:
        # Create directories
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        
        # Process Kaggle data
        preprocessor = KaggleDataPreprocessor()
        data_dict = preprocessor.process_kaggle_data()
        
        # Display sample of processed data
        print(f"\nSample of processed data:")
        print(data_dict['full_data'].head())
        
        print(f"\nFeature names:")
        for i, feature in enumerate(data_dict['feature_names']):
            print(f"{i+1:2d}. {feature}")
        
        print(f"\nData types:")
        print(data_dict['full_data'].dtypes.value_counts())
        
    except Exception as e:
        print(f"Error processing Kaggle data: {e}")
        print("Please ensure you have Kaggle API credentials set up.")
        print("You can set up credentials by:")
        print("1. Go to Kaggle.com -> Account -> API -> Create New API Token")
        print("2. Place the kaggle.json file in ~/.kaggle/")
        print("3. Run: chmod 600 ~/.kaggle/kaggle.json")

if __name__ == "__main__":
    main()
