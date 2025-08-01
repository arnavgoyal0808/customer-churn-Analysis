"""
Data preprocessing module for customer churn prediction project.
Handles data cleaning, feature engineering, and dataset preparation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def generate_sample_data(self, n_samples=10000):
        """Generate synthetic customer data for demonstration"""
        np.random.seed(42)
        
        # Customer demographics
        customer_ids = [f"CUST_{i:06d}" for i in range(1, n_samples + 1)]
        ages = np.random.normal(40, 15, n_samples).clip(18, 80)
        genders = np.random.choice(['M', 'F'], n_samples)
        
        # Account information
        tenure_months = np.random.exponential(24, n_samples).clip(1, 120)
        monthly_charges = np.random.normal(65, 25, n_samples).clip(20, 150)
        total_charges = monthly_charges * tenure_months + np.random.normal(0, 100, n_samples)
        
        # Service usage
        contract_types = np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                        n_samples, p=[0.5, 0.3, 0.2])
        payment_methods = np.random.choice(['Electronic check', 'Mailed check', 
                                          'Bank transfer', 'Credit card'], n_samples)
        
        # Service features
        internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], 
                                          n_samples, p=[0.4, 0.4, 0.2])
        online_security = np.random.choice(['Yes', 'No', 'No internet service'], n_samples)
        tech_support = np.random.choice(['Yes', 'No', 'No internet service'], n_samples)
        
        # Customer behavior
        support_calls = np.random.poisson(2, n_samples)
        late_payments = np.random.poisson(1, n_samples)
        
        # Calculate churn probability based on features
        churn_prob = (
            0.3 * (contract_types == 'Month-to-month') +
            0.2 * (payment_methods == 'Electronic check') +
            0.15 * (support_calls > 3) +
            0.1 * (late_payments > 2) +
            0.1 * (tenure_months < 12) +
            0.05 * (monthly_charges > 80) +
            np.random.normal(0, 0.1, n_samples)
        ).clip(0, 1)
        
        churned = np.random.binomial(1, churn_prob, n_samples)
        
        # Create DataFrame
        data = pd.DataFrame({
            'customer_id': customer_ids,
            'age': ages.astype(int),
            'gender': genders,
            'tenure_months': tenure_months.round(1),
            'monthly_charges': monthly_charges.round(2),
            'total_charges': total_charges.round(2),
            'contract_type': contract_types,
            'payment_method': payment_methods,
            'internet_service': internet_service,
            'online_security': online_security,
            'tech_support': tech_support,
            'support_calls': support_calls,
            'late_payments': late_payments,
            'churned': churned
        })
        
        return data
    
    def clean_data(self, df):
        """Clean and prepare the dataset"""
        df_clean = df.copy()
        
        # Handle missing values
        df_clean['total_charges'] = pd.to_numeric(df_clean['total_charges'], errors='coerce')
        df_clean['total_charges'].fillna(df_clean['monthly_charges'] * df_clean['tenure_months'], inplace=True)
        
        # Remove outliers using IQR method
        numeric_columns = ['age', 'tenure_months', 'monthly_charges', 'total_charges']
        for col in numeric_columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        return df_clean
    
    def engineer_features(self, df):
        """Create additional features for better prediction"""
        df_features = df.copy()
        
        # Financial features
        df_features['avg_monthly_spend'] = df_features['total_charges'] / df_features['tenure_months']
        df_features['charges_per_month_ratio'] = df_features['monthly_charges'] / df_features['avg_monthly_spend']
        
        # Tenure categories
        df_features['tenure_category'] = pd.cut(df_features['tenure_months'], 
                                              bins=[0, 12, 24, 48, float('inf')],
                                              labels=['New', 'Medium', 'Long', 'Veteran'])
        
        # Service complexity score
        service_features = ['online_security', 'tech_support']
        df_features['service_complexity'] = 0
        for feature in service_features:
            df_features['service_complexity'] += (df_features[feature] == 'Yes').astype(int)
        
        # Risk indicators
        df_features['high_risk'] = (
            (df_features['contract_type'] == 'Month-to-month') &
            (df_features['payment_method'] == 'Electronic check') &
            (df_features['support_calls'] > 2)
        ).astype(int)
        
        # Customer value segments
        df_features['customer_value'] = pd.qcut(df_features['total_charges'], 
                                              q=4, labels=['Low', 'Medium', 'High', 'Premium'])
        
        return df_features
    
    def encode_categorical_features(self, df, fit=True):
        """Encode categorical variables"""
        df_encoded = df.copy()
        categorical_columns = df.select_dtypes(include=['object']).columns
        categorical_columns = [col for col in categorical_columns if col != 'customer_id']
        
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
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale numerical features
        numerical_columns = X.select_dtypes(include=[np.number]).columns
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numerical_columns] = self.scaler.fit_transform(X_train[numerical_columns])
        X_test_scaled[numerical_columns] = self.scaler.transform(X_test[numerical_columns])
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def process_data(self, df=None):
        """Complete data processing pipeline"""
        if df is None:
            print("Generating sample data...")
            df = self.generate_sample_data()
        
        print("Cleaning data...")
        df_clean = self.clean_data(df)
        
        print("Engineering features...")
        df_features = self.engineer_features(df_clean)
        
        print("Encoding categorical features...")
        df_encoded = self.encode_categorical_features(df_features, fit=True)
        
        print("Preparing train/test sets...")
        X_train, X_test, y_train, y_test = self.prepare_features(df_encoded)
        
        # Save processed data
        df_encoded.to_csv('/q/bin/customer_churn_project/data/processed/processed_data.csv', index=False)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'full_data': df_encoded,
            'feature_names': X_train.columns.tolist()
        }

if __name__ == "__main__":
    # Create directories
    import os
    os.makedirs('/q/bin/customer_churn_project/data/raw', exist_ok=True)
    os.makedirs('/q/bin/customer_churn_project/data/processed', exist_ok=True)
    
    # Process data
    preprocessor = DataPreprocessor()
    data_dict = preprocessor.process_data()
    
    print(f"Data processing complete!")
    print(f"Training set shape: {data_dict['X_train'].shape}")
    print(f"Test set shape: {data_dict['X_test'].shape}")
    print(f"Churn rate: {data_dict['y_train'].mean():.2%}")
