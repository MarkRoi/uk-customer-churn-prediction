import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.onehot_encoder = None
        self.columns_to_drop = ['customer_id', 'products_held', 'churn_reason']
        
    def preprocess(self, df):
        """Main preprocessing pipeline"""
        
        # Create a copy
        df_processed = df.copy()
        
        # Drop unnecessary columns
        df_processed = df_processed.drop(columns=self.columns_to_drop, errors='ignore')
        
        # Handle missing values
        df_processed = self.handle_missing_values(df_processed)
        
        # Feature engineering
        df_processed = self.create_features(df_processed)
        
        # Encode categorical variables
        df_processed = self.encode_categorical(df_processed)
        
        # Separate features and target
        if 'churned' in df_processed.columns:
            X = df_processed.drop('churned', axis=1)
            y = df_processed['churned']
        else:
            X = df_processed
            y = None
            
        return X, y
    
    def handle_missing_values(self, df):
        """Handle missing values"""
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])
                
        return df
    
    def create_features(self, df):
        """Create additional features"""
        
        # Interaction features
        df['income_per_product'] = df['annual_income'] / (df['num_products'] + 1)
        df['engagement_score'] = df['app_usage_hours'] / (df['days_since_last_login'] + 1)
        
        # Risk features
        df['risk_score'] = (df['complaints_last_year'] * 0.3 + 
                           (850 - df['credit_score']) / 550 * 0.7)
        
        # Behavioral features
        df['total_monthly_value'] = df['avg_transaction_value'] * df['transaction_frequency'] / 30
        
        # Age groups
        df['age_group'] = pd.cut(df['age'], 
                                bins=[0, 25, 35, 45, 55, 65, 100],
                                labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
        
        # Tenure groups
        df['tenure_group'] = pd.cut(df['tenure_months'],
                                   bins=[0, 12, 36, 60, 120, 240],
                                   labels=['<1yr', '1-3yr', '3-5yr', '5-10yr', '10+yr'])
        
        return df
    
    def encode_categorical(self, df):
        """Encode categorical variables"""
        
        # Label encode binary categorical
        binary_cols = ['gender']
        for col in binary_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        
        # One-hot encode region
        if 'region' in df.columns:
            region_dummies = pd.get_dummies(df['region'], prefix='region')
            df = pd.concat([df, region_dummies], axis=1)
            df = df.drop('region', axis=1)
        
        # One-hot encode age and tenure groups
        for col in ['age_group', 'tenure_group']:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(col, axis=1)
        
        return df
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        """Split data into train, validation, and test sets"""
        
        # First split: train+val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train and validation
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio, 
            random_state=random_state, stratify=y_train_val
        )
        
        # Scale numerical features
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
        X_val_scaled[numeric_cols] = self.scaler.transform(X_val[numeric_cols])
        X_test_scaled[numeric_cols] = self.scaler.transform(X_test[numeric_cols])
        
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': X_train.columns.tolist()
        }
    
    def save_preprocessor(self, path='models/preprocessor.joblib'):
        """Save preprocessor objects"""
        preprocessor_obj = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }
        joblib.dump(preprocessor_obj, path)
    
    def load_preprocessor(self, path='models/preprocessor.joblib'):
        """Load preprocessor objects"""
        preprocessor_obj = joblib.load(path)
        self.scaler = preprocessor_obj['scaler']
        self.label_encoders = preprocessor_obj['label_encoders']