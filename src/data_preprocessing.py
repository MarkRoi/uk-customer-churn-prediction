
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
        self.feature_names = None
        
        # 🚨 CRITICAL FIX: ADD ALL DATA LEAKAGE COLUMNS
        self.columns_to_drop = [
            'customer_id', 
            'products_held', 
            'churn_reason',
            'churn_probability',     # Generation formula
            'days_since_churn',      # Future information
            'estimated_clv',         # Might correlate with churn
            'account_age_days',      # Redundant with tenure_months
            'random_noise_1',        # Synthetic noise
            'random_noise_2'         # Synthetic noise
        ]
        
    def preprocess(self, df, training=True):
        """Main preprocessing pipeline"""
        
        # Create a copy
        df_processed = df.copy()
        
        # 🚨 FIRST: Remove data leakage columns
        existing_columns = [col for col in self.columns_to_drop if col in df_processed.columns]
        df_processed = df_processed.drop(columns=existing_columns, errors='ignore')
        
        # Handle missing values
        df_processed = self.handle_missing_values(df_processed)
        
        # Feature engineering
        df_processed = self.create_features(df_processed)
        
        # Encode categorical variables
        df_processed = self.encode_categorical(df_processed, training=training)
        
        # Separate features and target
        if 'churned' in df_processed.columns:
            X = df_processed.drop('churned', axis=1)
            y = df_processed['churned']
        else:
            X = df_processed
            y = None

        if not training:
            X = self.align_features(X)
          
        return X, y
    
    def handle_missing_values(self, df):
        """Handle missing values"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])
                
        return df
    
    def create_features(self, df):
        """Create additional features - FIXED for XGBoost"""
        
        # Safe interaction features
        df['income_per_product'] = df['annual_income'] / (df['num_products'] + 1)
        
        # Safe engagement score
        df['engagement_score'] = np.where(
            df['days_since_last_login'] > 0,
            df['app_usage_hours'] / df['days_since_last_login'],
            0
        )
        
        # Safe risk features
        df['risk_score'] = (df['complaints_last_year'] * 0.3 + 
                        (850 - df['credit_score']) / 550 * 0.7)
        
        # Behavioral features
        df['total_monthly_value'] = df['avg_transaction_value'] * df['transaction_frequency'] / 30
        
        # Age groups - FIXED: Use labels without special characters
        df['age_group'] = pd.cut(df['age'], 
                                bins=[0, 25, 35, 45, 55, 65, 100],
                                labels=['18_25', '26_35', '36_45', '46_55', '56_65', '65_plus'])  # Use underscore
        
        # Tenure groups - FIXED: Use labels without special characters
        df['tenure_group'] = pd.cut(df['tenure_months'],
                                bins=[0, 12, 36, 60, 120, 240],
                                labels=['less_1yr', '1_3yr', '3_5yr', '5_10yr', '10_plus_yr'])  # Use underscore
        
        # Polynomial features
        df['credit_score_squared'] = df['credit_score'] ** 2
        df['log_income'] = np.log1p(df['annual_income'])
        
        return df
    
    def encode_categorical(self, df, training=True):
        """Encode categorical variables - FIXED for XGBoost compatibility"""
        
        # Label encode binary categorical
        binary_cols = ['gender']
        for col in binary_cols:
            if col in df.columns:
                if training:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
                        df[col] = le.transform(df[col])
                    else:
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col])
                        self.label_encoders[col] = le
        
        # One-hot encode region - FIXED for XGBoost
        if 'region' in df.columns:
            if training:
                # Use clean column names without special characters
                region_dummies = pd.get_dummies(df['region'], prefix='region')
                # Clean column names for XGBoost compatibility
                region_dummies.columns = [col.replace(' ', '_').replace('-', '_').replace('[', '').replace(']', '').replace('<', '')
                                        for col in region_dummies.columns]
                self.region_columns = region_dummies.columns.tolist()
                df = pd.concat([df, region_dummies], axis=1)
            else:
                region_dummies = pd.get_dummies(df['region'], prefix='region')
                # Clean column names
                region_dummies.columns = [col.replace(' ', '_').replace('-', '_').replace('[', '').replace(']', '').replace('<', '')
                                        for col in region_dummies.columns]
                for col in self.region_columns:
                    if col not in region_dummies.columns:
                        region_dummies[col] = 0
                region_dummies = region_dummies[self.region_columns]
                df = pd.concat([df, region_dummies], axis=1)
            df = df.drop('region', axis=1)
        
        # One-hot encode age and tenure groups - FIXED
        for col in ['age_group', 'tenure_group']:
            if col in df.columns:
                if training:
                    dummies = pd.get_dummies(df[col], prefix=col)
                    # Clean column names
                    dummies.columns = [col_name.replace(' ', '_').replace('-', '_').replace('[', '').replace(']', '').replace('<', '')
                                    for col_name in dummies.columns]
                    setattr(self, f'{col}_columns', dummies.columns.tolist())
                else:
                    dummies = pd.get_dummies(df[col], prefix=col)
                    # Clean column names
                    dummies.columns = [col_name.replace(' ', '_').replace('-', '_').replace('[', '').replace(']', '').replace('<', '')
                                    for col_name in dummies.columns]
                    expected_columns = getattr(self, f'{col}_columns', [])
                    for exp_col in expected_columns:
                        if exp_col not in dummies.columns:
                            dummies[exp_col] = 0
                    if expected_columns:
                        dummies = dummies[expected_columns]
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(col, axis=1)
        
        # Additional cleaning of all column names
        df.columns = [str(col).replace('[', '').replace(']', '').replace('<', '').replace(' ', '_').replace('-', '_')
                    for col in df.columns]
        
        return df
    
    def align_features(self, X):
        """Align features during inference to match training features"""
        if self.feature_names is None:
            return X
        
        # Add missing columns
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        
        # Remove extra columns
        X = X[self.feature_names]
        
        return X
    
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
        
        # Save feature names before scaling
        self.feature_names = X_train.columns.tolist()
        
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
            'feature_names': self.feature_names
        }
    
    def save_preprocessor(self, path='models/preprocessor.joblib'):
        """Save preprocessor objects"""
        preprocessor_obj = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'region_columns': getattr(self, 'region_columns', []),
            'age_group_columns': getattr(self, 'age_group_columns', []),
            'tenure_group_columns': getattr(self, 'tenure_group_columns', []),
            'columns_to_drop': self.columns_to_drop  # Save this too
        }
        joblib.dump(preprocessor_obj, path)
        print(f"Preprocessor saved to {path}")
    
    def load_preprocessor(self, path='models/preprocessor.joblib'):
        """Load preprocessor objects"""
        preprocessor_obj = joblib.load(path)
        self.scaler = preprocessor_obj['scaler']
        self.label_encoders = preprocessor_obj['label_encoders']
        self.feature_names = preprocessor_obj['feature_names']
        self.region_columns = preprocessor_obj.get('region_columns', [])
        self.age_group_columns = preprocessor_obj.get('age_group_columns', [])
        self.tenure_group_columns = preprocessor_obj.get('tenure_group_columns', [])
        self.columns_to_drop = preprocessor_obj.get('columns_to_drop', 
            ['customer_id', 'products_held', 'churn_reason', 'churn_probability', 'days_since_churn'])