import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report)
import joblib
import warnings
warnings.filterwarnings('ignore')

class ChurnPredictor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
        
    def initialize_models(self):
        """Initialize multiple models with base parameters"""
        
        self.models = {
            'logistic': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            'xgboost': XGBClassifier(
                n_estimators=100,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=self.calculate_scale_pos_weight
            ),
            'lightgbm': LGBMClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced',
                verbosity=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state
            )
        }
        
    def calculate_scale_pos_weight(self, y_train):
        """Calculate scale_pos_weight for XGBoost"""
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        neg = np.sum(y_train == 0)
        pos = np.sum(y_train == 1)
        return neg / pos if pos > 0 else 1
    
    def train_models(self, X_train, y_train, X_val=None, y_val=None):
        """Train all models"""
        
        self.initialize_models()
        
        # Calculate class weight for XGBoost
        scale_pos_weight = self.calculate_scale_pos_weight(y_train)
        self.models['xgboost'].set_params(scale_pos_weight=scale_pos_weight)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Make predictions
            if X_val is not None and y_val is not None:
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                
                # Calculate metrics
                results[name] = {
                    'accuracy': accuracy_score(y_val, y_pred),
                    'precision': precision_score(y_val, y_pred),
                    'recall': recall_score(y_val, y_pred),
                    'f1': f1_score(y_val, y_pred),
                    'roc_auc': roc_auc_score(y_val, y_pred_proba),
                    'model': model
                }
                
                print(f"  {name} - AUC: {results[name]['roc_auc']:.4f}, F1: {results[name]['f1']:.4f}")
        
        return results
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='xgboost'):
        """Perform hyperparameter tuning for selected model"""
        
        if model_name == 'xgboost':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            model = XGBClassifier(
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=self.calculate_scale_pos_weight(y_train)
            )
        
        elif model_name == 'lightgbm':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'num_leaves': [31, 50, 100],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            model = LGBMClassifier(
                random_state=self.random_state,
                class_weight='balanced',
                verbosity=-1
            )
        
        elif model_name == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
            model = RandomForestClassifier(
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
        
        # Randomized search
        random_search = RandomizedSearchCV(
            model, param_grid, n_iter=20,
            cv=3, scoring='roc_auc',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )
        
        print(f"Performing hyperparameter tuning for {model_name}...")
        random_search.fit(X_train, y_train)
        
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best CV score: {random_search.best_score_:.4f}")
        
        return random_search.best_estimator_
    
    def evaluate_model(self, model, X_test, y_test):
        """Comprehensive model evaluation"""
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Feature importance if available
        if hasattr(model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(model.feature_importances_))],
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return {
            'metrics': metrics,
            'confusion_matrix': cm,
            'classification_report': report,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def select_best_model(self, results):
        """Select the best model based on ROC-AUC"""
        
        best_auc = -1
        for name, result in results.items():
            if result['roc_auc'] > best_auc:
                best_auc = result['roc_auc']
                self.best_model = result['model']
                self.best_model_name = name
        
        print(f"\nBest model: {self.best_model_name} with AUC: {best_auc:.4f}")
        return self.best_model
    
    def save_model(self, model, path='models/churn_model.joblib'):
        """Save trained model"""
        joblib.dump(model, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path='models/churn_model.joblib'):
        """Load trained model"""
        model = joblib.load(path)
        return model