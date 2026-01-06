import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.data_preprocessing import DataPreprocessor
from src.model import ChurnPredictor
from src.evaluation import ModelEvaluator
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("=" * 60)
    print("UK CUSTOMER CHURN PREDICTION - COMPLETE PIPELINE")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n1. Loading data...")
    try:
        df = pd.read_csv('data/uk_customers.csv')
        print(f"   Loaded {len(df)} customer records")
        print(f"   Churn rate: {df['churned'].mean():.2%}")
    except FileNotFoundError:
        print("   Error: Data file not found.")
        print("   Run generate_synthetic_data.py first.")
        return
    
    # Step 2: Preprocess data
    print("\n2. Preprocessing data...")
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess(df, training=True)
    
    # Split data
    print("   Splitting data into train/validation/test sets...")
    data_splits = preprocessor.split_data(X, y, test_size=0.2, val_size=0.1)
    
    preprocessor.feature_names = data_splits['feature_names']

    # Save preprocessor
    preprocessor.save_preprocessor('models/preprocessor.joblib')
    
    # Step 3: Train models
    print("\n3. Training models...")
    predictor = ChurnPredictor()
    
    # Train all models
    results = predictor.train_models(
        data_splits['X_train'], data_splits['y_train'],
        data_splits['X_val'], data_splits['y_val']
    )
    
    # Select best model
    best_model = predictor.select_best_model(results)
    
    # Step 4: Hyperparameter tuning
    print("\n4. Performing hyperparameter tuning...")
    tuned_model = predictor.hyperparameter_tuning(
        data_splits['X_train'], data_splits['y_train'],
        model_name=predictor.best_model_name
    )
    
    # Step 5: Evaluate on test set
    print("\n5. Evaluating on test set...")
    evaluation = predictor.evaluate_model(tuned_model, 
                                         data_splits['X_test'], 
                                         data_splits['y_test'])
    
    print(f"\n   Test Set Performance:")
    print(f"   ROC-AUC: {evaluation['metrics']['roc_auc']:.4f}")
    print(f"   F1-Score: {evaluation['metrics']['f1']:.4f}")
    print(f"   Precision: {evaluation['metrics']['precision']:.4f}")
    print(f"   Recall: {evaluation['metrics']['recall']:.4f}")
    
    # Step 6: Save model
    predictor.save_model(tuned_model, 'models/best_churn_model.joblib')
    
    # Step 7: Visualizations
    print("\n6. Generating visualizations...")
    evaluator = ModelEvaluator()
    
    # Plot ROC curves for all models
    evaluator.plot_roc_curves(
        {name: results[name]['model'] for name in results},
        data_splits['X_test'], data_splits['y_test']
    )
    
    # Plot confusion matrix for best model
    y_pred = tuned_model.predict(data_splits['X_test'])
    evaluator.plot_confusion_matrix(
        data_splits['y_test'], y_pred,
        predictor.best_model_name
    )
    
    # Plot feature importance
    if hasattr(tuned_model, 'feature_importances_'):
        evaluator.plot_feature_importance(
            tuned_model, data_splits['feature_names']
        )
    
    # SHAP analysis
    try:
        evaluator.plot_shap_summary(
            tuned_model, data_splits['X_test'][:100],
            feature_names=data_splits['feature_names']
        )
    except:
        print("   Skipping SHAP analysis (requires TreeExplainer)")
    
    # Step 8: Business Insights
    print("\n7. Generating business insights...")
    generate_business_insights(df, tuned_model, preprocessor, data_splits)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print("\nOutputs generated:")
    print("✓ Trained model: models/best_churn_model.joblib")
    print("✓ Preprocessor: models/preprocessor.joblib")
    print("✓ Visualizations: reports/ folder")
    print("\nNext steps:")
    print("1. Run streamlit_app.py for interactive dashboard")
    print("2. Check notebooks/ for detailed analysis")

def generate_business_insights(df, model, preprocessor, data_splits):
    """Generate actionable business insights"""
    
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': data_splits['feature_names'],
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n   Top 5 Churn Drivers:")
        for i, (_, row) in enumerate(feature_importance.head(5).iterrows(), 1):
            print(f"   {i}. {row['feature']}: {row['importance']:.3f}")
    
    # Calculate potential savings
    total_customers = len(df)
    churn_rate = df['churned'].mean()
    avg_clv = df['estimated_clv'].mean()
    
    potential_savings = total_customers * churn_rate * avg_clv * 0.3  # Assume 30% retention
    
    print(f"\n   Business Impact Analysis:")
    print(f"   Total customers: {total_customers:,}")
    print(f"   Current churn rate: {churn_rate:.2%}")
    print(f"   Average CLV: £{avg_clv:,.0f}")
    print(f"   Potential annual savings: £{potential_savings:,.0f}")
    
    # Regional analysis
    if 'region' in df.columns:
        regional_churn = df.groupby('region')['churned'].mean().sort_values(ascending=False)
        print(f"\n   Regional Churn Rates:")
        for region, rate in regional_churn.head(3).items():
            print(f"   {region}: {rate:.2%}")
    
    # Customer segmentation insights
    high_risk_profile = df[
        (df['days_since_last_login'] > 30) & 
        (df['complaints_last_year'] > 0) &
        (df['credit_score'] < 600)
    ]
    
    print(f"\n   High-risk customers identified: {len(high_risk_profile):,}")
    print(f"   High-risk churn rate: {high_risk_profile['churned'].mean():.2%}")

if __name__ == "__main__":
    main()