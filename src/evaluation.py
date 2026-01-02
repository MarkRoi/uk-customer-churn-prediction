import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                           confusion_matrix, classification_report)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap

class ModelEvaluator:
    def __init__(self):
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
    def plot_roc_curves(self, models_results, X_test, y_test, figsize=(12, 8)):
        """Plot ROC curves for multiple models"""
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for (name, model), color in zip(models_results.items(), self.colors):
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, color=color, lw=2,
                       label=f'{name} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reports/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_confusion_matrix(self, y_true, y_pred, model_name, figsize=(10, 8)):
        """Plot confusion matrix"""
        
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot regular confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   cbar_kws={'label': 'Count'})
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        ax1.set_title(f'Confusion Matrix - {model_name}')
        
        # Plot normalized confusion matrix
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Reds', ax=ax2,
                   cbar_kws={'label': 'Percentage'})
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        ax2.set_title(f'Normalized Confusion Matrix - {model_name}')
        
        plt.tight_layout()
        plt.savefig(f'reports/confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_feature_importance(self, model, feature_names, top_n=20, figsize=(12, 8)):
        """Plot feature importance"""
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            
            fig, ax = plt.subplots(figsize=figsize)
            
            ax.barh(range(top_n), importances[indices][::-1], color=self.colors[0])
            ax.set_yticks(range(top_n))
            
            # Use feature names if available
            if feature_names is not None and len(feature_names) == len(importances):
                feature_labels = [feature_names[i] for i in indices][::-1]
            else:
                feature_labels = [f'Feature {i}' for i in indices][::-1]
                
            ax.set_yticklabels(feature_labels, fontsize=10)
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
            ax.grid(True, axis='x', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('reports/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        else:
            print("Model doesn't have feature_importances_ attribute")
            
    def plot_shap_summary(self, model, X, feature_names=None, max_display=20):
        """Generate SHAP summary plot"""
        
        try:
            # Create SHAP explainer
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                
                # For binary classification
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
                fig, ax = plt.subplots(figsize=(12, 8))
                shap.summary_plot(shap_values, X, feature_names=feature_names,
                                 max_display=max_display, show=False)
                
                plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig('reports/shap_summary.png', dpi=300, bbox_inches='tight')
                plt.show()
                
        except Exception as e:
            print(f"Could not generate SHAP plot: {e}")
            
    def create_interactive_dashboard(self, model_results, feature_importance):
        """Create interactive Plotly dashboard"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Performance Comparison', 
                          'Feature Importance',
                          'Confusion Matrix Heatmap',
                          'Precision-Recall Curve'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'heatmap'}, {'type': 'scatter'}]]
        )
        
        # Model comparison bar chart
        model_names = list(model_results.keys())
        auc_scores = [results['roc_auc'] for results in model_results.values()]
        
        fig.add_trace(
            go.Bar(x=model_names, y=auc_scores, name='ROC-AUC',
                  marker_color=self.colors[0]),
            row=1, col=1
        )
        
        # Feature importance
        top_features = feature_importance.head(10)
        fig.add_trace(
            go.Bar(x=top_features['importance'], y=top_features['feature'],
                  orientation='h', name='Feature Importance',
                  marker_color=self.colors[1]),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="UK Customer Churn Prediction Dashboard",
            showlegend=True,
            height=800
        )
        
        fig.write_html("reports/interactive_dashboard.html")
        print("Interactive dashboard saved to reports/interactive_dashboard.html")