"""
Machine learning models for customer churn prediction.
Implements multiple algorithms with hyperparameter optimization.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import optuna
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')

class ChurnPredictor:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.feature_importance = None
        self.shap_explainer = None
        
    def initialize_models(self):
        """Initialize different ML models"""
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'xgboost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'lightgbm': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'svm': SVC(random_state=42, probability=True)
        }
    
    def handle_imbalanced_data(self, X_train, y_train):
        """Handle class imbalance using SMOTE"""
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        return X_resampled, y_resampled
    
    def optimize_hyperparameters(self, model_name, X_train, y_train, n_trials=50):
        """Optimize hyperparameters using Optuna"""
        def objective(trial):
            if model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'random_state': 42
                }
                model = RandomForestClassifier(**params)
                
            elif model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': 42,
                    'eval_metric': 'logloss'
                }
                model = xgb.XGBClassifier(**params)
                
            elif model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                    'random_state': 42,
                    'verbose': -1
                }
                model = lgb.LGBMClassifier(**params)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            return cv_scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params
    
    def train_models(self, X_train, y_train, X_test, y_test, optimize=True):
        """Train all models and evaluate performance"""
        self.initialize_models()
        
        # Handle imbalanced data
        X_train_balanced, y_train_balanced = self.handle_imbalanced_data(X_train, y_train)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Optimize hyperparameters for key models
            if optimize and name in ['random_forest', 'xgboost', 'lightgbm']:
                print(f"Optimizing hyperparameters for {name}...")
                best_params = self.optimize_hyperparameters(name, X_train_balanced, y_train_balanced)
                
                # Update model with best parameters
                if name == 'random_forest':
                    model = RandomForestClassifier(**best_params)
                elif name == 'xgboost':
                    model = xgb.XGBClassifier(**best_params)
                elif name == 'lightgbm':
                    model = lgb.LGBMClassifier(**best_params)
                
                self.models[name] = model
            
            # Train model
            model.fit(X_train_balanced, y_train_balanced)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, 
                                      cv=5, scoring='roc_auc')
            
            results[name] = {
                'model': model,
                'auc_score': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"{name} - AUC: {auc_score:.4f}, CV: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\nBest model: {best_model_name} with AUC: {results[best_model_name]['auc_score']:.4f}")
        
        return results
    
    def analyze_feature_importance(self, X_train, feature_names):
        """Analyze feature importance using SHAP"""
        if self.best_model is None:
            raise ValueError("No trained model found. Train models first.")
        
        # Get feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            self.feature_importance = importance_df
        
        # SHAP analysis
        if self.best_model_name in ['xgboost', 'lightgbm', 'random_forest']:
            self.shap_explainer = shap.TreeExplainer(self.best_model)
            shap_values = self.shap_explainer.shap_values(X_train.sample(1000))  # Sample for speed
            
            return {
                'feature_importance': self.feature_importance,
                'shap_values': shap_values,
                'shap_explainer': self.shap_explainer
            }
        
        return {'feature_importance': self.feature_importance}
    
    def plot_model_comparison(self, results):
        """Plot model performance comparison"""
        model_names = list(results.keys())
        auc_scores = [results[name]['auc_score'] for name in model_names]
        cv_means = [results[name]['cv_mean'] for name in model_names]
        cv_stds = [results[name]['cv_std'] for name in model_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # AUC scores
        bars1 = ax1.bar(model_names, auc_scores, color='skyblue', alpha=0.7)
        ax1.set_title('Model Performance - AUC Score')
        ax1.set_ylabel('AUC Score')
        ax1.set_ylim(0.5, 1.0)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, score in zip(bars1, auc_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Cross-validation scores with error bars
        bars2 = ax2.bar(model_names, cv_means, yerr=cv_stds, 
                       color='lightcoral', alpha=0.7, capsize=5)
        ax2.set_title('Cross-Validation Performance')
        ax2.set_ylabel('CV AUC Score')
        ax2.set_ylim(0.5, 1.0)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, score in zip(bars2, cv_means):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('/q/bin/customer_churn_project/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        if self.feature_importance is None:
            print("Feature importance not calculated. Run analyze_feature_importance first.")
            return
        
        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance.head(15)
        
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title(f'Top 15 Feature Importance - {self.best_model_name}')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('/q/bin/customer_churn_project/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_predictions(self, X_test, customer_ids=None):
        """Generate predictions with probabilities"""
        if self.best_model is None:
            raise ValueError("No trained model found. Train models first.")
        
        predictions = self.best_model.predict(X_test)
        probabilities = self.best_model.predict_proba(X_test)[:, 1]
        
        results_df = pd.DataFrame({
            'customer_id': customer_ids if customer_ids is not None else range(len(predictions)),
            'churn_prediction': predictions,
            'churn_probability': probabilities,
            'risk_category': pd.cut(probabilities, 
                                  bins=[0, 0.3, 0.7, 1.0], 
                                  labels=['Low', 'Medium', 'High'])
        })
        
        return results_df
    
    def save_model(self, filepath='/q/bin/customer_churn_project/best_churn_model.pkl'):
        """Save the best trained model"""
        if self.best_model is None:
            raise ValueError("No trained model found. Train models first.")
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'feature_importance': self.feature_importance,
            'shap_explainer': self.shap_explainer
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='/q/bin/customer_churn_project/best_churn_model.pkl'):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']
        self.feature_importance = model_data.get('feature_importance')
        self.shap_explainer = model_data.get('shap_explainer')
        print(f"Model loaded from {filepath}")

if __name__ == "__main__":
    # Load processed data
    from data_preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    data_dict = preprocessor.process_data()
    
    # Train models
    predictor = ChurnPredictor()
    results = predictor.train_models(
        data_dict['X_train'], data_dict['y_train'],
        data_dict['X_test'], data_dict['y_test'],
        optimize=True
    )
    
    # Analyze feature importance
    importance_analysis = predictor.analyze_feature_importance(
        data_dict['X_train'], data_dict['feature_names']
    )
    
    # Plot results
    predictor.plot_model_comparison(results)
    predictor.plot_feature_importance()
    
    # Save model
    predictor.save_model()
    
    print("Model training complete!")
