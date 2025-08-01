"""
Main execution script for the Customer Churn Prediction and Financial Impact Analysis project.
Orchestrates the complete analysis pipeline from data preprocessing to retention strategies.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor
from churn_models import ChurnPredictor
from clv_analysis import CLVAnalyzer
from financial_impact import FinancialImpactAnalyzer
from retention_strategies import RetentionStrategyEngine

def create_project_directories():
    """Create necessary project directories"""
    directories = [
        'data/raw',
        'data/processed',
        'notebooks',
        'dashboard',
        'reports'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Project directories created successfully")

def run_data_preprocessing():
    """Run data preprocessing pipeline"""
    print("\n" + "="*60)
    print("STEP 1: DATA PREPROCESSING")
    print("="*60)
    
    preprocessor = DataPreprocessor()
    data_dict = preprocessor.process_data()
    
    print(f"✅ Data preprocessing completed")
    print(f"   - Training samples: {data_dict['X_train'].shape[0]:,}")
    print(f"   - Test samples: {data_dict['X_test'].shape[0]:,}")
    print(f"   - Features: {len(data_dict['feature_names'])}")
    print(f"   - Churn rate: {data_dict['y_train'].mean():.2%}")
    
    return data_dict

def run_churn_prediction(data_dict):
    """Run churn prediction models"""
    print("\n" + "="*60)
    print("STEP 2: CHURN PREDICTION MODELING")
    print("="*60)
    
    predictor = ChurnPredictor()
    
    # Train models
    print("Training multiple ML models...")
    results = predictor.train_models(
        data_dict['X_train'], data_dict['y_train'],
        data_dict['X_test'], data_dict['y_test'],
        optimize=True
    )
    
    # Analyze feature importance
    print("Analyzing feature importance...")
    importance_analysis = predictor.analyze_feature_importance(
        data_dict['X_train'], data_dict['feature_names']
    )
    
    # Generate predictions for full dataset
    print("Generating predictions for all customers...")
    full_predictions = predictor.generate_predictions(
        data_dict['full_data'].drop(['customer_id', 'churned'], axis=1, errors='ignore'),
        customer_ids=data_dict['full_data'].get('customer_id')
    )
    
    # Add predictions to full dataset
    df_with_predictions = data_dict['full_data'].copy()
    df_with_predictions = df_with_predictions.merge(
        full_predictions[['customer_id', 'churn_probability', 'risk_category']], 
        on='customer_id', 
        how='left'
    )
    
    # Save model and predictions
    predictor.save_model()
    df_with_predictions.to_csv('data/processed/data_with_predictions.csv', index=False)
    
    print(f"✅ Churn prediction completed")
    print(f"   - Best model: {predictor.best_model_name}")
    print(f"   - Model AUC: {results[predictor.best_model_name]['auc_score']:.4f}")
    print(f"   - High-risk customers: {len(df_with_predictions[df_with_predictions['churn_probability'] > 0.7]):,}")
    
    return df_with_predictions, predictor

def run_clv_analysis(df_with_predictions):
    """Run Customer Lifetime Value analysis"""
    print("\n" + "="*60)
    print("STEP 3: CUSTOMER LIFETIME VALUE ANALYSIS")
    print("="*60)
    
    clv_analyzer = CLVAnalyzer()
    df_with_clv, clv_report = clv_analyzer.run_complete_clv_analysis(df_with_predictions)
    
    # Save CLV results
    df_with_clv.to_csv('data/processed/data_with_clv.csv', index=False)
    
    print(f"✅ CLV analysis completed")
    print(f"   - Total customers: {clv_report['total_customers']:,}")
    print(f"   - Total CLV: ${clv_report['total_clv']:,.2f}")
    print(f"   - Average CLV: ${clv_report['average_clv']:,.2f}")
    
    if 'total_clv_at_risk' in clv_report:
        print(f"   - CLV at risk: ${clv_report['total_clv_at_risk']:,.2f} ({clv_report['clv_at_risk_percentage']:.1f}%)")
        print(f"   - High-risk high-value customers: {clv_report['high_risk_high_value_customers']:,}")
    
    return df_with_clv, clv_report

def run_financial_impact_analysis(df_with_clv):
    """Run financial impact analysis"""
    print("\n" + "="*60)
    print("STEP 4: FINANCIAL IMPACT ANALYSIS")
    print("="*60)
    
    # Define intervention costs
    intervention_costs = {
        'basic_outreach': 50,
        'discount_offer': 200,
        'premium_support': 150,
        'personalized_retention': 300
    }
    
    financial_analyzer = FinancialImpactAnalyzer()
    financial_report = financial_analyzer.generate_financial_report(df_with_clv, intervention_costs)
    
    print(f"✅ Financial impact analysis completed")
    print(f"   - Total revenue loss: ${financial_report['financial_impact']['total_revenue_loss']:,.2f}")
    print(f"   - Revenue at risk: ${financial_report['financial_impact']['revenue_at_risk']:,.2f}")
    print(f"   - Churned customers: {financial_report['financial_impact']['churned_customers_count']:,}")
    print(f"   - High-risk customers: {financial_report['financial_impact']['high_risk_customers_count']:,}")
    
    # Print best retention strategy
    best_strategy = max(
        financial_report['retention_scenarios'].items(), 
        key=lambda x: x[1]['roi_percentage']
    )
    print(f"   - Best retention strategy: {best_strategy[0]} (ROI: {best_strategy[1]['roi_percentage']:.1f}%)")
    
    return financial_report

def run_retention_strategies(df_with_clv):
    """Run retention strategy analysis"""
    print("\n" + "="*60)
    print("STEP 5: RETENTION STRATEGY DEVELOPMENT")
    print("="*60)
    
    retention_engine = RetentionStrategyEngine()
    retention_report = retention_engine.generate_retention_report(df_with_clv, total_budget=50000)
    
    print(f"✅ Retention strategy analysis completed")
    print(f"   - At-risk customers: {len(df_with_clv[df_with_clv.get('churn_probability', 0) > 0.3]):,}")
    print(f"   - Customers with strategies: {len(df_with_clv[df_with_clv.get('retention_strategy', 'no_action') != 'no_action']):,}")
    
    summary = retention_report['allocation_summary']
    print(f"   - Budget allocation: ${summary['total_budget_used']:,.2f} for {summary['total_customers_selected']:,} customers")
    print(f"   - Expected ROI: {summary['expected_total_roi']:.1f}%")
    print(f"   - Retention campaigns: {len(retention_report['campaigns'])}")
    
    return retention_report

def generate_executive_summary(data_dict, clv_report, financial_report, retention_report):
    """Generate executive summary report"""
    print("\n" + "="*60)
    print("EXECUTIVE SUMMARY")
    print("="*60)
    
    summary = f"""
CUSTOMER CHURN PREDICTION & FINANCIAL IMPACT ANALYSIS
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

KEY FINDINGS:
============
• Total Customers Analyzed: {len(data_dict['full_data']):,}
• Overall Churn Rate: {data_dict['y_train'].mean():.1%}
• Total Customer Lifetime Value: ${clv_report['total_clv']:,.2f}
• Revenue at Risk: ${financial_report['financial_impact']['revenue_at_risk']:,.2f}
• Total Financial Impact: ${financial_report['financial_impact']['total_financial_impact']:,.2f}

HIGH-PRIORITY ACTIONS:
=====================
• {financial_report['financial_impact']['high_risk_customers_count']:,} customers at high risk of churning
• ${clv_report.get('total_clv_at_risk', 0):,.2f} in CLV at immediate risk
• {retention_report['allocation_summary']['total_customers_selected']:,} customers selected for retention campaigns

RECOMMENDED RETENTION STRATEGY:
==============================
"""
    
    # Add best retention strategy details
    best_strategy = max(
        financial_report['retention_scenarios'].items(), 
        key=lambda x: x[1]['roi_percentage']
    )
    
    summary += f"""• Strategy: {best_strategy[0].replace('_', ' ').title()}
• Target Customers: {best_strategy[1]['target_customers']:,}
• Investment Required: ${best_strategy[1]['intervention_cost']:,.2f}
• Expected Revenue Saved: ${best_strategy[1]['revenue_saved']:,.2f}
• ROI: {best_strategy[1]['roi_percentage']:.1f}%

BUSINESS RECOMMENDATIONS:
========================
"""
    
    for rec in financial_report['recommendations']:
        summary += f"• {rec}\n"
    
    summary += f"""
NEXT STEPS:
===========
• Implement retention campaigns for high-priority customers
• Monitor churn prediction model performance
• Track retention campaign effectiveness
• Refine strategies based on results
• Schedule quarterly model retraining

For detailed analysis, please refer to the interactive dashboard.
"""
    
    # Save summary to file
    with open('reports/executive_summary.txt', 'w') as f:
        f.write(summary)
    
    print(summary)
    
    return summary

def main():
    """Main execution function"""
    print("🚀 CUSTOMER CHURN PREDICTION & FINANCIAL IMPACT ANALYSIS")
    print("=" * 80)
    print("This comprehensive analysis will:")
    print("1. Process customer data and engineer features")
    print("2. Train machine learning models for churn prediction")
    print("3. Calculate Customer Lifetime Value (CLV)")
    print("4. Quantify financial impact of churn")
    print("5. Develop data-driven retention strategies")
    print("=" * 80)
    
    try:
        # Create project structure
        create_project_directories()
        
        # Step 1: Data Preprocessing
        data_dict = run_data_preprocessing()
        
        # Step 2: Churn Prediction
        df_with_predictions, predictor = run_churn_prediction(data_dict)
        
        # Step 3: CLV Analysis
        df_with_clv, clv_report = run_clv_analysis(df_with_predictions)
        
        # Step 4: Financial Impact Analysis
        financial_report = run_financial_impact_analysis(df_with_clv)
        
        # Step 5: Retention Strategies
        retention_report = run_retention_strategies(df_with_clv)
        
        # Generate Executive Summary
        os.makedirs('reports', exist_ok=True)
        executive_summary = generate_executive_summary(
            data_dict, clv_report, financial_report, retention_report
        )
        
        print("\n" + "="*60)
        print("🎉 ANALYSIS COMPLETE!")
        print("="*60)
        print("Generated files:")
        print("• data/processed/data_with_retention_strategies.csv - Complete dataset with all analysis")
        print("• best_churn_model.pkl - Trained machine learning model")
        print("• reports/executive_summary.txt - Executive summary report")
        print("• Various visualization files (*.png)")
        print("\nTo view the interactive dashboard, run:")
        print("streamlit run dashboard/streamlit_app.py")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        print("Please check the error details and try again.")
        raise

if __name__ == "__main__":
    main()
