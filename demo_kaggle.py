"""
Customer Churn Prediction Demo using Real Kaggle Telco Dataset.
This demo showcases the complete analysis pipeline with real customer data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append('src')

from data_preprocessing_kaggle import KaggleDataPreprocessor

# Set style
plt.style.use('default')
sns.set_palette("husl")

def analyze_kaggle_churn_patterns(df):
    """Analyze churn patterns in the Kaggle dataset"""
    print("\n" + "="*60)
    print("KAGGLE TELCO CHURN PATTERN ANALYSIS")
    print("="*60)
    
    # Basic statistics
    churn_rate = df['churned'].mean()
    print(f"Overall churn rate: {churn_rate:.2%}")
    print(f"Total customers: {len(df):,}")
    print(f"Churned customers: {df['churned'].sum():,}")
    print(f"Retained customers: {(df['churned'] == 0).sum():,}")
    
    # Analyze key categorical variables
    categorical_vars = ['contract_type', 'payment_method', 'internet_service', 'gender']
    
    for var in categorical_vars:
        if var in df.columns:
            print(f"\nChurn rate by {var.replace('_', ' ').title()}:")
            churn_analysis = df.groupby(var)['churned'].agg(['count', 'sum', 'mean'])
            churn_analysis.columns = ['Total', 'Churned', 'Churn_Rate']
            churn_analysis['Churn_Rate_Pct'] = (churn_analysis['Churn_Rate'] * 100).round(1)
            print(churn_analysis.sort_values('Churn_Rate', ascending=False))
    
    # Analyze numerical variables
    print(f"\nChurn analysis by tenure:")
    df['tenure_group'] = pd.cut(df['tenure_months'], 
                               bins=[0, 12, 24, 48, float('inf')],
                               labels=['0-12m', '12-24m', '24-48m', '48m+'])
    
    tenure_analysis = df.groupby('tenure_group')['churned'].agg(['count', 'mean'])
    tenure_analysis.columns = ['Customer_Count', 'Churn_Rate']
    tenure_analysis['Churn_Rate_Pct'] = (tenure_analysis['Churn_Rate'] * 100).round(1)
    print(tenure_analysis)
    
    # Monthly charges analysis
    print(f"\nChurn analysis by monthly charges:")
    df['charges_group'] = pd.qcut(df['monthly_charges'], 
                                 q=4, 
                                 labels=['Low', 'Medium', 'High', 'Premium'])
    
    charges_analysis = df.groupby('charges_group')['churned'].agg(['count', 'mean'])
    charges_analysis.columns = ['Customer_Count', 'Churn_Rate']
    charges_analysis['Churn_Rate_Pct'] = (charges_analysis['Churn_Rate'] * 100).round(1)
    print(charges_analysis)
    
    return df

def calculate_kaggle_clv_and_impact(df):
    """Calculate CLV and financial impact for Kaggle dataset"""
    print("\n" + "="*60)
    print("KAGGLE DATASET - CLV & FINANCIAL IMPACT ANALYSIS")
    print("="*60)
    
    # Calculate CLV metrics
    df['estimated_clv'] = df['monthly_charges'] * 24  # 2-year estimate
    df['historical_clv'] = df['total_charges']
    
    # More sophisticated CLV calculation based on tenure
    df['monthly_clv_rate'] = df['total_charges'] / df['tenure_months']
    df['monthly_clv_rate'] = df['monthly_clv_rate'].fillna(df['monthly_charges'])
    df['predicted_clv'] = df['monthly_clv_rate'] * 24
    
    # Financial impact metrics
    churned_customers = df[df['churned'] == 1]
    retained_customers = df[df['churned'] == 0]
    
    immediate_loss = churned_customers['monthly_charges'].sum()
    historical_loss = churned_customers['total_charges'].sum()
    future_loss = churned_customers['predicted_clv'].sum()
    
    print(f"Financial Impact Analysis:")
    print(f"  • Immediate monthly revenue loss: ${immediate_loss:,.2f}")
    print(f"  • Historical revenue from churned customers: ${historical_loss:,.2f}")
    print(f"  • Predicted future revenue loss: ${future_loss:,.2f}")
    print(f"  • Average CLV per customer: ${df['predicted_clv'].mean():,.2f}")
    print(f"  • Average CLV per churned customer: ${churned_customers['predicted_clv'].mean():,.2f}")
    print(f"  • Average CLV per retained customer: ${retained_customers['predicted_clv'].mean():,.2f}")
    
    # Revenue concentration analysis
    print(f"\nRevenue Concentration Analysis:")
    high_value_customers = df[df['predicted_clv'] > df['predicted_clv'].quantile(0.8)]
    high_value_churn_rate = high_value_customers['churned'].mean()
    
    print(f"  • Top 20% customers by CLV: {len(high_value_customers):,}")
    print(f"  • Churn rate among high-value customers: {high_value_churn_rate:.2%}")
    print(f"  • Revenue at risk from high-value customers: ${high_value_customers[high_value_customers['churned']==1]['predicted_clv'].sum():,.2f}")
    
    return {
        'immediate_loss': immediate_loss,
        'historical_loss': historical_loss,
        'future_loss': future_loss,
        'high_value_churn_rate': high_value_churn_rate,
        'total_customers': len(df),
        'churned_customers': len(churned_customers)
    }

def develop_kaggle_retention_strategies(df):
    """Develop retention strategies based on Kaggle dataset insights"""
    print("\n" + "="*60)
    print("KAGGLE DATASET - RETENTION STRATEGY DEVELOPMENT")
    print("="*60)
    
    # Create risk segments based on actual data patterns
    def assign_risk_category(row):
        risk_score = 0
        
        # Contract type risk
        if row.get('contract_type') == 'Month-to-month':
            risk_score += 3
        elif row.get('contract_type') == 'One year':
            risk_score += 1
        
        # Payment method risk
        if row.get('payment_method') == 'Electronic check':
            risk_score += 2
        
        # Tenure risk
        if row['tenure_months'] < 12:
            risk_score += 2
        elif row['tenure_months'] < 24:
            risk_score += 1
        
        # Service complexity risk
        if row.get('service_complexity', 0) < 2:
            risk_score += 1
        
        # Internet service risk
        if row.get('internet_service') == 'Fiber optic':
            risk_score += 1
        
        # Categorize risk
        if risk_score >= 6:
            return 'Critical Risk'
        elif risk_score >= 4:
            return 'High Risk'
        elif risk_score >= 2:
            return 'Medium Risk'
        else:
            return 'Low Risk'
    
    # Apply risk categorization
    df['risk_category'] = df.apply(assign_risk_category, axis=1)
    
    # Create value segments
    df['value_category'] = pd.qcut(df['predicted_clv'], 
                                  q=4, 
                                  labels=['Low Value', 'Medium Value', 'High Value', 'Premium Value'])
    
    # Strategy assignment based on risk and value
    def assign_retention_strategy(row):
        risk = row['risk_category']
        value = row['value_category']
        
        if risk == 'Critical Risk' and value in ['High Value', 'Premium Value']:
            return 'Executive Intervention Program'
        elif risk == 'Critical Risk':
            return 'Urgent Retention Campaign'
        elif risk == 'High Risk' and value in ['High Value', 'Premium Value']:
            return 'Premium Retention Program'
        elif risk == 'High Risk':
            return 'Standard Retention Outreach'
        elif risk == 'Medium Risk' and value in ['High Value', 'Premium Value']:
            return 'Proactive Value Protection'
        elif risk == 'Medium Risk':
            return 'Engagement Enhancement'
        else:
            return 'Loyalty Maintenance'
    
    df['retention_strategy'] = df.apply(assign_retention_strategy, axis=1)
    
    # Strategy analysis
    strategy_analysis = df.groupby(['retention_strategy', 'risk_category']).agg({
        'customer_id': 'count',
        'monthly_charges': 'sum',
        'predicted_clv': 'sum',
        'churned': 'mean'
    }).round(2)
    strategy_analysis.columns = ['Customers', 'Monthly_Revenue', 'Total_CLV', 'Actual_Churn_Rate']
    
    print("Retention Strategy Analysis:")
    print(strategy_analysis)
    
    # ROI calculation for strategies
    intervention_costs = {
        'Executive Intervention Program': 500,
        'Urgent Retention Campaign': 250,
        'Premium Retention Program': 300,
        'Standard Retention Outreach': 150,
        'Proactive Value Protection': 200,
        'Engagement Enhancement': 75,
        'Loyalty Maintenance': 25
    }
    
    print(f"\nRetention Strategy ROI Analysis:")
    strategy_summary = df.groupby('retention_strategy').agg({
        'customer_id': 'count',
        'predicted_clv': 'mean',
        'churned': 'mean'
    })
    
    for strategy in strategy_summary.index:
        customers = strategy_summary.loc[strategy, 'customer_id']
        avg_clv = strategy_summary.loc[strategy, 'predicted_clv']
        actual_churn_rate = strategy_summary.loc[strategy, 'churned']
        cost_per_customer = intervention_costs.get(strategy, 100)
        
        # Assume intervention reduces churn by 40%
        churn_reduction = actual_churn_rate * 0.4
        revenue_saved = avg_clv * churn_reduction
        roi = ((revenue_saved - cost_per_customer) / cost_per_customer) * 100 if cost_per_customer > 0 else 0
        
        print(f"\n{strategy}:")
        print(f"  • Target customers: {customers:,}")
        print(f"  • Current churn rate: {actual_churn_rate:.1%}")
        print(f"  • Average CLV: ${avg_clv:,.2f}")
        print(f"  • Cost per customer: ${cost_per_customer}")
        print(f"  • Expected revenue saved: ${revenue_saved:,.2f}")
        print(f"  • ROI: {roi:.1f}%")
    
    return df, strategy_analysis

def create_kaggle_visualizations(df, financial_metrics):
    """Create visualizations for Kaggle dataset analysis"""
    print("\n" + "="*60)
    print("CREATING KAGGLE DATASET VISUALIZATIONS")
    print("="*60)
    
    # Create comprehensive dashboard
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # 1. Churn rate by contract type
    if 'contract_type' in df.columns:
        contract_churn = df.groupby('contract_type')['churned'].mean() * 100
        contract_churn.plot(kind='bar', ax=axes[0, 0], color='lightcoral')
        axes[0, 0].set_title('Churn Rate by Contract Type')
        axes[0, 0].set_ylabel('Churn Rate (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Churn rate by payment method
    if 'payment_method' in df.columns:
        payment_churn = df.groupby('payment_method')['churned'].mean() * 100
        payment_churn.plot(kind='bar', ax=axes[0, 1], color='orange')
        axes[0, 1].set_title('Churn Rate by Payment Method')
        axes[0, 1].set_ylabel('Churn Rate (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. CLV distribution
    axes[1, 0].hist(df['predicted_clv'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 0].set_title('Customer Lifetime Value Distribution')
    axes[1, 0].set_xlabel('Predicted CLV ($)')
    axes[1, 0].set_ylabel('Frequency')
    
    # 4. Tenure vs Monthly Charges (colored by churn)
    scatter = axes[1, 1].scatter(df['tenure_months'], df['monthly_charges'], 
                                c=df['churned'], cmap='RdYlBu_r', alpha=0.6)
    axes[1, 1].set_title('Tenure vs Monthly Charges (Red=Churned)')
    axes[1, 1].set_xlabel('Tenure (Months)')
    axes[1, 1].set_ylabel('Monthly Charges ($)')
    plt.colorbar(scatter, ax=axes[1, 1])
    
    # 5. Risk category distribution
    if 'risk_category' in df.columns:
        risk_counts = df['risk_category'].value_counts()
        colors = ['green', 'yellow', 'orange', 'red']
        axes[2, 0].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
                      colors=colors[:len(risk_counts)], startangle=90)
        axes[2, 0].set_title('Customer Risk Distribution')
    
    # 6. Financial impact breakdown
    categories = ['Immediate Loss', 'Historical Loss', 'Future Loss']
    values = [financial_metrics['immediate_loss'], 
             financial_metrics['historical_loss'], 
             financial_metrics['future_loss']]
    
    bars = axes[2, 1].bar(categories, values, color=['lightcoral', 'orange', 'red'])
    axes[2, 1].set_title('Financial Impact Analysis')
    axes[2, 1].set_ylabel('Revenue ($)')
    axes[2, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        axes[2, 1].text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                       f'${value:,.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('kaggle_churn_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    print("✅ Kaggle dataset visualizations saved as 'kaggle_churn_analysis_dashboard.png'")
    plt.show()

def generate_kaggle_executive_summary(df, financial_metrics, strategy_analysis):
    """Generate executive summary for Kaggle dataset analysis"""
    print("\n" + "="*80)
    print("EXECUTIVE SUMMARY - KAGGLE TELCO CUSTOMER CHURN ANALYSIS")
    print("="*80)
    
    # Calculate key metrics
    total_customers = len(df)
    churn_rate = df['churned'].mean()
    total_clv = df['predicted_clv'].sum()
    avg_clv = df['predicted_clv'].mean()
    
    # High-risk analysis
    high_risk_customers = len(df[df['risk_category'].isin(['Critical Risk', 'High Risk'])])
    high_value_customers = len(df[df['value_category'].isin(['High Value', 'Premium Value'])])
    
    summary = f"""
ANALYSIS DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
DATASET: Kaggle Telco Customer Churn Dataset

BUSINESS OVERVIEW:
==================
• Total Customers Analyzed: {total_customers:,}
• Overall Churn Rate: {churn_rate:.1%}
• Total Customer Portfolio Value: ${total_clv:,.2f}
• Average Customer Lifetime Value: ${avg_clv:,.2f}

FINANCIAL IMPACT:
=================
• Immediate Monthly Revenue Loss: ${financial_metrics['immediate_loss']:,.2f}
• Historical Revenue from Churned Customers: ${financial_metrics['historical_loss']:,.2f}
• Predicted Future Revenue Loss: ${financial_metrics['future_loss']:,.2f}
• Total Financial Impact: ${sum([financial_metrics['immediate_loss'], financial_metrics['future_loss']]):,.2f}

RISK ASSESSMENT:
================
• High-Risk Customers: {high_risk_customers:,} ({high_risk_customers/total_customers:.1%})
• High-Value Customers: {high_value_customers:,} ({high_value_customers/total_customers:.1%})
• High-Value Customer Churn Rate: {financial_metrics.get('high_value_churn_rate', 0):.1%}

KEY CHURN DRIVERS (from Kaggle dataset):
========================================
• Contract Type: Month-to-month contracts show highest churn risk
• Payment Method: Electronic check payments correlate with higher churn
• Tenure: New customers (< 12 months) are most vulnerable
• Internet Service: Fiber optic customers show elevated churn rates
• Service Complexity: Customers with fewer add-on services churn more

RETENTION STRATEGY RECOMMENDATIONS:
===================================
• PRIORITY 1: Executive Intervention for high-value, critical-risk customers
• PRIORITY 2: Contract conversion campaigns (month-to-month to annual)
• PRIORITY 3: Payment method optimization (reduce electronic check dependency)
• PRIORITY 4: Enhanced onboarding for new customers
• PRIORITY 5: Service bundling incentives to increase complexity

EXPECTED BUSINESS IMPACT:
========================
• Potential Revenue Protection: ${df[df['risk_category'].isin(['Critical Risk', 'High Risk'])]['predicted_clv'].sum():,.2f}
• Recommended Retention Investment: ${len(df[df['retention_strategy'] != 'Loyalty Maintenance']) * 200:,.2f}
• Expected Customer Retention: {len(df[df['risk_category'].isin(['Critical Risk', 'High Risk'])]) * 0.4:.0f} customers

IMPLEMENTATION ROADMAP:
======================
1. IMMEDIATE (0-30 days):
   - Launch executive intervention for critical-risk, high-value customers
   - Implement electronic check payment migration program
   
2. SHORT-TERM (1-3 months):
   - Deploy contract conversion campaigns
   - Enhance new customer onboarding process
   
3. MEDIUM-TERM (3-6 months):
   - Develop service bundling incentive programs
   - Implement predictive churn scoring system
   
4. LONG-TERM (6+ months):
   - Establish continuous churn monitoring and response system
   - Develop customer lifetime value optimization programs

MONITORING & SUCCESS METRICS:
=============================
• Monthly churn rate reduction target: 2-3 percentage points
• Customer retention rate improvement: 15-20%
• Revenue protection target: ${financial_metrics['future_loss'] * 0.3:,.2f}
• ROI target for retention programs: 200%+
"""
    
    print(summary)
    
    # Save summary
    with open('kaggle_executive_summary.txt', 'w') as f:
        f.write(summary)
    
    print("\n✅ Kaggle executive summary saved as 'kaggle_executive_summary.txt'")
    return summary

def main():
    """Main execution function for Kaggle dataset analysis"""
    print("🚀 KAGGLE TELCO CUSTOMER CHURN PREDICTION & ANALYSIS")
    print("=" * 80)
    print("This analysis uses the real Kaggle Telco Customer Churn dataset to:")
    print("• Analyze actual customer churn patterns")
    print("• Calculate real customer lifetime values")
    print("• Assess genuine financial impact")
    print("• Develop data-driven retention strategies")
    print("• Generate actionable business insights")
    print("=" * 80)
    
    try:
        # Step 1: Load and process Kaggle data
        print("\nStep 1: Loading and processing Kaggle dataset...")
        preprocessor = KaggleDataPreprocessor()
        data_dict = preprocessor.process_kaggle_data()
        df = data_dict['full_data']
        
        # Step 2: Analyze churn patterns
        print("\nStep 2: Analyzing churn patterns...")
        df = analyze_kaggle_churn_patterns(df)
        
        # Step 3: Calculate CLV and financial impact
        print("\nStep 3: Calculating CLV and financial impact...")
        financial_metrics = calculate_kaggle_clv_and_impact(df)
        
        # Step 4: Develop retention strategies
        print("\nStep 4: Developing retention strategies...")
        df, strategy_analysis = develop_kaggle_retention_strategies(df)
        
        # Step 5: Create visualizations
        print("\nStep 5: Creating visualizations...")
        create_kaggle_visualizations(df, financial_metrics)
        
        # Step 6: Generate executive summary
        print("\nStep 6: Generating executive summary...")
        executive_summary = generate_kaggle_executive_summary(df, financial_metrics, strategy_analysis)
        
        # Save complete results
        df.to_csv('kaggle_churn_analysis_complete.csv', index=False)
        print("\n✅ Complete Kaggle analysis results saved as 'kaggle_churn_analysis_complete.csv'")
        
        print("\n" + "="*60)
        print("🎉 KAGGLE TELCO CHURN ANALYSIS COMPLETE!")
        print("="*60)
        print("Generated files:")
        print("• kaggle_churn_analysis_complete.csv - Complete dataset with analysis")
        print("• kaggle_churn_analysis_dashboard.png - Comprehensive visualizations")
        print("• kaggle_executive_summary.txt - Executive summary report")
        print("• data/processed/kaggle_processed_data.csv - Processed dataset")
        print("\nThis analysis provides real insights from actual telecom customer data")
        print("and can be directly applied to business decision-making.")
        
        # Display key insights
        print(f"\n📊 KEY INSIGHTS:")
        print(f"• Dataset contains {len(df):,} real customer records")
        print(f"• Actual churn rate: {df['churned'].mean():.1%}")
        print(f"• Total revenue at risk: ${financial_metrics['future_loss']:,.2f}")
        print(f"• High-risk customers identified: {len(df[df['risk_category'].isin(['Critical Risk', 'High Risk'])]):,}")
        
    except Exception as e:
        print(f"\n❌ Error during Kaggle analysis: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Ensure you have Kaggle API credentials set up")
        print("2. Check your internet connection")
        print("3. Verify kagglehub package is installed")
        print("\nTo set up Kaggle credentials:")
        print("• Go to kaggle.com -> Account -> API -> Create New API Token")
        print("• Place kaggle.json in ~/.kaggle/ directory")
        print("• Run: chmod 600 ~/.kaggle/kaggle.json")
        raise

if __name__ == "__main__":
    main()
