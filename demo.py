"""
Simplified demo of the Customer Churn Prediction project.
This demo showcases the core functionality with basic libraries.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

def generate_sample_data(n_samples=5000):
    """Generate synthetic customer data for demonstration"""
    print("Generating sample customer data...")
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
        'support_calls': support_calls,
        'late_payments': late_payments,
        'churned': churned,
        'churn_probability': churn_prob
    })
    
    return data

def analyze_churn_patterns(df):
    """Analyze churn patterns in the data"""
    print("\\n" + "="*60)
    print("CHURN PATTERN ANALYSIS")
    print("="*60)
    
    # Basic statistics
    churn_rate = df['churned'].mean()
    print(f"Overall churn rate: {churn_rate:.2%}")
    print(f"Total customers: {len(df):,}")
    print(f"Churned customers: {df['churned'].sum():,}")
    print(f"Retained customers: {(df['churned'] == 0).sum():,}")
    
    # Churn by contract type
    print("\\nChurn rate by contract type:")
    churn_by_contract = df.groupby('contract_type')['churned'].agg(['count', 'sum', 'mean'])
    churn_by_contract.columns = ['Total', 'Churned', 'Churn_Rate']
    churn_by_contract['Churn_Rate_Pct'] = (churn_by_contract['Churn_Rate'] * 100).round(1)
    print(churn_by_contract.sort_values('Churn_Rate', ascending=False))
    
    # Churn by payment method
    print("\\nChurn rate by payment method:")
    churn_by_payment = df.groupby('payment_method')['churned'].agg(['count', 'sum', 'mean'])
    churn_by_payment.columns = ['Total', 'Churned', 'Churn_Rate']
    churn_by_payment['Churn_Rate_Pct'] = (churn_by_payment['Churn_Rate'] * 100).round(1)
    print(churn_by_payment.sort_values('Churn_Rate', ascending=False))
    
    return churn_by_contract, churn_by_payment

def calculate_clv_and_financial_impact(df):
    """Calculate Customer Lifetime Value and financial impact"""
    print("\\n" + "="*60)
    print("CUSTOMER LIFETIME VALUE & FINANCIAL IMPACT ANALYSIS")
    print("="*60)
    
    # Calculate CLV (simplified)
    df['estimated_clv'] = df['monthly_charges'] * 24  # 2-year estimate
    df['clv_at_risk'] = df['estimated_clv'] * df['churn_probability']
    
    # Financial impact metrics
    churned_customers = df[df['churned'] == 1]
    immediate_loss = churned_customers['monthly_charges'].sum()
    future_loss = churned_customers['estimated_clv'].sum()
    total_clv_at_risk = df['clv_at_risk'].sum()
    
    print(f"Immediate monthly revenue loss: ${immediate_loss:,.2f}")
    print(f"Future revenue loss (CLV): ${future_loss:,.2f}")
    print(f"Total CLV at risk: ${total_clv_at_risk:,.2f}")
    print(f"Average CLV per customer: ${df['estimated_clv'].mean():,.2f}")
    print(f"Average CLV per churned customer: ${churned_customers['estimated_clv'].mean():,.2f}")
    
    # High-risk customers
    high_risk_customers = df[df['churn_probability'] > 0.7]
    print(f"\\nHigh-risk customers (>70% churn probability): {len(high_risk_customers):,}")
    print(f"Revenue at risk from high-risk customers: ${high_risk_customers['monthly_charges'].sum():,.2f}")
    
    return {
        'immediate_loss': immediate_loss,
        'future_loss': future_loss,
        'total_clv_at_risk': total_clv_at_risk,
        'high_risk_customers': len(high_risk_customers)
    }

def develop_retention_strategies(df):
    """Develop retention strategies based on customer segments"""
    print("\\n" + "="*60)
    print("RETENTION STRATEGY DEVELOPMENT")
    print("="*60)
    
    # Segment customers by risk and value
    df['risk_category'] = pd.cut(df['churn_probability'], 
                                bins=[0, 0.3, 0.7, 1.0], 
                                labels=['Low Risk', 'Medium Risk', 'High Risk'])
    
    df['value_category'] = pd.qcut(df['estimated_clv'], 
                                  q=3, 
                                  labels=['Low Value', 'Medium Value', 'High Value'])
    
    # Strategy assignment
    def assign_strategy(row):
        if row['risk_category'] == 'High Risk' and row['value_category'] == 'High Value':
            return 'Premium Retention Program'
        elif row['risk_category'] == 'High Risk':
            return 'Standard Retention Outreach'
        elif row['risk_category'] == 'Medium Risk' and row['value_category'] == 'High Value':
            return 'Proactive Engagement'
        elif row['risk_category'] == 'Medium Risk':
            return 'Basic Monitoring'
        else:
            return 'No Action Required'
    
    df['retention_strategy'] = df.apply(assign_strategy, axis=1)
    
    # Strategy summary
    strategy_summary = df.groupby('retention_strategy').agg({
        'customer_id': 'count',
        'monthly_charges': 'sum',
        'estimated_clv': 'sum',
        'churn_probability': 'mean'
    }).round(2)
    strategy_summary.columns = ['Customers', 'Monthly_Revenue', 'Total_CLV', 'Avg_Churn_Prob']
    
    print("Retention strategy summary:")
    print(strategy_summary.sort_values('Total_CLV', ascending=False))
    
    # Calculate intervention ROI
    intervention_costs = {
        'Premium Retention Program': 300,
        'Standard Retention Outreach': 150,
        'Proactive Engagement': 100,
        'Basic Monitoring': 25,
        'No Action Required': 0
    }
    
    print("\\nRetention strategy ROI analysis:")
    for strategy in strategy_summary.index:
        if strategy == 'No Action Required':
            continue
            
        customers = strategy_summary.loc[strategy, 'Customers']
        avg_clv = strategy_summary.loc[strategy, 'Total_CLV'] / customers
        avg_churn_prob = strategy_summary.loc[strategy, 'Avg_Churn_Prob']
        cost_per_customer = intervention_costs[strategy]
        
        # Assume 30% reduction in churn probability
        churn_reduction = avg_churn_prob * 0.3
        revenue_saved = avg_clv * churn_reduction
        roi = ((revenue_saved - cost_per_customer) / cost_per_customer) * 100
        
        print(f"{strategy}:")
        print(f"  - Customers: {customers}")
        print(f"  - Cost per customer: ${cost_per_customer}")
        print(f"  - Expected revenue saved: ${revenue_saved:.2f}")
        print(f"  - ROI: {roi:.1f}%")
        print()
    
    return df, strategy_summary

def create_visualizations(df, churn_by_contract, financial_metrics):
    """Create key visualizations"""
    print("\\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Churn rate by contract type
    churn_by_contract['Churn_Rate_Pct'].plot(kind='bar', ax=axes[0, 0], color='lightcoral')
    axes[0, 0].set_title('Churn Rate by Contract Type')
    axes[0, 0].set_ylabel('Churn Rate (%)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. CLV distribution
    axes[0, 1].hist(df['estimated_clv'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].set_title('Customer Lifetime Value Distribution')
    axes[0, 1].set_xlabel('CLV ($)')
    axes[0, 1].set_ylabel('Frequency')
    
    # 3. Risk category distribution
    risk_counts = df['risk_category'].value_counts()
    colors = ['green', 'orange', 'red']
    axes[1, 0].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
    axes[1, 0].set_title('Customer Risk Distribution')
    
    # 4. Financial impact
    categories = ['Immediate Loss', 'Future Loss']
    values = [financial_metrics['immediate_loss'], financial_metrics['future_loss']]
    axes[1, 1].bar(categories, values, color=['lightcoral', 'orange'])
    axes[1, 1].set_title('Financial Impact of Churn')
    axes[1, 1].set_ylabel('Revenue Loss ($)')
    
    # Add value labels on bars
    for i, v in enumerate(values):
        axes[1, 1].text(i, v + max(values) * 0.01, f'${v:,.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('churn_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    print("✅ Visualizations saved as 'churn_analysis_dashboard.png'")
    plt.show()

def generate_executive_summary(df, financial_metrics, strategy_summary):
    """Generate executive summary"""
    print("\\n" + "="*80)
    print("EXECUTIVE SUMMARY - CUSTOMER CHURN PREDICTION & RETENTION ANALYSIS")
    print("="*80)
    
    summary = f"""
ANALYSIS DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

KEY FINDINGS:
=============
• Total Customers Analyzed: {len(df):,}
• Overall Churn Rate: {df['churned'].mean():.1%}
• High-Risk Customers: {len(df[df['churn_probability'] > 0.7]):,} ({len(df[df['churn_probability'] > 0.7])/len(df):.1%})
• Total Customer Lifetime Value: ${df['estimated_clv'].sum():,.2f}

FINANCIAL IMPACT:
================
• Immediate Monthly Revenue Loss: ${financial_metrics['immediate_loss']:,.2f}
• Future Revenue Loss (CLV): ${financial_metrics['future_loss']:,.2f}
• Total CLV at Risk: ${financial_metrics['total_clv_at_risk']:,.2f}
• Average CLV per Customer: ${df['estimated_clv'].mean():,.2f}

HIGH-PRIORITY SEGMENTS:
======================
• Month-to-month contracts show highest churn risk
• Electronic check payment method correlates with higher churn
• Customers with high support calls are at elevated risk
• New customers (< 12 months tenure) require attention

RETENTION STRATEGY RECOMMENDATIONS:
==================================
• Implement Premium Retention Program for high-value, high-risk customers
• Focus on contract conversion from month-to-month to longer terms
• Improve payment experience to reduce electronic check dependency
• Enhance onboarding process for new customers
• Proactive support for customers with multiple service calls

EXPECTED ROI:
============
• Premium Retention Program: Highest ROI for valuable customers
• Standard Retention Outreach: Cost-effective for medium-risk segments
• Proactive Engagement: Preventive approach for valuable customers

NEXT STEPS:
===========
1. Implement retention campaigns for identified high-risk segments
2. Monitor campaign effectiveness and adjust strategies
3. Develop predictive models for early churn detection
4. Establish regular churn analysis reporting
5. Train customer service teams on retention techniques

BUSINESS IMPACT:
===============
• Potential revenue protection: ${df['clv_at_risk'].sum():,.2f}
• Recommended retention budget: ${len(df[df['retention_strategy'] != 'No Action Required']) * 150:,.2f}
• Expected customer saves: {len(df[df['churn_probability'] > 0.5]) * 0.3:.0f} customers
"""
    
    print(summary)
    
    # Save summary to file
    with open('executive_summary.txt', 'w') as f:
        f.write(summary)
    
    print("\\n✅ Executive summary saved as 'executive_summary.txt'")
    
    return summary

def main():
    """Main execution function"""
    print("🚀 CUSTOMER CHURN PREDICTION & FINANCIAL IMPACT ANALYSIS - DEMO")
    print("=" * 80)
    print("This demo showcases a comprehensive customer churn analysis including:")
    print("• Data generation and preprocessing")
    print("• Churn pattern analysis")
    print("• Customer Lifetime Value calculation")
    print("• Financial impact assessment")
    print("• Retention strategy development")
    print("• Executive reporting")
    print("=" * 80)
    
    try:
        # Step 1: Generate sample data
        df = generate_sample_data(n_samples=5000)
        print(f"✅ Generated {len(df):,} customer records")
        
        # Step 2: Analyze churn patterns
        churn_by_contract, churn_by_payment = analyze_churn_patterns(df)
        
        # Step 3: Calculate CLV and financial impact
        financial_metrics = calculate_clv_and_financial_impact(df)
        
        # Step 4: Develop retention strategies
        df_with_strategies, strategy_summary = develop_retention_strategies(df)
        
        # Step 5: Create visualizations
        create_visualizations(df_with_strategies, churn_by_contract, financial_metrics)
        
        # Step 6: Generate executive summary
        executive_summary = generate_executive_summary(df_with_strategies, financial_metrics, strategy_summary)
        
        # Save processed data
        df_with_strategies.to_csv('customer_churn_analysis_results.csv', index=False)
        print("\\n✅ Complete analysis results saved as 'customer_churn_analysis_results.csv'")
        
        print("\\n" + "="*60)
        print("🎉 ANALYSIS COMPLETE!")
        print("="*60)
        print("Generated files:")
        print("• customer_churn_analysis_results.csv - Complete dataset with analysis")
        print("• churn_analysis_dashboard.png - Key visualizations")
        print("• executive_summary.txt - Executive summary report")
        print("\\nThis demo showcases the core capabilities of a comprehensive")
        print("customer churn prediction and retention strategy system.")
        
    except Exception as e:
        print(f"\\n❌ Error during analysis: {str(e)}")
        print("Please check the error details and try again.")
        raise

if __name__ == "__main__":
    main()
