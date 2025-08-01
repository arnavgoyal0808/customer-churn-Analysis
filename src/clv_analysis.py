"""
Customer Lifetime Value (CLV) analysis module.
Calculates historical and predictive CLV for financial impact assessment.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from lifelines import KaplanMeierFitter, WeibullFitter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class CLVAnalyzer:
    def __init__(self):
        self.clv_model = None
        self.survival_model = None
        self.clv_segments = None
        
    def calculate_historical_clv(self, df):
        """Calculate historical CLV based on existing data"""
        clv_df = df.copy()
        
        # Basic CLV calculation: Total revenue generated
        clv_df['historical_clv'] = clv_df['total_charges']
        
        # Monthly CLV rate
        clv_df['monthly_clv'] = clv_df['monthly_charges']
        
        # Average revenue per month
        clv_df['avg_revenue_per_month'] = clv_df['total_charges'] / clv_df['tenure_months']
        
        # CLV efficiency (revenue per month of tenure)
        clv_df['clv_efficiency'] = clv_df['historical_clv'] / clv_df['tenure_months']
        
        return clv_df
    
    def predict_future_clv(self, df, prediction_horizon_months=24):
        """Predict future CLV using machine learning"""
        # Prepare features for CLV prediction
        feature_columns = [
            'age', 'tenure_months', 'monthly_charges', 'support_calls', 
            'late_payments', 'contract_type', 'payment_method', 
            'internet_service', 'service_complexity'
        ]
        
        # Filter for active customers (non-churned)
        active_customers = df[df['churned'] == 0].copy()
        
        if len(active_customers) == 0:
            print("No active customers found for CLV prediction")
            return df
        
        X = active_customers[feature_columns]
        y = active_customers['historical_clv']
        
        # Train CLV prediction model
        self.clv_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.clv_model.fit(X, y)
        
        # Predict CLV for all customers
        X_all = df[feature_columns]
        predicted_monthly_clv = self.clv_model.predict(X_all)
        
        # Calculate future CLV based on prediction horizon
        df['predicted_monthly_clv'] = predicted_monthly_clv / df['tenure_months']
        df['predicted_future_clv'] = df['predicted_monthly_clv'] * prediction_horizon_months
        
        # Adjust for churn probability (if available)
        if 'churn_probability' in df.columns:
            # Discount future CLV by survival probability
            survival_probability = 1 - df['churn_probability']
            df['risk_adjusted_clv'] = df['predicted_future_clv'] * survival_probability
        else:
            df['risk_adjusted_clv'] = df['predicted_future_clv']
        
        return df
    
    def survival_analysis(self, df):
        """Perform survival analysis to estimate customer lifespan"""
        # Prepare survival data
        durations = df['tenure_months']
        event_observed = df['churned']  # 1 if churned, 0 if censored
        
        # Kaplan-Meier survival analysis
        kmf = KaplanMeierFitter()
        kmf.fit(durations, event_observed, label='Customer Survival')
        
        # Weibull survival model for parametric estimation
        wf = WeibullFitter()
        wf.fit(durations, event_observed)
        
        self.survival_model = {
            'kaplan_meier': kmf,
            'weibull': wf
        }
        
        # Calculate expected remaining lifetime for active customers
        active_customers = df[df['churned'] == 0].copy()
        if len(active_customers) > 0:
            # Estimate remaining lifetime using Weibull model
            remaining_lifetime = []
            for tenure in active_customers['tenure_months']:
                # Conditional survival function
                remaining = wf.conditional_expected_durations(tenure)
                remaining_lifetime.append(remaining)
            
            active_customers['expected_remaining_months'] = remaining_lifetime
            
            # Merge back to main dataframe
            df = df.merge(
                active_customers[['customer_id', 'expected_remaining_months']], 
                on='customer_id', 
                how='left'
            )
            df['expected_remaining_months'].fillna(0, inplace=True)
        
        return df
    
    def segment_customers_by_clv(self, df):
        """Segment customers based on CLV"""
        # Use risk-adjusted CLV if available, otherwise use predicted future CLV
        clv_column = 'risk_adjusted_clv' if 'risk_adjusted_clv' in df.columns else 'predicted_future_clv'
        
        # Create CLV segments using quantiles
        df['clv_segment'] = pd.qcut(
            df[clv_column], 
            q=5, 
            labels=['Low Value', 'Below Average', 'Average', 'Above Average', 'High Value']
        )
        
        # Create CLV score (0-100)
        df['clv_score'] = (df[clv_column].rank(pct=True) * 100).round(0)
        
        # Segment analysis
        segment_analysis = df.groupby('clv_segment').agg({
            clv_column: ['count', 'mean', 'median', 'sum'],
            'churn_probability': 'mean' if 'churn_probability' in df.columns else lambda x: np.nan,
            'monthly_charges': 'mean',
            'tenure_months': 'mean'
        }).round(2)
        
        self.clv_segments = segment_analysis
        
        return df
    
    def calculate_clv_at_risk(self, df):
        """Calculate CLV at risk due to churn"""
        if 'churn_probability' not in df.columns:
            print("Churn probability not available. Cannot calculate CLV at risk.")
            return df
        
        # CLV at risk = CLV * Churn Probability
        clv_column = 'risk_adjusted_clv' if 'risk_adjusted_clv' in df.columns else 'predicted_future_clv'
        df['clv_at_risk'] = df[clv_column] * df['churn_probability']
        
        # High-risk customers (high CLV and high churn probability)
        df['high_risk_high_value'] = (
            (df['churn_probability'] > 0.7) & 
            (df['clv_score'] > 75)
        ).astype(int)
        
        return df
    
    def plot_clv_distribution(self, df):
        """Plot CLV distribution and segments"""
        clv_column = 'risk_adjusted_clv' if 'risk_adjusted_clv' in df.columns else 'predicted_future_clv'
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # CLV distribution
        axes[0, 0].hist(df[clv_column], bins=50, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('CLV Distribution')
        axes[0, 0].set_xlabel('Customer Lifetime Value ($)')
        axes[0, 0].set_ylabel('Frequency')
        
        # CLV by segment
        if 'clv_segment' in df.columns:
            segment_means = df.groupby('clv_segment')[clv_column].mean()
            axes[0, 1].bar(range(len(segment_means)), segment_means.values, color='lightcoral')
            axes[0, 1].set_title('Average CLV by Segment')
            axes[0, 1].set_xlabel('CLV Segment')
            axes[0, 1].set_ylabel('Average CLV ($)')
            axes[0, 1].set_xticks(range(len(segment_means)))
            axes[0, 1].set_xticklabels(segment_means.index, rotation=45)
        
        # CLV vs Churn Probability
        if 'churn_probability' in df.columns:
            scatter = axes[1, 0].scatter(df['churn_probability'], df[clv_column], 
                                       alpha=0.6, c=df['clv_score'], cmap='viridis')
            axes[1, 0].set_title('CLV vs Churn Probability')
            axes[1, 0].set_xlabel('Churn Probability')
            axes[1, 0].set_ylabel('CLV ($)')
            plt.colorbar(scatter, ax=axes[1, 0], label='CLV Score')
        
        # CLV at Risk
        if 'clv_at_risk' in df.columns:
            axes[1, 1].hist(df['clv_at_risk'], bins=50, alpha=0.7, color='orange')
            axes[1, 1].set_title('CLV at Risk Distribution')
            axes[1, 1].set_xlabel('CLV at Risk ($)')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('/q/bin/customer_churn_project/clv_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_survival_curves(self, df):
        """Plot survival analysis curves"""
        if self.survival_model is None:
            print("Survival analysis not performed. Run survival_analysis first.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Kaplan-Meier survival curve
        kmf = self.survival_model['kaplan_meier']
        kmf.plot_survival_function(ax=axes[0])
        axes[0].set_title('Kaplan-Meier Survival Curve')
        axes[0].set_xlabel('Tenure (Months)')
        axes[0].set_ylabel('Survival Probability')
        axes[0].grid(True, alpha=0.3)
        
        # Weibull hazard function
        wf = self.survival_model['weibull']
        wf.plot_hazard(ax=axes[1])
        axes[1].set_title('Weibull Hazard Function')
        axes[1].set_xlabel('Tenure (Months)')
        axes[1].set_ylabel('Hazard Rate')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/q/bin/customer_churn_project/survival_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_clv_report(self, df):
        """Generate comprehensive CLV analysis report"""
        clv_column = 'risk_adjusted_clv' if 'risk_adjusted_clv' in df.columns else 'predicted_future_clv'
        
        report = {
            'total_customers': len(df),
            'total_clv': df[clv_column].sum(),
            'average_clv': df[clv_column].mean(),
            'median_clv': df[clv_column].median(),
            'clv_std': df[clv_column].std(),
        }
        
        if 'clv_at_risk' in df.columns:
            report.update({
                'total_clv_at_risk': df['clv_at_risk'].sum(),
                'avg_clv_at_risk': df['clv_at_risk'].mean(),
                'high_risk_high_value_customers': df['high_risk_high_value'].sum(),
                'clv_at_risk_percentage': (df['clv_at_risk'].sum() / df[clv_column].sum()) * 100
            })
        
        if self.clv_segments is not None:
            report['segment_analysis'] = self.clv_segments
        
        return report
    
    def run_complete_clv_analysis(self, df):
        """Run complete CLV analysis pipeline"""
        print("Calculating historical CLV...")
        df = self.calculate_historical_clv(df)
        
        print("Predicting future CLV...")
        df = self.predict_future_clv(df)
        
        print("Performing survival analysis...")
        df = self.survival_analysis(df)
        
        print("Segmenting customers by CLV...")
        df = self.segment_customers_by_clv(df)
        
        print("Calculating CLV at risk...")
        df = self.calculate_clv_at_risk(df)
        
        print("Generating visualizations...")
        self.plot_clv_distribution(df)
        self.plot_survival_curves(df)
        
        print("Generating CLV report...")
        report = self.generate_clv_report(df)
        
        return df, report

if __name__ == "__main__":
    # Load processed data with churn predictions
    df = pd.read_csv('/q/bin/customer_churn_project/data/processed/processed_data.csv')
    
    # Add customer IDs if not present
    if 'customer_id' not in df.columns:
        df['customer_id'] = [f"CUST_{i:06d}" for i in range(1, len(df) + 1)]
    
    # Run CLV analysis
    clv_analyzer = CLVAnalyzer()
    df_with_clv, clv_report = clv_analyzer.run_complete_clv_analysis(df)
    
    # Save results
    df_with_clv.to_csv('/q/bin/customer_churn_project/data/processed/data_with_clv.csv', index=False)
    
    print("\nCLV Analysis Report:")
    print(f"Total Customers: {clv_report['total_customers']:,}")
    print(f"Total CLV: ${clv_report['total_clv']:,.2f}")
    print(f"Average CLV: ${clv_report['average_clv']:,.2f}")
    
    if 'total_clv_at_risk' in clv_report:
        print(f"Total CLV at Risk: ${clv_report['total_clv_at_risk']:,.2f}")
        print(f"CLV at Risk %: {clv_report['clv_at_risk_percentage']:.1f}%")
    
    print("\nCLV analysis complete!")
