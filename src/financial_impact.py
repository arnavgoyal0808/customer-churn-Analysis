"""
Financial impact analysis module for customer churn.
Quantifies revenue loss and calculates ROI for retention strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FinancialImpactAnalyzer:
    def __init__(self):
        self.impact_metrics = {}
        self.retention_scenarios = {}
        
    def calculate_churn_financial_impact(self, df):
        """Calculate direct financial impact of customer churn"""
        # Immediate revenue loss (monthly charges of churned customers)
        churned_customers = df[df['churned'] == 1]
        immediate_loss = churned_customers['monthly_charges'].sum()
        
        # Future revenue loss (CLV of churned customers)
        if 'risk_adjusted_clv' in df.columns:
            future_loss = churned_customers['risk_adjusted_clv'].sum()
        elif 'predicted_future_clv' in df.columns:
            future_loss = churned_customers['predicted_future_clv'].sum()
        else:
            # Estimate based on average tenure and monthly charges
            avg_remaining_months = 24  # Assumption
            future_loss = churned_customers['monthly_charges'].sum() * avg_remaining_months
        
        # Potential revenue at risk (high probability churn customers)
        if 'churn_probability' in df.columns:
            high_risk_customers = df[df['churn_probability'] > 0.7]
            revenue_at_risk = high_risk_customers['monthly_charges'].sum()
            
            if 'clv_at_risk' in df.columns:
                clv_at_risk = df['clv_at_risk'].sum()
            else:
                clv_at_risk = revenue_at_risk * 24  # Estimate
        else:
            revenue_at_risk = 0
            clv_at_risk = 0
        
        # Customer acquisition cost impact
        # Assume it costs 5x monthly revenue to acquire a new customer
        acquisition_cost_multiplier = 5
        replacement_cost = churned_customers['monthly_charges'].sum() * acquisition_cost_multiplier
        
        self.impact_metrics = {
            'immediate_revenue_loss': immediate_loss,
            'future_revenue_loss': future_loss,
            'total_revenue_loss': immediate_loss + future_loss,
            'revenue_at_risk': revenue_at_risk,
            'clv_at_risk': clv_at_risk,
            'customer_replacement_cost': replacement_cost,
            'total_financial_impact': immediate_loss + future_loss + replacement_cost,
            'churned_customers_count': len(churned_customers),
            'high_risk_customers_count': len(df[df.get('churn_probability', 0) > 0.7]),
            'average_clv_churned': churned_customers.get('risk_adjusted_clv', churned_customers.get('predicted_future_clv', churned_customers['monthly_charges'] * 24)).mean()
        }
        
        return self.impact_metrics
    
    def analyze_churn_by_segments(self, df):
        """Analyze financial impact by customer segments"""
        segment_analysis = {}
        
        # Analyze by CLV segments
        if 'clv_segment' in df.columns:
            clv_impact = df.groupby('clv_segment').agg({
                'churned': ['count', 'sum', 'mean'],
                'monthly_charges': ['sum', 'mean'],
                'risk_adjusted_clv': 'sum' if 'risk_adjusted_clv' in df.columns else 'count',
                'clv_at_risk': 'sum' if 'clv_at_risk' in df.columns else 'count'
            }).round(2)
            segment_analysis['clv_segments'] = clv_impact
        
        # Analyze by contract type
        contract_impact = df.groupby('contract_type').agg({
            'churned': ['count', 'sum', 'mean'],
            'monthly_charges': ['sum', 'mean'],
            'risk_adjusted_clv': 'sum' if 'risk_adjusted_clv' in df.columns else 'count'
        }).round(2)
        segment_analysis['contract_type'] = contract_impact
        
        # Analyze by tenure categories
        if 'tenure_category' in df.columns:
            tenure_impact = df.groupby('tenure_category').agg({
                'churned': ['count', 'sum', 'mean'],
                'monthly_charges': ['sum', 'mean'],
                'risk_adjusted_clv': 'sum' if 'risk_adjusted_clv' in df.columns else 'count'
            }).round(2)
            segment_analysis['tenure_categories'] = tenure_impact
        
        return segment_analysis
    
    def calculate_retention_roi(self, df, intervention_costs):
        """Calculate ROI for different retention interventions"""
        if 'churn_probability' not in df.columns:
            print("Churn probability not available. Cannot calculate retention ROI.")
            return {}
        
        retention_scenarios = {}
        
        # Define intervention scenarios
        scenarios = {
            'basic_outreach': {
                'cost_per_customer': intervention_costs.get('basic_outreach', 50),
                'effectiveness': 0.15,  # 15% reduction in churn probability
                'target_probability': 0.5  # Target customers with >50% churn probability
            },
            'discount_offer': {
                'cost_per_customer': intervention_costs.get('discount_offer', 200),
                'effectiveness': 0.30,  # 30% reduction in churn probability
                'target_probability': 0.6  # Target customers with >60% churn probability
            },
            'premium_support': {
                'cost_per_customer': intervention_costs.get('premium_support', 150),
                'effectiveness': 0.25,  # 25% reduction in churn probability
                'target_probability': 0.7  # Target customers with >70% churn probability
            },
            'personalized_retention': {
                'cost_per_customer': intervention_costs.get('personalized_retention', 300),
                'effectiveness': 0.40,  # 40% reduction in churn probability
                'target_probability': 0.8  # Target customers with >80% churn probability
            }
        }
        
        for scenario_name, params in scenarios.items():
            # Identify target customers
            target_customers = df[df['churn_probability'] > params['target_probability']].copy()
            
            if len(target_customers) == 0:
                continue
            
            # Calculate intervention costs
            total_intervention_cost = len(target_customers) * params['cost_per_customer']
            
            # Calculate potential revenue saved
            # Reduced churn probability
            reduced_churn_prob = target_customers['churn_probability'] * (1 - params['effectiveness'])
            churn_reduction = target_customers['churn_probability'] - reduced_churn_prob
            
            # Revenue saved (CLV * churn reduction)
            if 'risk_adjusted_clv' in target_customers.columns:
                revenue_saved = (target_customers['risk_adjusted_clv'] * churn_reduction).sum()
            else:
                # Estimate based on monthly charges
                revenue_saved = (target_customers['monthly_charges'] * 24 * churn_reduction).sum()
            
            # Calculate ROI
            roi = ((revenue_saved - total_intervention_cost) / total_intervention_cost) * 100
            
            retention_scenarios[scenario_name] = {
                'target_customers': len(target_customers),
                'intervention_cost': total_intervention_cost,
                'revenue_saved': revenue_saved,
                'net_benefit': revenue_saved - total_intervention_cost,
                'roi_percentage': roi,
                'cost_per_customer': params['cost_per_customer'],
                'effectiveness': params['effectiveness'],
                'avg_customer_value': target_customers.get('risk_adjusted_clv', target_customers['monthly_charges'] * 24).mean()
            }
        
        self.retention_scenarios = retention_scenarios
        return retention_scenarios
    
    def prioritize_retention_efforts(self, df, budget_constraint=None):
        """Prioritize customers for retention efforts based on ROI"""
        if 'churn_probability' not in df.columns or 'risk_adjusted_clv' not in df.columns:
            print("Required columns not available for prioritization.")
            return pd.DataFrame()
        
        # Calculate retention priority score
        df_priority = df.copy()
        
        # Priority score = CLV * Churn Probability * (1 / typical intervention cost)
        typical_intervention_cost = 200  # Baseline cost
        df_priority['retention_priority_score'] = (
            df_priority['risk_adjusted_clv'] * 
            df_priority['churn_probability'] / 
            typical_intervention_cost
        )
        
        # Expected value of retention
        df_priority['expected_retention_value'] = (
            df_priority['risk_adjusted_clv'] * df_priority['churn_probability']
        )
        
        # Sort by priority score
        priority_customers = df_priority.sort_values('retention_priority_score', ascending=False)
        
        # Apply budget constraint if provided
        if budget_constraint:
            cumulative_cost = 0
            selected_customers = []
            
            for idx, customer in priority_customers.iterrows():
                intervention_cost = typical_intervention_cost
                if cumulative_cost + intervention_cost <= budget_constraint:
                    selected_customers.append(idx)
                    cumulative_cost += intervention_cost
                else:
                    break
            
            priority_customers = priority_customers.loc[selected_customers]
        
        return priority_customers[['customer_id', 'churn_probability', 'risk_adjusted_clv', 
                                 'retention_priority_score', 'expected_retention_value', 'clv_segment']]
    
    def plot_financial_impact(self, df):
        """Create visualizations for financial impact analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Revenue loss breakdown
        if self.impact_metrics:
            categories = ['Immediate Loss', 'Future Loss', 'Replacement Cost']
            values = [
                self.impact_metrics['immediate_revenue_loss'],
                self.impact_metrics['future_revenue_loss'],
                self.impact_metrics['customer_replacement_cost']
            ]
            
            axes[0, 0].pie(values, labels=categories, autopct='%1.1f%%', startangle=90)
            axes[0, 0].set_title('Revenue Loss Breakdown')
        
        # Churn rate by CLV segment
        if 'clv_segment' in df.columns:
            churn_by_segment = df.groupby('clv_segment')['churned'].mean()
            axes[0, 1].bar(range(len(churn_by_segment)), churn_by_segment.values, color='lightcoral')
            axes[0, 1].set_title('Churn Rate by CLV Segment')
            axes[0, 1].set_xlabel('CLV Segment')
            axes[0, 1].set_ylabel('Churn Rate')
            axes[0, 1].set_xticks(range(len(churn_by_segment)))
            axes[0, 1].set_xticklabels(churn_by_segment.index, rotation=45)
        
        # Revenue at risk by churn probability
        if 'churn_probability' in df.columns:
            # Create probability bins
            df['prob_bin'] = pd.cut(df['churn_probability'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            revenue_at_risk = df.groupby('prob_bin')['monthly_charges'].sum()
            
            axes[1, 0].bar(range(len(revenue_at_risk)), revenue_at_risk.values, color='orange')
            axes[1, 0].set_title('Monthly Revenue at Risk by Churn Probability')
            axes[1, 0].set_xlabel('Churn Probability Bin')
            axes[1, 0].set_ylabel('Monthly Revenue ($)')
            axes[1, 0].set_xticks(range(len(revenue_at_risk)))
            axes[1, 0].set_xticklabels(revenue_at_risk.index, rotation=45)
        
        # ROI comparison for retention scenarios
        if self.retention_scenarios:
            scenario_names = list(self.retention_scenarios.keys())
            roi_values = [self.retention_scenarios[name]['roi_percentage'] for name in scenario_names]
            
            colors = ['green' if roi > 0 else 'red' for roi in roi_values]
            axes[1, 1].bar(range(len(scenario_names)), roi_values, color=colors, alpha=0.7)
            axes[1, 1].set_title('ROI by Retention Strategy')
            axes[1, 1].set_xlabel('Retention Strategy')
            axes[1, 1].set_ylabel('ROI (%)')
            axes[1, 1].set_xticks(range(len(scenario_names)))
            axes[1, 1].set_xticklabels(scenario_names, rotation=45)
            axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('/q/bin/customer_churn_project/financial_impact.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_financial_report(self, df, intervention_costs=None):
        """Generate comprehensive financial impact report"""
        # Calculate basic financial impact
        impact_metrics = self.calculate_churn_financial_impact(df)
        
        # Analyze by segments
        segment_analysis = self.analyze_churn_by_segments(df)
        
        # Calculate retention ROI if intervention costs provided
        if intervention_costs:
            retention_roi = self.calculate_retention_roi(df, intervention_costs)
        else:
            # Use default costs
            default_costs = {
                'basic_outreach': 50,
                'discount_offer': 200,
                'premium_support': 150,
                'personalized_retention': 300
            }
            retention_roi = self.calculate_retention_roi(df, default_costs)
        
        # Generate priority customer list
        priority_customers = self.prioritize_retention_efforts(df, budget_constraint=50000)
        
        # Create visualizations
        self.plot_financial_impact(df)
        
        report = {
            'financial_impact': impact_metrics,
            'segment_analysis': segment_analysis,
            'retention_scenarios': retention_roi,
            'priority_customers': priority_customers,
            'recommendations': self._generate_recommendations(impact_metrics, retention_roi)
        }
        
        return report
    
    def _generate_recommendations(self, impact_metrics, retention_scenarios):
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # High-level financial recommendations
        if impact_metrics['total_financial_impact'] > 100000:
            recommendations.append("HIGH PRIORITY: Total financial impact exceeds $100K. Immediate retention action required.")
        
        # ROI-based recommendations
        best_roi_scenario = max(retention_scenarios.items(), key=lambda x: x[1]['roi_percentage'])
        if best_roi_scenario[1]['roi_percentage'] > 100:
            recommendations.append(f"RECOMMENDED: Implement '{best_roi_scenario[0]}' strategy with {best_roi_scenario[1]['roi_percentage']:.1f}% ROI.")
        
        # Budget allocation recommendations
        total_budget_needed = sum([scenario['intervention_cost'] for scenario in retention_scenarios.values()])
        recommendations.append(f"Budget allocation: ${total_budget_needed:,.0f} needed for comprehensive retention program.")
        
        # Customer prioritization
        if impact_metrics['high_risk_customers_count'] > 0:
            recommendations.append(f"Focus on {impact_metrics['high_risk_customers_count']} high-risk customers first.")
        
        return recommendations

if __name__ == "__main__":
    # Load data with CLV analysis
    try:
        df = pd.read_csv('/q/bin/customer_churn_project/data/processed/data_with_clv.csv')
    except FileNotFoundError:
        print("CLV analysis data not found. Running CLV analysis first...")
        from clv_analysis import CLVAnalyzer
        df = pd.read_csv('/q/bin/customer_churn_project/data/processed/processed_data.csv')
        clv_analyzer = CLVAnalyzer()
        df, _ = clv_analyzer.run_complete_clv_analysis(df)
    
    # Define intervention costs
    intervention_costs = {
        'basic_outreach': 50,      # Email campaigns, basic support
        'discount_offer': 200,     # Promotional discounts
        'premium_support': 150,    # Dedicated support agent
        'personalized_retention': 300  # Custom retention package
    }
    
    # Run financial impact analysis
    financial_analyzer = FinancialImpactAnalyzer()
    financial_report = financial_analyzer.generate_financial_report(df, intervention_costs)
    
    # Print key findings
    print("=== FINANCIAL IMPACT ANALYSIS ===")
    print(f"Total Revenue Loss: ${financial_report['financial_impact']['total_revenue_loss']:,.2f}")
    print(f"Revenue at Risk: ${financial_report['financial_impact']['revenue_at_risk']:,.2f}")
    print(f"Churned Customers: {financial_report['financial_impact']['churned_customers_count']:,}")
    print(f"High-Risk Customers: {financial_report['financial_impact']['high_risk_customers_count']:,}")
    
    print("\n=== RETENTION STRATEGY ROI ===")
    for strategy, metrics in financial_report['retention_scenarios'].items():
        print(f"{strategy}: ROI = {metrics['roi_percentage']:.1f}%, Net Benefit = ${metrics['net_benefit']:,.2f}")
    
    print("\n=== RECOMMENDATIONS ===")
    for rec in financial_report['recommendations']:
        print(f"• {rec}")
    
    print(f"\nTop 10 Priority Customers for Retention:")
    print(financial_report['priority_customers'].head(10))
    
    print("\nFinancial impact analysis complete!")
