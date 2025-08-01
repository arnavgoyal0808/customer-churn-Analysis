"""
Retention strategies module for data-driven customer retention recommendations.
Provides personalized retention strategies based on customer profiles and risk factors.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class RetentionStrategyEngine:
    def __init__(self):
        self.customer_clusters = None
        self.strategy_templates = self._initialize_strategy_templates()
        self.personalized_strategies = {}
        
    def _initialize_strategy_templates(self):
        """Initialize retention strategy templates"""
        return {
            'price_sensitive': {
                'primary_actions': ['discount_offer', 'loyalty_program', 'payment_plan'],
                'secondary_actions': ['usage_optimization', 'service_downgrade'],
                'communication_channel': 'email',
                'urgency': 'medium',
                'cost_range': (50, 200),
                'expected_effectiveness': 0.25
            },
            'service_dissatisfied': {
                'primary_actions': ['premium_support', 'service_upgrade', 'technical_assistance'],
                'secondary_actions': ['training_session', 'dedicated_account_manager'],
                'communication_channel': 'phone',
                'urgency': 'high',
                'cost_range': (100, 300),
                'expected_effectiveness': 0.35
            },
            'high_value_at_risk': {
                'primary_actions': ['personalized_retention', 'executive_outreach', 'custom_package'],
                'secondary_actions': ['exclusive_benefits', 'priority_support'],
                'communication_channel': 'in_person',
                'urgency': 'critical',
                'cost_range': (200, 500),
                'expected_effectiveness': 0.45
            },
            'new_customer_unstable': {
                'primary_actions': ['onboarding_support', 'welcome_package', 'early_engagement'],
                'secondary_actions': ['tutorial_sessions', 'check_in_calls'],
                'communication_channel': 'phone',
                'urgency': 'high',
                'cost_range': (75, 150),
                'expected_effectiveness': 0.30
            },
            'contract_ending': {
                'primary_actions': ['renewal_incentive', 'contract_upgrade', 'loyalty_discount'],
                'secondary_actions': ['flexible_terms', 'service_review'],
                'communication_channel': 'email',
                'urgency': 'medium',
                'cost_range': (100, 250),
                'expected_effectiveness': 0.40
            },
            'low_engagement': {
                'primary_actions': ['engagement_campaign', 'feature_education', 'usage_incentives'],
                'secondary_actions': ['gamification', 'community_access'],
                'communication_channel': 'app_notification',
                'urgency': 'low',
                'cost_range': (25, 100),
                'expected_effectiveness': 0.20
            }
        }
    
    def segment_customers_for_retention(self, df):
        """Segment customers based on churn risk factors for targeted retention"""
        # Prepare features for clustering
        feature_columns = [
            'churn_probability', 'monthly_charges', 'tenure_months', 
            'support_calls', 'late_payments', 'service_complexity'
        ]
        
        # Add CLV if available
        if 'risk_adjusted_clv' in df.columns:
            feature_columns.append('risk_adjusted_clv')
        elif 'predicted_future_clv' in df.columns:
            feature_columns.append('predicted_future_clv')
        
        # Filter for at-risk customers (churn probability > 0.3)
        if 'churn_probability' in df.columns:
            at_risk_customers = df[df['churn_probability'] > 0.3].copy()
        else:
            at_risk_customers = df.copy()
        
        if len(at_risk_customers) == 0:
            print("No at-risk customers found for segmentation.")
            return df
        
        # Prepare data for clustering
        X = at_risk_customers[feature_columns].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        n_clusters = min(6, len(at_risk_customers) // 10)  # Adaptive number of clusters
        if n_clusters < 2:
            n_clusters = 2
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        at_risk_customers['retention_segment'] = cluster_labels
        
        # Merge back to main dataframe
        df = df.merge(
            at_risk_customers[['customer_id', 'retention_segment']], 
            on='customer_id', 
            how='left'
        )
        df['retention_segment'].fillna(-1, inplace=True)  # -1 for low-risk customers
        
        # Analyze clusters
        cluster_analysis = at_risk_customers.groupby('retention_segment').agg({
            'churn_probability': ['mean', 'std'],
            'monthly_charges': ['mean', 'std'],
            'tenure_months': ['mean', 'std'],
            'support_calls': 'mean',
            'late_payments': 'mean',
            feature_columns[-1]: 'mean' if len(feature_columns) > 6 else lambda x: 0  # CLV column
        }).round(2)
        
        self.customer_clusters = cluster_analysis
        return df
    
    def assign_retention_strategies(self, df):
        """Assign specific retention strategies to customers based on their profiles"""
        df_strategies = df.copy()
        df_strategies['retention_strategy'] = 'no_action'
        df_strategies['strategy_priority'] = 'low'
        df_strategies['estimated_cost'] = 0
        df_strategies['expected_success_rate'] = 0
        
        # Only process at-risk customers
        at_risk_mask = df_strategies.get('churn_probability', 0) > 0.3
        
        for idx, customer in df_strategies[at_risk_mask].iterrows():
            strategy = self._determine_customer_strategy(customer)
            
            df_strategies.loc[idx, 'retention_strategy'] = strategy['name']
            df_strategies.loc[idx, 'strategy_priority'] = strategy['priority']
            df_strategies.loc[idx, 'estimated_cost'] = strategy['cost']
            df_strategies.loc[idx, 'expected_success_rate'] = strategy['success_rate']
            df_strategies.loc[idx, 'primary_actions'] = ', '.join(strategy['actions'])
            df_strategies.loc[idx, 'communication_channel'] = strategy['channel']
        
        return df_strategies
    
    def _determine_customer_strategy(self, customer):
        """Determine the best retention strategy for a specific customer"""
        churn_prob = customer.get('churn_probability', 0)
        monthly_charges = customer.get('monthly_charges', 0)
        tenure = customer.get('tenure_months', 0)
        support_calls = customer.get('support_calls', 0)
        clv = customer.get('risk_adjusted_clv', customer.get('predicted_future_clv', monthly_charges * 24))
        contract_type = customer.get('contract_type', 'Month-to-month')
        
        # Decision logic for strategy assignment
        if clv > 2000 and churn_prob > 0.7:
            # High-value, high-risk customers
            strategy_template = self.strategy_templates['high_value_at_risk']
            strategy_name = 'high_value_at_risk'
            priority = 'critical'
            
        elif tenure < 12 and churn_prob > 0.6:
            # New customers at risk
            strategy_template = self.strategy_templates['new_customer_unstable']
            strategy_name = 'new_customer_unstable'
            priority = 'high'
            
        elif support_calls > 3 and churn_prob > 0.5:
            # Service-dissatisfied customers
            strategy_template = self.strategy_templates['service_dissatisfied']
            strategy_name = 'service_dissatisfied'
            priority = 'high'
            
        elif contract_type == 'Month-to-month' and churn_prob > 0.6:
            # Contract ending or flexible contract customers
            strategy_template = self.strategy_templates['contract_ending']
            strategy_name = 'contract_ending'
            priority = 'medium'
            
        elif monthly_charges > 80 and churn_prob > 0.5:
            # Price-sensitive customers
            strategy_template = self.strategy_templates['price_sensitive']
            strategy_name = 'price_sensitive'
            priority = 'medium'
            
        else:
            # Low engagement or general at-risk
            strategy_template = self.strategy_templates['low_engagement']
            strategy_name = 'low_engagement'
            priority = 'low'
        
        # Calculate cost based on customer value
        base_cost = np.mean(strategy_template['cost_range'])
        if clv > 1500:
            cost_multiplier = 1.5
        elif clv > 800:
            cost_multiplier = 1.2
        else:
            cost_multiplier = 0.8
        
        estimated_cost = base_cost * cost_multiplier
        
        # Adjust success rate based on customer characteristics
        base_success_rate = strategy_template['expected_effectiveness']
        if tenure > 24:  # Loyal customers more likely to respond
            success_rate = base_success_rate * 1.2
        elif tenure < 6:  # New customers less predictable
            success_rate = base_success_rate * 0.9
        else:
            success_rate = base_success_rate
        
        success_rate = min(success_rate, 0.8)  # Cap at 80%
        
        return {
            'name': strategy_name,
            'priority': priority,
            'cost': estimated_cost,
            'success_rate': success_rate,
            'actions': strategy_template['primary_actions'],
            'channel': strategy_template['communication_channel']
        }
    
    def optimize_retention_budget(self, df, total_budget):
        """Optimize retention budget allocation across customers"""
        # Filter customers with retention strategies
        customers_with_strategies = df[df['retention_strategy'] != 'no_action'].copy()
        
        if len(customers_with_strategies) == 0:
            print("No customers with retention strategies found.")
            return pd.DataFrame()
        
        # Calculate expected ROI for each customer
        customers_with_strategies['expected_revenue_saved'] = (
            customers_with_strategies.get('risk_adjusted_clv', customers_with_strategies['monthly_charges'] * 24) *
            customers_with_strategies['churn_probability'] *
            customers_with_strategies['expected_success_rate']
        )
        
        customers_with_strategies['expected_roi'] = (
            (customers_with_strategies['expected_revenue_saved'] - customers_with_strategies['estimated_cost']) /
            customers_with_strategies['estimated_cost']
        )
        
        # Sort by ROI and priority
        priority_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        customers_with_strategies['priority_score'] = customers_with_strategies['strategy_priority'].map(priority_order)
        
        # Sort by priority first, then by ROI
        customers_sorted = customers_with_strategies.sort_values(
            ['priority_score', 'expected_roi'], 
            ascending=[False, False]
        )
        
        # Allocate budget
        selected_customers = []
        remaining_budget = total_budget
        
        for idx, customer in customers_sorted.iterrows():
            if customer['estimated_cost'] <= remaining_budget:
                selected_customers.append(idx)
                remaining_budget -= customer['estimated_cost']
        
        budget_allocation = customers_sorted.loc[selected_customers]
        
        # Summary statistics
        allocation_summary = {
            'total_customers_selected': len(selected_customers),
            'total_budget_used': total_budget - remaining_budget,
            'remaining_budget': remaining_budget,
            'expected_total_revenue_saved': budget_allocation['expected_revenue_saved'].sum(),
            'expected_total_roi': ((budget_allocation['expected_revenue_saved'].sum() - 
                                  budget_allocation['estimated_cost'].sum()) / 
                                 budget_allocation['estimated_cost'].sum()) * 100,
            'customers_by_priority': budget_allocation['strategy_priority'].value_counts().to_dict()
        }
        
        return budget_allocation, allocation_summary
    
    def create_retention_campaigns(self, df):
        """Create specific retention campaigns based on customer segments"""
        campaigns = {}
        
        # Group customers by retention strategy
        strategy_groups = df[df['retention_strategy'] != 'no_action'].groupby('retention_strategy')
        
        for strategy_name, customers in strategy_groups:
            if len(customers) == 0:
                continue
                
            campaign = {
                'campaign_name': f"{strategy_name.replace('_', ' ').title()} Retention Campaign",
                'target_customers': len(customers),
                'total_budget': customers['estimated_cost'].sum(),
                'expected_saves': (customers['expected_success_rate'] * len(customers)).sum(),
                'expected_revenue_impact': customers['expected_revenue_saved'].sum(),
                'primary_channel': customers['communication_channel'].mode().iloc[0] if len(customers) > 0 else 'email',
                'timeline': self._determine_campaign_timeline(strategy_name),
                'kpis': self._define_campaign_kpis(strategy_name),
                'customer_list': customers[['customer_id', 'churn_probability', 'estimated_cost', 
                                          'expected_success_rate', 'primary_actions']].to_dict('records')
            }
            
            campaigns[strategy_name] = campaign
        
        return campaigns
    
    def _determine_campaign_timeline(self, strategy_name):
        """Determine campaign timeline based on strategy type"""
        timelines = {
            'high_value_at_risk': '1-2 weeks (urgent)',
            'service_dissatisfied': '1-3 weeks',
            'new_customer_unstable': '2-4 weeks',
            'contract_ending': '4-6 weeks (before renewal)',
            'price_sensitive': '2-4 weeks',
            'low_engagement': '4-8 weeks (ongoing)'
        }
        return timelines.get(strategy_name, '2-4 weeks')
    
    def _define_campaign_kpis(self, strategy_name):
        """Define KPIs for each campaign type"""
        base_kpis = ['churn_reduction_rate', 'campaign_response_rate', 'cost_per_save', 'roi']
        
        strategy_specific_kpis = {
            'high_value_at_risk': base_kpis + ['executive_engagement_rate', 'custom_package_acceptance'],
            'service_dissatisfied': base_kpis + ['support_satisfaction_improvement', 'issue_resolution_rate'],
            'new_customer_unstable': base_kpis + ['onboarding_completion_rate', 'early_engagement_score'],
            'contract_ending': base_kpis + ['renewal_rate', 'contract_upgrade_rate'],
            'price_sensitive': base_kpis + ['discount_acceptance_rate', 'payment_plan_adoption'],
            'low_engagement': base_kpis + ['feature_adoption_rate', 'usage_increase']
        }
        
        return strategy_specific_kpis.get(strategy_name, base_kpis)
    
    def plot_retention_analysis(self, df, campaigns):
        """Create visualizations for retention strategy analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Strategy distribution
        strategy_counts = df[df['retention_strategy'] != 'no_action']['retention_strategy'].value_counts()
        axes[0, 0].pie(strategy_counts.values, labels=strategy_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Distribution of Retention Strategies')
        
        # Cost vs Expected ROI
        strategy_data = df[df['retention_strategy'] != 'no_action']
        if len(strategy_data) > 0:
            scatter = axes[0, 1].scatter(strategy_data['estimated_cost'], 
                                       strategy_data['expected_success_rate'] * 100,
                                       c=strategy_data['churn_probability'], 
                                       cmap='Reds', alpha=0.6)
            axes[0, 1].set_xlabel('Estimated Cost ($)')
            axes[0, 1].set_ylabel('Expected Success Rate (%)')
            axes[0, 1].set_title('Cost vs Success Rate (colored by churn probability)')
            plt.colorbar(scatter, ax=axes[0, 1])
        
        # Campaign budget allocation
        if campaigns:
            campaign_names = list(campaigns.keys())
            campaign_budgets = [campaigns[name]['total_budget'] for name in campaign_names]
            
            axes[1, 0].bar(range(len(campaign_names)), campaign_budgets, color='skyblue')
            axes[1, 0].set_title('Budget Allocation by Campaign')
            axes[1, 0].set_xlabel('Campaign')
            axes[1, 0].set_ylabel('Budget ($)')
            axes[1, 0].set_xticks(range(len(campaign_names)))
            axes[1, 0].set_xticklabels([name.replace('_', ' ').title() for name in campaign_names], rotation=45)
        
        # Priority distribution
        if 'strategy_priority' in df.columns:
            priority_counts = df[df['retention_strategy'] != 'no_action']['strategy_priority'].value_counts()
            colors = {'critical': 'red', 'high': 'orange', 'medium': 'yellow', 'low': 'green'}
            bar_colors = [colors.get(priority, 'gray') for priority in priority_counts.index]
            
            axes[1, 1].bar(range(len(priority_counts)), priority_counts.values, color=bar_colors)
            axes[1, 1].set_title('Customers by Strategy Priority')
            axes[1, 1].set_xlabel('Priority Level')
            axes[1, 1].set_ylabel('Number of Customers')
            axes[1, 1].set_xticks(range(len(priority_counts)))
            axes[1, 1].set_xticklabels(priority_counts.index)
        
        plt.tight_layout()
        plt.savefig('/q/bin/customer_churn_project/retention_strategies.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_retention_report(self, df, total_budget=50000):
        """Generate comprehensive retention strategy report"""
        print("Segmenting customers for retention...")
        df_segmented = self.segment_customers_for_retention(df)
        
        print("Assigning retention strategies...")
        df_with_strategies = self.assign_retention_strategies(df_segmented)
        
        print("Optimizing budget allocation...")
        budget_allocation, allocation_summary = self.optimize_retention_budget(df_with_strategies, total_budget)
        
        print("Creating retention campaigns...")
        campaigns = self.create_retention_campaigns(df_with_strategies)
        
        print("Generating visualizations...")
        self.plot_retention_analysis(df_with_strategies, campaigns)
        
        # Save results
        df_with_strategies.to_csv('/q/bin/customer_churn_project/data/processed/data_with_retention_strategies.csv', index=False)
        budget_allocation.to_csv('/q/bin/customer_churn_project/data/processed/budget_allocation.csv', index=False)
        
        report = {
            'customer_data': df_with_strategies,
            'budget_allocation': budget_allocation,
            'allocation_summary': allocation_summary,
            'campaigns': campaigns,
            'cluster_analysis': self.customer_clusters
        }
        
        return report

if __name__ == "__main__":
    # Load data with financial analysis
    try:
        df = pd.read_csv('/q/bin/customer_churn_project/data/processed/data_with_clv.csv')
    except FileNotFoundError:
        print("CLV analysis data not found. Please run CLV analysis first.")
        exit(1)
    
    # Add customer IDs if not present
    if 'customer_id' not in df.columns:
        df['customer_id'] = [f"CUST_{i:06d}" for i in range(1, len(df) + 1)]
    
    # Run retention strategy analysis
    retention_engine = RetentionStrategyEngine()
    retention_report = retention_engine.generate_retention_report(df, total_budget=50000)
    
    # Print summary
    print("\n=== RETENTION STRATEGY ANALYSIS ===")
    print(f"Total customers analyzed: {len(df):,}")
    print(f"At-risk customers: {len(df[df.get('churn_probability', 0) > 0.3]):,}")
    print(f"Customers with retention strategies: {len(df[df.get('retention_strategy', 'no_action') != 'no_action']):,}")
    
    print(f"\n=== BUDGET ALLOCATION SUMMARY ===")
    summary = retention_report['allocation_summary']
    print(f"Selected customers: {summary['total_customers_selected']:,}")
    print(f"Budget used: ${summary['total_budget_used']:,.2f}")
    print(f"Expected revenue saved: ${summary['expected_total_revenue_saved']:,.2f}")
    print(f"Expected ROI: {summary['expected_total_roi']:.1f}%")
    
    print(f"\n=== RETENTION CAMPAIGNS ===")
    for campaign_name, campaign in retention_report['campaigns'].items():
        print(f"{campaign['campaign_name']}: {campaign['target_customers']} customers, ${campaign['total_budget']:,.0f} budget")
    
    print("\nRetention strategy analysis complete!")
