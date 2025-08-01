"""
Interactive Streamlit dashboard for customer churn prediction and retention analysis.
Provides comprehensive visualization and insights for business stakeholders.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import DataPreprocessor
from churn_models import ChurnPredictor
from clv_analysis import CLVAnalyzer
from financial_impact import FinancialImpactAnalyzer
from retention_strategies import RetentionStrategyEngine

# Page configuration
st.set_page_config(
    page_title="Customer Churn Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .alert-low {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the processed data"""
    try:
        # Try to load fully processed data
        df = pd.read_csv('/q/bin/customer_churn_project/data/processed/data_with_retention_strategies.csv')
        return df, "full"
    except FileNotFoundError:
        try:
            # Try to load CLV data
            df = pd.read_csv('/q/bin/customer_churn_project/data/processed/data_with_clv.csv')
            return df, "clv"
        except FileNotFoundError:
            try:
                # Try to load basic processed data
                df = pd.read_csv('/q/bin/customer_churn_project/data/processed/processed_data.csv')
                return df, "basic"
            except FileNotFoundError:
                # Generate sample data
                preprocessor = DataPreprocessor()
                data_dict = preprocessor.process_data()
                return data_dict['full_data'], "generated"

def create_overview_metrics(df):
    """Create overview metrics for the dashboard"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = len(df)
        st.metric("Total Customers", f"{total_customers:,}")
    
    with col2:
        if 'churned' in df.columns:
            churn_rate = df['churned'].mean() * 100
            st.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
        else:
            st.metric("Overall Churn Rate", "N/A")
    
    with col3:
        if 'churn_probability' in df.columns:
            high_risk = len(df[df['churn_probability'] > 0.7])
            st.metric("High Risk Customers", f"{high_risk:,}")
        else:
            st.metric("High Risk Customers", "N/A")
    
    with col4:
        if 'risk_adjusted_clv' in df.columns:
            total_clv = df['risk_adjusted_clv'].sum()
            st.metric("Total CLV", f"${total_clv:,.0f}")
        elif 'predicted_future_clv' in df.columns:
            total_clv = df['predicted_future_clv'].sum()
            st.metric("Total CLV", f"${total_clv:,.0f}")
        else:
            st.metric("Total CLV", "N/A")

def create_churn_analysis_tab(df):
    """Create churn analysis visualizations"""
    st.header("🎯 Churn Prediction Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn probability distribution
        if 'churn_probability' in df.columns:
            fig = px.histogram(
                df, x='churn_probability', nbins=30,
                title="Distribution of Churn Probabilities",
                labels={'churn_probability': 'Churn Probability', 'count': 'Number of Customers'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Churn probability data not available. Please run the churn prediction model.")
    
    with col2:
        # Churn by contract type
        if 'contract_type' in df.columns and 'churned' in df.columns:
            churn_by_contract = df.groupby('contract_type')['churned'].agg(['count', 'sum', 'mean']).reset_index()
            churn_by_contract['churn_rate'] = churn_by_contract['mean'] * 100
            
            fig = px.bar(
                churn_by_contract, x='contract_type', y='churn_rate',
                title="Churn Rate by Contract Type",
                labels={'churn_rate': 'Churn Rate (%)', 'contract_type': 'Contract Type'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Risk segmentation
    if 'churn_probability' in df.columns:
        st.subheader("Customer Risk Segmentation")
        
        # Create risk categories
        df_risk = df.copy()
        df_risk['risk_category'] = pd.cut(
            df_risk['churn_probability'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        risk_summary = df_risk.groupby('risk_category').agg({
            'customer_id': 'count',
            'monthly_charges': 'sum',
            'churn_probability': 'mean'
        }).reset_index()
        risk_summary.columns = ['Risk Category', 'Customer Count', 'Total Monthly Revenue', 'Avg Churn Probability']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                risk_summary, values='Customer Count', names='Risk Category',
                title="Customer Distribution by Risk Level",
                color_discrete_map={'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                risk_summary, x='Risk Category', y='Total Monthly Revenue',
                title="Monthly Revenue by Risk Level",
                color='Risk Category',
                color_discrete_map={'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'}
            )
            st.plotly_chart(fig, use_container_width=True)

def create_clv_analysis_tab(df):
    """Create CLV analysis visualizations"""
    st.header("💰 Customer Lifetime Value Analysis")
    
    if 'risk_adjusted_clv' not in df.columns and 'predicted_future_clv' not in df.columns:
        st.info("CLV data not available. Please run the CLV analysis.")
        return
    
    clv_column = 'risk_adjusted_clv' if 'risk_adjusted_clv' in df.columns else 'predicted_future_clv'
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CLV distribution
        fig = px.histogram(
            df, x=clv_column, nbins=30,
            title="Distribution of Customer Lifetime Value",
            labels={clv_column: 'CLV ($)', 'count': 'Number of Customers'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # CLV vs Churn Probability scatter
        if 'churn_probability' in df.columns:
            fig = px.scatter(
                df, x='churn_probability', y=clv_column,
                title="CLV vs Churn Probability",
                labels={'churn_probability': 'Churn Probability', clv_column: 'CLV ($)'},
                opacity=0.6
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # CLV segments analysis
    if 'clv_segment' in df.columns:
        st.subheader("CLV Segment Analysis")
        
        segment_analysis = df.groupby('clv_segment').agg({
            'customer_id': 'count',
            clv_column: ['mean', 'sum'],
            'churn_probability': 'mean' if 'churn_probability' in df.columns else lambda x: 0
        }).round(2)
        
        # Flatten column names
        segment_analysis.columns = ['Customer Count', 'Avg CLV', 'Total CLV', 'Avg Churn Prob']
        segment_analysis = segment_analysis.reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                segment_analysis, x='clv_segment', y='Total CLV',
                title="Total CLV by Segment",
                labels={'clv_segment': 'CLV Segment', 'Total CLV': 'Total CLV ($)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'churn_probability' in df.columns:
                fig = px.bar(
                    segment_analysis, x='clv_segment', y='Avg Churn Prob',
                    title="Average Churn Probability by CLV Segment",
                    labels={'clv_segment': 'CLV Segment', 'Avg Churn Prob': 'Avg Churn Probability'}
                )
                st.plotly_chart(fig, use_container_width=True)

def create_financial_impact_tab(df):
    """Create financial impact analysis visualizations"""
    st.header("📈 Financial Impact Analysis")
    
    # Calculate basic financial metrics
    if 'churned' in df.columns:
        churned_customers = df[df['churned'] == 1]
        immediate_loss = churned_customers['monthly_charges'].sum()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Immediate Monthly Loss", f"${immediate_loss:,.2f}")
        
        with col2:
            if 'risk_adjusted_clv' in df.columns:
                future_loss = churned_customers['risk_adjusted_clv'].sum()
                st.metric("Future Revenue Loss", f"${future_loss:,.2f}")
            else:
                st.metric("Future Revenue Loss", "N/A")
        
        with col3:
            replacement_cost = immediate_loss * 5  # Assume 5x cost to replace
            st.metric("Replacement Cost", f"${replacement_cost:,.2f}")
    
    # Revenue at risk analysis
    if 'churn_probability' in df.columns and 'clv_at_risk' in df.columns:
        st.subheader("Revenue at Risk Analysis")
        
        # Create probability bins
        df_risk = df.copy()
        df_risk['prob_bin'] = pd.cut(
            df_risk['churn_probability'],
            bins=5,
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        risk_analysis = df_risk.groupby('prob_bin').agg({
            'customer_id': 'count',
            'monthly_charges': 'sum',
            'clv_at_risk': 'sum'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                risk_analysis, x='prob_bin', y='monthly_charges',
                title="Monthly Revenue at Risk by Churn Probability",
                labels={'prob_bin': 'Churn Probability Bin', 'monthly_charges': 'Monthly Revenue ($)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                risk_analysis, x='prob_bin', y='clv_at_risk',
                title="CLV at Risk by Churn Probability",
                labels={'prob_bin': 'Churn Probability Bin', 'clv_at_risk': 'CLV at Risk ($)'}
            )
            st.plotly_chart(fig, use_container_width=True)

def create_retention_strategies_tab(df):
    """Create retention strategies visualizations"""
    st.header("🎯 Retention Strategies")
    
    if 'retention_strategy' not in df.columns:
        st.info("Retention strategy data not available. Please run the retention analysis.")
        return
    
    # Strategy distribution
    strategy_counts = df[df['retention_strategy'] != 'no_action']['retention_strategy'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            values=strategy_counts.values,
            names=[name.replace('_', ' ').title() for name in strategy_counts.index],
            title="Distribution of Retention Strategies"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'strategy_priority' in df.columns:
            priority_counts = df[df['retention_strategy'] != 'no_action']['strategy_priority'].value_counts()
            colors = {'critical': 'red', 'high': 'orange', 'medium': 'yellow', 'low': 'green'}
            
            fig = px.bar(
                x=priority_counts.index,
                y=priority_counts.values,
                title="Customers by Strategy Priority",
                labels={'x': 'Priority Level', 'y': 'Number of Customers'},
                color=priority_counts.index,
                color_discrete_map=colors
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Cost vs Success Rate analysis
    if 'estimated_cost' in df.columns and 'expected_success_rate' in df.columns:
        st.subheader("Strategy Cost vs Success Rate Analysis")
        
        strategy_data = df[df['retention_strategy'] != 'no_action']
        
        fig = px.scatter(
            strategy_data, x='estimated_cost', y='expected_success_rate',
            color='retention_strategy',
            size='churn_probability' if 'churn_probability' in df.columns else None,
            title="Cost vs Expected Success Rate by Strategy",
            labels={'estimated_cost': 'Estimated Cost ($)', 'expected_success_rate': 'Expected Success Rate'}
        )
        st.plotly_chart(fig, use_container_width=True)

def create_customer_lookup_tab(df):
    """Create customer lookup and individual analysis"""
    st.header("🔍 Individual Customer Analysis")
    
    # Customer selection
    if 'customer_id' in df.columns:
        customer_ids = df['customer_id'].tolist()
        selected_customer = st.selectbox("Select Customer ID:", customer_ids)
        
        if selected_customer:
            customer_data = df[df['customer_id'] == selected_customer].iloc[0]
            
            # Customer overview
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Customer Profile")
                st.write(f"**Age:** {customer_data.get('age', 'N/A')}")
                st.write(f"**Tenure:** {customer_data.get('tenure_months', 'N/A')} months")
                st.write(f"**Contract:** {customer_data.get('contract_type', 'N/A')}")
                st.write(f"**Monthly Charges:** ${customer_data.get('monthly_charges', 0):.2f}")
            
            with col2:
                st.subheader("Risk Assessment")
                if 'churn_probability' in customer_data:
                    churn_prob = customer_data['churn_probability']
                    st.metric("Churn Probability", f"{churn_prob:.1%}")
                    
                    if churn_prob > 0.7:
                        st.markdown('<div class="alert-high">🚨 HIGH RISK CUSTOMER</div>', unsafe_allow_html=True)
                    elif churn_prob > 0.3:
                        st.markdown('<div class="alert-medium">⚠️ MEDIUM RISK CUSTOMER</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="alert-low">✅ LOW RISK CUSTOMER</div>', unsafe_allow_html=True)
            
            with col3:
                st.subheader("Value Assessment")
                if 'risk_adjusted_clv' in customer_data:
                    clv = customer_data['risk_adjusted_clv']
                    st.metric("Customer Lifetime Value", f"${clv:,.2f}")
                elif 'predicted_future_clv' in customer_data:
                    clv = customer_data['predicted_future_clv']
                    st.metric("Predicted CLV", f"${clv:,.2f}")
                
                if 'clv_segment' in customer_data:
                    st.write(f"**CLV Segment:** {customer_data['clv_segment']}")
            
            # Retention recommendation
            if 'retention_strategy' in customer_data:
                st.subheader("Retention Recommendation")
                strategy = customer_data['retention_strategy']
                
                if strategy != 'no_action':
                    st.write(f"**Recommended Strategy:** {strategy.replace('_', ' ').title()}")
                    
                    if 'strategy_priority' in customer_data:
                        priority = customer_data['strategy_priority']
                        st.write(f"**Priority Level:** {priority.title()}")
                    
                    if 'estimated_cost' in customer_data:
                        cost = customer_data['estimated_cost']
                        st.write(f"**Estimated Cost:** ${cost:.2f}")
                    
                    if 'expected_success_rate' in customer_data:
                        success_rate = customer_data['expected_success_rate']
                        st.write(f"**Expected Success Rate:** {success_rate:.1%}")
                    
                    if 'primary_actions' in customer_data:
                        actions = customer_data['primary_actions']
                        st.write(f"**Recommended Actions:** {actions}")
                else:
                    st.info("No specific retention action recommended for this customer.")

def main():
    """Main dashboard function"""
    st.markdown('<h1 class="main-header">Customer Churn Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        df, data_status = load_data()
    
    # Data status indicator
    status_messages = {
        "full": "✅ Complete analysis data loaded",
        "clv": "⚠️ CLV analysis data loaded (retention strategies not available)",
        "basic": "⚠️ Basic processed data loaded (CLV and retention analysis not available)",
        "generated": "ℹ️ Using generated sample data"
    }
    st.info(status_messages[data_status])
    
    # Overview metrics
    create_overview_metrics(df)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "🎯 Churn Analysis", "💰 CLV Analysis", 
        "📈 Financial Impact", "🎯 Retention Strategies", "🔍 Customer Lookup"
    ])
    
    with tab1:
        st.header("📊 Business Overview")
        
        # Key insights
        st.subheader("Key Insights")
        
        insights = []
        
        if 'churned' in df.columns:
            churn_rate = df['churned'].mean() * 100
            insights.append(f"Current churn rate is {churn_rate:.1f}%")
        
        if 'churn_probability' in df.columns:
            high_risk_count = len(df[df['churn_probability'] > 0.7])
            insights.append(f"{high_risk_count:,} customers are at high risk of churning")
        
        if 'clv_at_risk' in df.columns:
            clv_at_risk = df['clv_at_risk'].sum()
            insights.append(f"${clv_at_risk:,.2f} in customer lifetime value is at risk")
        
        for insight in insights:
            st.write(f"• {insight}")
        
        # Data summary table
        st.subheader("Data Summary")
        summary_stats = df.describe()
        st.dataframe(summary_stats)
    
    with tab2:
        create_churn_analysis_tab(df)
    
    with tab3:
        create_clv_analysis_tab(df)
    
    with tab4:
        create_financial_impact_tab(df)
    
    with tab5:
        create_retention_strategies_tab(df)
    
    with tab6:
        create_customer_lookup_tab(df)
    
    # Sidebar with additional controls
    st.sidebar.header("Dashboard Controls")
    
    # Data refresh button
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()
    
    # Export options
    st.sidebar.header("Export Options")
    
    if st.sidebar.button("Download Customer Data"):
        csv = df.to_csv(index=False)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name="customer_data.csv",
            mime="text/csv"
        )
    
    # Model information
    st.sidebar.header("Model Information")
    st.sidebar.info(
        "This dashboard uses machine learning models to predict customer churn, "
        "calculate lifetime value, and recommend retention strategies. "
        "All predictions should be validated with business context."
    )

if __name__ == "__main__":
    main()
