import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="UK Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #FFFBEB;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 3px solid #F59E0B;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/best_churn_model.joblib')
        preprocessor_dict = joblib.load('models/preprocessor.joblib')  # now a dict
        return model, preprocessor_dict
    except Exception as e:
        st.error(f"Error loading files: {e}")
        st.error("Please run main.py first to generate the model and preprocessor.")
        return None, None

@st.cache_data
def load_sample_data():
    """Load sample customer data"""
    try:
        df = pd.read_csv('data/uk_customers.csv')
        return df
    except:
        st.error("Data file not found. Please generate data first.")
        return None

# def predict_churn(customer_data, model, preprocessor):
#     """Predict churn for a customer"""
#     # Preprocess the input data
#     X_processed, _ = preprocessor.preprocess(customer_data)
    
#     # Make prediction
#     churn_prob = model.predict_proba(X_processed)[0, 1]
#     churn_pred = 1 if churn_prob > 0.5 else 0
    
#     return churn_pred, churn_prob

def predict_churn(customer_data, model, preprocessor_dict):
    """Predict churn using manually applied preprocessing"""
    df = customer_data.copy()
    
    # === 1. Drop unnecessary columns ===
    cols_to_drop = ['customer_id', 'products_held', 'churn_reason']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    # === 2. Create engineered features (same as in training) ===
    df['income_per_product'] = df['annual_income'] / (df['num_products'] + 1)
    df['engagement_score'] = df['app_usage_hours'] / (df['days_since_last_login'] + 1)
    df['risk_score'] = (df['complaints_last_year'] * 0.3 +
                       (850 - df['credit_score']) / 550 * 0.7)
    df['total_monthly_value'] = df['avg_transaction_value'] * df['transaction_frequency'] / 30
    
    df['age_group'] = pd.cut(df['age'],
                             bins=[0, 25, 35, 45, 55, 65, 100],
                             labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
    df['tenure_group'] = pd.cut(df['tenure_months'],
                                bins=[0, 12, 36, 60, 120, 240],
                                labels=['lt_1yr', '1_3yr', '3_5yr', '5_10yr', '10plus_yr'])  # safer labels
    
    # === 3. Encode categorical variables ===
    # Gender
    if 'gender' in df.columns and 'gender' in preprocessor_dict['label_encoders']:
        le = preprocessor_dict['label_encoders']['gender']
        df['gender'] = le.transform(df['gender'])
    
    # Region one-hot
    if 'region' in df.columns:
        region_dummies = pd.get_dummies(df['region'], prefix='region')
        df = pd.concat([df, region_dummies], axis=1)
        df.drop('region', axis=1, inplace=True)
    
    # Age and tenure groups one-hot
    for col in ['age_group', 'tenure_group']:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            df.drop(col, axis=1, inplace=True)
    
    # Clean column names for XGBoost safety
    df.columns = df.columns.str.replace(r'[<>[\],]', '_', regex=True)
    
    # === 4. Scale numerical features ===
    scaler = preprocessor_dict['scaler']
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    # === 5. Predict ===
    churn_prob = model.predict_proba(df)[0, 1]
    churn_pred = 1 if churn_prob > 0.5 else 0
    
    return churn_pred, churn_prob

def create_customer_input_form():
    """Create form for customer input"""
    st.sidebar.header("📝 Customer Information")
    
    with st.sidebar.form("customer_form"):
        age = st.slider("Age", 18, 85, 35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        region = st.selectbox("Region", [
            "London", "South East", "North West", "South West", 
            "West Midlands", "East Midlands", "Yorkshire",
            "East of England", "Scotland", "Wales", "Northern Ireland"
        ])
        
        col1, col2 = st.columns(2)
        with col1:
            tenure_months = st.number_input("Tenure (months)", 1, 240, 24)
            credit_score = st.number_input("Credit Score", 300, 850, 650)
        with col2:
            annual_income = st.number_input("Annual Income (£)", 15000, 150000, 35000)
            num_products = st.slider("Number of Products", 1, 5, 2)
        
        days_since_last_login = st.number_input("Days Since Last Login", 0, 365, 7)
        complaints_last_year = st.number_input("Complaints Last Year", 0, 10, 0)
        transaction_frequency = st.number_input("Monthly Transactions", 0, 100, 15)
        
        submitted = st.form_submit_button("Predict Churn Risk")
    
    if submitted:
        # Create customer dictionary
        customer = {
            'customer_id': 'NEW_CUST',
            'age': age,
            'gender': gender,
            'region': region,
            'tenure_months': tenure_months,
            'account_age_days': tenure_months * 30,
            'credit_score': credit_score,
            'annual_income': annual_income,
            'num_products': num_products,
            'products_held': 'Current Account,Savings Account',
            'has_current_account': 1,
            'has_savings_account': 1,
            'has_credit_card': 1 if num_products > 2 else 0,
            'has_mortgage': 1 if num_products > 3 else 0,
            'avg_transaction_value': 150,
            'transaction_frequency': transaction_frequency,
            'complaints_last_year': complaints_last_year,
            'days_since_last_login': days_since_last_login,
            'app_usage_hours': 5,
            'estimated_clv': annual_income * 0.05,
            # 'churned': 0,
            'churn_probability': 0,
            'churn_reason': None,
            'days_since_churn': 0
        }
        
        return pd.DataFrame([customer])
    
    return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🏦 UK Customer Churn Prediction Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Load model and data
    model, preprocessor = load_model()
    df = load_sample_data()
    
    if model is None or df is None:
        st.stop()
    
    # Sidebar
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2721/2721264.png", 
                    width=100)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", 
                           ["📈 Overview", "🔍 Predict Churn", "📊 Customer Insights", 
                            "🎯 Retention Strategies"])
    
    if page == "📈 Overview":
        display_overview(df)
    elif page == "🔍 Predict Churn":
        display_prediction(model, preprocessor, df)
    elif page == "📊 Customer Insights":
        display_insights(df)
    elif page == "🎯 Retention Strategies":
        display_strategies(df)

def display_overview(df):
    """Display overview dashboard"""
    
    st.markdown('<h2 class="sub-header">📊 Business Overview</h2>', 
                unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    with col2:
        churn_rate = df['churned'].mean()
        st.metric("Churn Rate", f"{churn_rate:.2%}")
    with col3:
        avg_clv = df['estimated_clv'].mean()
        st.metric("Average CLV", f"£{avg_clv:,.0f}")
    with col4:
        high_risk = len(df[df['days_since_last_login'] > 30])
        st.metric("High Risk Customers", f"{high_risk:,}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Regional Churn Analysis")
        regional_churn = df.groupby('region')['churned'].mean().reset_index()
        regional_churn = regional_churn.sort_values('churned', ascending=False)
        
        fig = px.bar(regional_churn, x='region', y='churned',
                    color='churned', color_continuous_scale='Reds',
                    title="Churn Rate by UK Region")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Customer Segmentation")
        df['risk_category'] = pd.cut(df['credit_score'], 
                                    bins=[0, 580, 670, 740, 850],
                                    labels=['Poor', 'Fair', 'Good', 'Excellent'])
        
        risk_churn = df.groupby('risk_category')['churned'].mean().reset_index()
        
        fig = px.pie(risk_churn, values='churned', names='risk_category',
                    title="Churn Distribution by Credit Risk",
                    color_discrete_sequence=px.colors.sequential.RdBu)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Customer demographics
    st.markdown('<h3 class="sub-header">👥 Customer Demographics</h3>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        fig = px.histogram(df, x='age', nbins=20, 
                          title="Age Distribution",
                          color_discrete_sequence=['#3B82F6'])
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Product holdings
        product_cols = ['has_current_account', 'has_savings_account', 
                       'has_credit_card', 'has_mortgage']
        product_counts = df[product_cols].sum().reset_index()
        product_counts.columns = ['Product', 'Count']
        
        fig = px.bar(product_counts, x='Product', y='Count',
                    title="Product Holdings Distribution",
                    color_discrete_sequence=['#10B981'])
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def display_prediction(model, preprocessor, df):
    """Display churn prediction interface"""
    
    st.markdown('<h2 class="sub-header">🔍 Churn Prediction</h2>', 
                unsafe_allow_html=True)
    
    # Get customer input
    customer_df = create_customer_input_form()
    
    if customer_df is not None:
        # Make prediction
        churn_pred, churn_prob = predict_churn(customer_df, model, preprocessor)
        
        # Display results
        st.markdown("### Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if churn_pred == 1:
                st.error("🚨 High Churn Risk")
            else:
                st.success("✅ Low Churn Risk")
        
        with col2:
            st.metric("Churn Probability", f"{churn_prob:.2%}")
        
        with col3:
            risk_level = "High" if churn_prob > 0.7 else "Medium" if churn_prob > 0.3 else "Low"
            st.metric("Risk Level", risk_level)
        
        # Probability gauge
        st.markdown("#### Churn Probability Gauge")
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = churn_prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Probability (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "green"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("#### 📋 Retention Recommendations")
        
        if churn_prob > 0.7:
            st.markdown("""
            <div class="highlight">
            <strong>Immediate Action Required:</strong>
            <ul>
                <li>📞 Personal phone call from relationship manager</li>
                <li>🎁 Exclusive retention offer (e.g., fee waiver for 6 months)</li>
                <li>📱 Schedule in-person meeting at local branch</li>
                <li>💳 Review product fit and offer upgrades</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        elif churn_prob > 0.3:
            st.markdown("""
            <div class="highlight">
            <strong>Proactive Engagement:</strong>
            <ul>
                <li>✉️ Personalized email with relevant offers</li>
                <li>📱 App notification with new feature highlights</li>
                <li>🤝 Invitation to customer webinar</li>
                <li>⭐ Loyalty points bonus offer</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="highlight">
            <strong>Maintenance Strategy:</strong>
            <ul>
                <li>📧 Regular newsletter with useful content</li>
                <li>🎯 Cross-sell complementary products</li>
                <li>⭐ Continue excellent service delivery</li>
                <li>📊 Monitor engagement metrics monthly</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

def display_insights(df):
    """Display customer insights"""
    
    st.markdown('<h2 class="sub-header">📊 Customer Insights & Analytics</h2>', 
                unsafe_allow_html=True)
    
    # Filter options
    st.sidebar.markdown("### 🔍 Filter Options")
    
    region_filter = st.sidebar.multiselect(
        "Select Regions",
        df['region'].unique(),
        default=df['region'].unique()[:3]
    )
    
    age_range = st.sidebar.slider(
        "Age Range",
        int(df['age'].min()), int(df['age'].max()),
        (25, 55)
    )
    
    # Apply filters
    filtered_df = df[
        (df['region'].isin(region_filter)) &
        (df['age'].between(age_range[0], age_range[1]))
    ]
    
    # Insights tabs
    tab1, tab2, tab3 = st.tabs(["📈 Behavioral Patterns", "💰 Financial Profile", "🎯 Churn Drivers"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Engagement vs Churn
            fig = px.scatter(filtered_df, x='days_since_last_login', y='app_usage_hours',
                           color='churned', title="Engagement vs Churn",
                           color_discrete_sequence=['green', 'red'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Complaint analysis
            complaint_churn = filtered_df.groupby('complaints_last_year')['churned'].mean().reset_index()
            fig = px.line(complaint_churn, x='complaints_last_year', y='churned',
                         title="Churn Rate by Number of Complaints",
                         markers=True)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Income distribution
            fig = px.box(filtered_df, x='churned', y='annual_income',
                        title="Income Distribution by Churn Status",
                        color='churned',
                        color_discrete_sequence=['green', 'red'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Credit score analysis
            fig = px.violin(filtered_df, x='churned', y='credit_score',
                          title="Credit Score Distribution by Churn Status",
                          color='churned',
                          color_discrete_sequence=['green', 'red'])
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Top Churn Risk Indicators")
        
        # Calculate correlation with churn
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
        correlations = filtered_df[numeric_cols].corr()['churned'].sort_values(ascending=False)
        
        top_drivers = correlations[1:6]  # Exclude churned itself
        bottom_drivers = correlations[-5:]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🚨 Risk Factors (Positive Correlation)")
            for feature, corr in top_drivers.items():
                st.progress(abs(corr), text=f"{feature}: {corr:.3f}")
        
        with col2:
            st.markdown("#### 🛡️ Protective Factors (Negative Correlation)")
            for feature, corr in bottom_drivers.items():
                st.progress(abs(corr), text=f"{feature}: {corr:.3f}")

def display_strategies(df):
    """Display retention strategies"""
    
    st.markdown('<h2 class="sub-header">🎯 Customer Retention Strategies</h2>', 
                unsafe_allow_html=True)
    
    # Calculate potential savings
    total_clv_at_risk = df[df['churned'] == 1]['estimated_clv'].sum()
    retention_rate_target = st.slider("Target Retention Rate Improvement (%)", 5, 50, 20)
    
    potential_savings = total_clv_at_risk * (retention_rate_target / 100)
    
    st.markdown(f"""
    <div class="metric-card">
    <h3>💰 Business Impact</h3>
    <p>Total CLV at Risk: <strong>£{total_clv_at_risk:,.0f}</strong></p>
    <p>Target Retention Improvement: <strong>{retention_rate_target}%</strong></p>
    <p>Potential Annual Savings: <strong>£{potential_savings:,.0f}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Strategy recommendations
    st.markdown("### 📋 Segment-Specific Strategies")
    
    segments = st.multiselect(
        "Select Customer Segments to Target",
        ["High Income Professionals", "Young Families", "Retirees", 
         "Students", "Small Business Owners", "Low Engagement Customers"],
        default=["High Income Professionals", "Low Engagement Customers"]
    )
    
    strategies = {
        "High Income Professionals": [
            "Dedicated relationship manager",
            "Premium service package",
            "Exclusive investment opportunities",
            "Priority customer service line"
        ],
        "Young Families": [
            "Education savings plan",
            "Family insurance bundle",
            "Child-friendly banking features",
            "Budget planning tools"
        ],
        "Retirees": [
            "Retirement planning workshops",
            "Senior-friendly digital platform",
            "Estate planning services",
            "Higher interest savings accounts"
        ],
        "Students": [
            "Fee-free student accounts",
            "Financial literacy courses",
            "Budgeting app integration",
            "Graduate transition program"
        ],
        "Small Business Owners": [
            "Business banking specialists",
            "Cash flow management tools",
            "Networking events",
            "Quick loan approval process"
        ],
        "Low Engagement Customers": [
            "Re-engagement email campaign",
            "App usage tutorial",
            "Welcome back offer",
            "Simplified product menu"
        ]
    }
    
    for segment in segments:
        with st.expander(f"📌 {segment}"):
            st.markdown("**Recommended Actions:**")
            for strategy in strategies.get(segment, []):
                st.markdown(f"- {strategy}")
            
            # Implementation timeline
            st.markdown("**Implementation Timeline:**")
            timeline_df = pd.DataFrame({
                'Phase': ['Planning', 'Development', 'Testing', 'Launch', 'Review'],
                'Duration (weeks)': [2, 4, 2, 1, 1],
                'Responsible': ['Product Manager', 'Tech Team', 'QA Team', 'Marketing', 'Analytics']
            })
            st.dataframe(timeline_df, use_container_width=True)
    
    # ROI Calculator
    st.markdown("### 📈 ROI Calculator")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        campaign_cost = st.number_input("Campaign Cost (£)", 1000, 100000, 10000)
    with col2:
        expected_conversion = st.slider("Expected Conversion Rate (%)", 1, 50, 15)
    with col3:
        avg_retention_value = st.number_input("Avg Retention Value (£)", 100, 10000, 1000)
    
    # Calculate ROI
    customers_targeted = len(df[df['churned'] == 1])
    expected_retained = int(customers_targeted * (expected_conversion / 100))
    value_retained = expected_retained * avg_retention_value
    roi = ((value_retained - campaign_cost) / campaign_cost) * 100
    
    st.metric("Expected ROI", f"{roi:.1f}%")
    st.metric("Customers Retained", expected_retained)
    st.metric("Value Retained", f"£{value_retained:,.0f}")

if __name__ == "__main__":
    main()