# adjust_churn_rate.py - RUN THIS ONCE
import pandas as pd
import numpy as np
import random

print("🔄 ADJUSTING CHURN RATE FROM 44% TO 20%...")
print("=" * 50)

# Load your current data
df = pd.read_csv('data/uk_customers.csv')

print(f"Current churn rate: {df['churned'].mean():.2%}")
print(f"Churned customers: {df['churned'].sum():,}")
print(f"Total customers: {len(df):,}")

# Calculate how many need to change
current_churned = df['churned'].sum()
target_churned = int(len(df) * 0.20)  # 20% target
to_unchurn = current_churned - target_churned

print(f"\nTarget: {target_churned:,} churned customers (20%)")
print(f"Need to convert {to_unchurn:,} churned to not-churned")

if to_unchurn > 0:
    # Find churned customers with lowest risk factors (convert these back)
    churned_customers = df[df['churned'] == 1].copy()
    
    # Calculate risk score (lower = less risky, should stay churned)
    churned_customers['risk_score'] = (
        (churned_customers['days_since_last_login'] < 7).astype(int) * -2 +  # Low risk
        (churned_customers['credit_score'] > 700).astype(int) * -1.5 +       # Low risk
        (churned_customers['complaints_last_year'] == 0).astype(int) * -1.5 + # Low risk
        (churned_customers['transaction_frequency'] > 20).astype(int) * -1 +  # Low risk
        (churned_customers['tenure_months'] > 24).astype(int) * -0.5         # Low risk
    )
    
    # Select lowest risk churned customers to convert to not-churned
    to_convert = churned_customers.nsmallest(to_unchurn, 'risk_score')
    
    # Convert them back to not-churned
    df.loc[to_convert.index, 'churned'] = 0
    df.loc[to_convert.index, 'churn_reason'] = 'N/A'
    
    print(f"✅ Converted {len(to_convert):,} low-risk churned customers to not-churned")
else:
    # If we need more churned (unlikely)
    not_churned = df[df['churned'] == 0].copy()
    needed = abs(to_unchurn)
    
    not_churned['risk_score'] = (
        (not_churned['days_since_last_login'] > 30).astype(int) * 2 +
        (not_churned['credit_score'] < 600).astype(int) * 1.5 +
        (not_churned['complaints_last_year'] > 0).astype(int) * 1.5 +
        (not_churned['transaction_frequency'] < 10).astype(int) * 1
    )
    
    to_convert = not_churned.nlargest(needed, 'risk_score')
    df.loc[to_convert.index, 'churned'] = 1
    df.loc[to_convert.index, 'churn_reason'] = to_convert['churn_reason'].apply(
        lambda x: random.choice([
            'Poor Customer Service', 'High Fees', 'Better Offer Elsewhere',
            'Financial Difficulties', 'Dissatisfied with Product'
        ]) if x == 'N/A' else x
    )
    
    print(f"✅ Converted {len(to_convert):,} high-risk not-churned customers to churned")

# Save the adjusted data
df.to_csv('data/uk_customers_adjusted.csv', index=False)
df.to_csv('data/uk_customers.csv', index=False)  # Overwrite original

print(f"\n📊 FINAL RESULTS:")
print(f"Total customers: {len(df):,}")
print(f"Churned customers: {df['churned'].sum():,}")
print(f"Churn rate: {df['churned'].mean():.2%}")

# Show correlations
print(f"\n🔍 Feature correlations with churn:")
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col not in ['churned', 'customer_id', 'risk_score']:
        corr = df[col].corr(df['churned'])
        if abs(corr) > 0.05:
            symbol = '✅' if abs(corr) > 0.1 else '⚠️'
            print(f"  {col:25s}: {corr:+.3f} {symbol}")

print(f"\n💾 Adjusted data saved to: data/uk_customers.csv")