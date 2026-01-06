import pandas as pd
import numpy as np
import random

class UKCustomerDataGenerator:
    def __init__(self, n_customers=5000, target_churn_rate=0.25):
        self.n_customers = n_customers
        self.target_churn_rate = target_churn_rate
        np.random.seed(42)
        random.seed(42)
        
        self.uk_regions = ['London', 'South_East', 'North_West', 'South_West', 
                          'West_Midlands', 'East_Midlands', 'Yorkshire', 
                          'East_of_England', 'Scotland', 'Wales']
        
    def generate_customers(self):
        """Generate data with STRONG, CLEAR churn signals"""
        print("🎯 GENERATING DATA WITH STRONG CHURN SIGNALS")
        print("=" * 50)
        
        data = []
        
        for i in range(self.n_customers):
            # 1. CREATE CLEAR CHURN SIGNALS FIRST
            
            # Signal 1: Engagement (VERY STRONG)
            # High days_since_login = HIGH churn probability
            if random.random() < 0.3:  # 30% disengaged (will likely churn)
                days_since_login = np.random.exponential(40) + 30  # 30-90 days
                engagement_churn_prob = 0.8  # 80% chance to churn
            else:  # 70% engaged (will likely stay)
                days_since_login = np.random.exponential(5)  # 0-15 days
                engagement_churn_prob = 0.1  # 10% chance to churn
            
            # Signal 2: Credit Score (STRONG)
            # Low credit_score = HIGH churn probability
            if random.random() < 0.25:  # 25% poor credit (will likely churn)
                credit_score = np.random.normal(500, 50)  # 400-600
                credit_churn_prob = 0.7  # 70% chance to churn
            else:  # 75% good credit (will likely stay)
                credit_score = np.random.normal(700, 50)  # 600-800
                credit_churn_prob = 0.15  # 15% chance to churn
            
            # Signal 3: Complaints (MEDIUM)
            # High complaints = MEDIUM churn probability
            if random.random() < 0.15:  # 15% complainers
                complaints = np.random.poisson(2) + 1  # 1-5 complaints
                complaint_churn_prob = 0.6  # 60% chance to churn
            else:  # 85% no complaints
                complaints = np.random.poisson(0.1)  # 0-1 complaints
                complaint_churn_prob = 0.1  # 10% chance to churn
            
            # Signal 4: Transaction Frequency (MEDIUM)
            # Low transactions = MEDIUM churn probability
            if random.random() < 0.2:  # 20% low activity
                transaction_freq = np.random.poisson(5)  # 0-10 transactions
                activity_churn_prob = 0.65  # 65% chance to churn
            else:  # 80% normal activity
                transaction_freq = np.random.poisson(20)  # 10-30 transactions
                activity_churn_prob = 0.15  # 15% chance to churn
            
            # 2. COMBINE SIGNALS FOR FINAL CHURN DECISION
            # Weighted average of probabilities
            total_prob = (
                engagement_churn_prob * 0.4 +  # 40% weight (strongest)
                credit_churn_prob * 0.3 +      # 30% weight (strong)
                complaint_churn_prob * 0.2 +   # 20% weight (medium)
                activity_churn_prob * 0.1      # 10% weight (medium)
            )
            
            # Add small random noise
            total_prob += np.random.normal(0, 0.05)
            total_prob = max(0.05, min(total_prob, 0.95))
            
            # Decide churn
            churned = 1 if random.random() < total_prob else 0
            
            # 3. GENERATE OTHER FEATURES
            customer = {
                'customer_id': f'CUST{10000 + i}',
                'age': np.random.randint(18, 75),
                'gender': random.choice(['Male', 'Female']),
                'region': random.choice(self.uk_regions),
                'tenure_months': np.random.randint(1, 120),
                'credit_score': max(300, min(850, int(credit_score))),
                'annual_income': max(15000, min(150000, np.random.lognormal(10.5, 0.4))),
                'days_since_last_login': min(int(days_since_login), 365),
                'complaints_last_year': int(complaints),
                'transaction_frequency': int(transaction_freq),
                'num_products': np.random.randint(1, 6),
                'has_current_account': 1,
                'has_savings_account': random.choices([0, 1], weights=[0.3, 0.7])[0],
                'has_credit_card': random.choices([0, 1], weights=[0.4, 0.6])[0],
                'has_mortgage': 1 if random.random() < 0.3 else 0,
                'avg_transaction_value': np.random.gamma(shape=2, scale=75),
                'app_usage_hours': np.random.gamma(shape=2, scale=2.5),
                'churned': churned
            }
            
            data.append(customer)
        
        df = pd.DataFrame(data)
        
        # 4. ENSURE TARGET CHURN RATE
        actual_churn = df['churned'].sum()
        target_churn = int(self.n_customers * self.target_churn_rate)
        
        if actual_churn < target_churn:
            # Add more churned customers (pick those with highest risk signals)
            non_churned = df[df['churned'] == 0].copy()
            non_churned['risk_score'] = (
                (non_churned['days_since_last_login'] > 30) * 3 +
                (non_churned['credit_score'] < 600) * 2 +
                (non_churned['complaints_last_year'] > 0) * 2 +
                (non_churned['transaction_frequency'] < 10) * 1
            )
            to_convert = non_churned.nlargest(target_churn - actual_churn, 'risk_score')
            df.loc[to_convert.index, 'churned'] = 1
        
        elif actual_churn > target_churn:
            # Remove some churned customers (pick those with lowest risk signals)
            churned = df[df['churned'] == 1].copy()
            churned['risk_score'] = (
                (churned['days_since_last_login'] < 7) * -3 +
                (churned['credit_score'] > 700) * -2 +
                (churned['complaints_last_year'] == 0) * -2 +
                (churned['transaction_frequency'] > 20) * -1
            )
            to_convert = churned.nsmallest(actual_churn - target_churn, 'risk_score')
            df.loc[to_convert.index, 'churned'] = 0
        
        # 5. PRINT DIAGNOSTICS
        print(f"✅ Generated {len(df)} customer records")
        print(f"🎯 Target churn rate: {self.target_churn_rate:.1%}")
        print(f"📊 Actual churn rate: {df['churned'].mean():.2%}")
        print(f"📈 Churned: {df['churned'].sum():,}, Not churned: {len(df) - df['churned'].sum():,}")
        
        # Show STRONG correlations
        print("\n🔍 FEATURE CORRELATIONS WITH CHURN (Should be > 0.2):")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        strong_signals = []
        for col in numeric_cols:
            if col not in ['churned', 'customer_id']:
                corr = df[col].corr(df['churned'])
                if abs(corr) > 0.1:
                    symbol = '✅' if abs(corr) > 0.2 else '⚠️'
                    print(f"  {col:25s}: {corr:+.3f} {symbol}")
                    if abs(corr) > 0.2:
                        strong_signals.append(col)
        
        if len(strong_signals) >= 3:
            print(f"\n🎉 EXCELLENT! Found {len(strong_signals)} strong signals")
        else:
            print(f"\n⚠️  WARNING: Only {len(strong_signals)} strong signals found")
        
        return df

# Generate and save data
if __name__ == "__main__":
    print("=" * 60)
    print("UK CUSTOMER CHURN DATA GENERATOR - STRONG SIGNAL VERSION")
    print("=" * 60)
    
    generator = UKCustomerDataGenerator(n_customers=5000, target_churn_rate=0.25)
    df = generator.generate_customers()
    
    # Save to CSV
    df.to_csv('data/uk_customers.csv', index=False)
    
    print(f"\n💾 Data saved to: data/uk_customers.csv")
    
    # Quick test with simple model
    print("\n🧪 QUICK MODEL TEST:")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    
    # Simple preprocessing
    test_df = df.drop(['customer_id'], axis=1)
    test_df['gender'] = test_df['gender'].map({'Male': 0, 'Female': 1})
    region_dummies = pd.get_dummies(test_df['region'], prefix='region')
    test_df = pd.concat([test_df, region_dummies], axis=1)
    test_df = test_df.drop('region', axis=1)
    
    X = test_df.drop('churned', axis=1)
    y = test_df['churned']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"  Simple Random Forest AUC: {auc:.4f}")
    print(f"  Expected: 0.85+ (good), 0.75+ (ok), <0.70 (problem)")
    
    if auc > 0.85:
        print("  🎉 PERFECT! Data has excellent signal for ML models.")
    elif auc > 0.75:
        print("  👍 GOOD! Data has reasonable signal.")
    else:
        print("  ⚠️  WEAK! Data needs stronger signals.")