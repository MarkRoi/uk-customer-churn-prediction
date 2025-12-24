# generate_synthetic_data.py
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
from sklearn.preprocessing import StandardScaler

fake = Faker('en_GB')  # UK locale

class UKCustomerDataGenerator:
    def __init__(self, n_customers=10000):
        self.n_customers = n_customers
        self.uk_regions = {
            'London': 0.25,
            'South East': 0.15,
            'North West': 0.12,
            'South West': 0.10,
            'West Midlands': 0.08,
            'East Midlands': 0.07,
            'Yorkshire': 0.07,
            'East of England': 0.06,
            'Scotland': 0.05,
            'Wales': 0.03,
            'Northern Ireland': 0.02
        }
        
        self.products = [
            'Current Account', 'Savings Account', 'Mortgage',
            'Credit Card', 'Personal Loan', 'Insurance',
            'Investment', 'Business Account'
        ]
        
        self.churn_reasons = [
            'Poor Customer Service', 'High Fees',
            'Better Offer Elsewhere', 'Moving Abroad',
            'Financial Difficulties', 'Dissatisfied with Product',
            'Technical Issues', 'No Reason Given'
        ]
    
    def generate_customers(self):
        data = []
        
        for i in range(self.n_customers):
            # Personal details
            customer_id = f'CUST{10000 + i}'
            age = np.random.randint(18, 85)
            gender = np.random.choice(['Male', 'Female'], p=[0.48, 0.52])
            
            # UK location
            region = np.random.choice(
                list(self.uk_regions.keys()),
                p=list(self.uk_regions.values())
            )
            
            # Account details
            tenure_months = np.random.randint(1, 240)  # up to 20 years
            account_age_days = np.random.randint(30, 3650)
            
            # Financial features
            credit_score = np.random.randint(300, 850)
            annual_income = np.random.normal(35000, 15000)
            annual_income = max(15000, min(annual_income, 150000))
            
            # Product holdings
            num_products = np.random.choice([1, 2, 3, 4, 5], 
                                          p=[0.1, 0.3, 0.4, 0.15, 0.05])
            products_held = random.sample(self.products, num_products)
            
            # Behavioral metrics
            avg_transaction_value = np.random.normal(150, 50)
            transaction_frequency = np.random.poisson(15)
            complaints_last_year = np.random.poisson(0.5)
            
            # Engagement metrics
            days_since_last_login = np.random.exponential(30)
            app_usage_hours = np.random.normal(5, 2)
            
            # Calculate churn probability
            churn_prob = self.calculate_churn_probability(
                age, tenure_months, credit_score, complaints_last_year,
                days_since_last_login, transaction_frequency
            )
            
            churned = 1 if random.random() < churn_prob else 0
            
            # Churn details if churned
            if churned:
                churn_reason = np.random.choice(self.churn_reasons)
                days_since_churn = np.random.randint(1, 90)
            else:
                churn_reason = None
                days_since_churn = 0
            
            # Calculate CLV
            clv = self.calculate_clv(annual_income, tenure_months, num_products)
            
            customer = {
                'customer_id': customer_id,
                'age': age,
                'gender': gender,
                'region': region,
                'tenure_months': tenure_months,
                'account_age_days': account_age_days,
                'credit_score': credit_score,
                'annual_income': annual_income,
                'num_products': num_products,
                'products_held': ','.join(products_held),
                'has_current_account': 1 if 'Current Account' in products_held else 0,
                'has_savings_account': 1 if 'Savings Account' in products_held else 0,
                'has_credit_card': 1 if 'Credit Card' in products_held else 0,
                'has_mortgage': 1 if 'Mortgage' in products_held else 0,
                'avg_transaction_value': avg_transaction_value,
                'transaction_frequency': transaction_frequency,
                'complaints_last_year': complaints_last_year,
                'days_since_last_login': days_since_last_login,
                'app_usage_hours': app_usage_hours,
                'estimated_clv': clv,
                'churned': churned,
                'churn_probability': churn_prob,
                'churn_reason': churn_reason,
                'days_since_churn': days_since_churn
            }
            
            data.append(customer)
        
        return pd.DataFrame(data)
    
    def calculate_churn_probability(self, age, tenure, credit_score, complaints, 
                                   days_since_login, trans_freq):
        """Calculate realistic churn probability"""
        base_prob = 0.15
        
        # Factors affecting churn
        tenure_factor = np.exp(-tenure/60)  # Longer tenure = lower churn
        age_factor = 1.5 if age > 65 else 1.0  # Retirees more likely to churn
        credit_factor = 1 - (credit_score/1000)  # Lower score = higher churn
        complaint_factor = 1 + (complaints * 0.3)
        engagement_factor = min(days_since_login/90, 2)  # Less engagement = higher churn
        activity_factor = 1 - (trans_freq/50)  # Less activity = higher churn
        
        prob = (base_prob * tenure_factor * age_factor * credit_factor * 
                complaint_factor * engagement_factor * activity_factor)
        
        return min(max(prob, 0.01), 0.95)
    
    def calculate_clv(self, income, tenure, num_products):
        """Calculate Customer Lifetime Value"""
        base_value = income * 0.05  # 5% of income
        tenure_bonus = tenure * 10  # £10 per month of tenure
        product_bonus = num_products * 500  # £500 per product
        
        return base_value + tenure_bonus + product_bonus

# Generate and save data
if __name__ == "__main__":
    generator = UKCustomerDataGenerator(n_customers=10000)
    df = generator.generate_customers()
    
    # Save to CSV
    df.to_csv('data/raw/uk_customers.csv', index=False)
    # df.to_csv('data/uk_customers.csv', index=False)
    
    print(f"Generated {len(df)} UK customer records")
    print(f"Churn rate: {df['churned'].mean():.2%}")
    print(f"Data saved to data/uk_customers.csv")

    