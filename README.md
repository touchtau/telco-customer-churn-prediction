## Telco Customer Churn
Customers are the lifeblood of any business.
**Customer Churn** is the phenomenon where customers terminate their relationship with a business or organization. It is significantly related to the rate of retention and customer loyalty. The cost of retaining a current customer is considerably lower than the cost of acquiring a new customer. Therefore, the abillity to know the churners help companies and business owners to implement customer retention stratergies and take proactive measures to maintain their customer base.

The first part of this project relate to:
1. Identifying customer churn (eda and feature engineering)
2. Present a framework for churn prediction model and test the model by applying techniques to accurately detect churn rates
3. Test the accuracy of the proposed framework using appropriate performance indicators such as precision, recall, accuracy and F1 score

## Dataset
The dataset used in this project is from the IBM Sample Data Sets: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Each row represents a customer, each column contains customer’s attributes described below:
- **gender** - Whether the customer is a male or a female
- **SeniorCitizen** - Whether the customer is a senior citizen or not (1, 0)
- **Partner** - Whether the customer has a partner or not (Yes, No)
- **Dependents** - Whether the customer has dependents or not (Yes, No)
- **tenure** - Number of months the customer has stayed with the company
- **PhoneService** - Whether the customer has a phone service or not (Yes, No)
- **MultipleLines** - Whether the customer has multiple lines or not (Yes, No, No phone service)
- **InternetService** - Customer’s internet service provider (DSL, Fiber optic, No)
- **OnlineSecurity** - Whether the customer has online security or not (Yes, No, No internet service)
- **OnlineBackup** - Whether the customer has online backup or not (Yes, No, No internet service)
- **DeviceProtection** - Whether the customer has device protection or not (Yes, No, No internet service)
- **TechSupport** - Whether the customer has tech support or not (Yes, No, No internet service)
- **StreamingTV** - Whether the customer has streaming TV or not (Yes, No, No internet service) 
- **StreamingMovies** - Whether the customer has streaming movies or not (Yes, No, No internet service)
- **Contract** - The contract term of the customer (Month-to-month, One year, Two year)
- **PaperlessBilling** - Whether the customer has paperless billing or not (Yes, No)
- **PaymentMethod** - The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card 
- **MonthlyCharges** - The amount charged to the customer monthly
- **TotalCharges** - The total amount charged to the customer
- **Churn** - Whether the customer churned or not (Yes or No)

The “Churn” column is our target.
