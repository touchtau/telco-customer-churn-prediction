import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set a consistent seaborn style
sns.set(style='whitegrid')

def plot_tenure_distribution(df):
    """Plot distribution of customer tenure."""
    plt.figure(figsize=(8,4))
    sns.histplot(df['tenure'], bins=30, kde=True, color='steelblue')
    plt.title('Distribution of Customer Tenure')
    plt.xlabel('Tenure (months)')
    plt.ylabel('Number of Customers')
    plt.show()

def plot_senior_citizens_pie(df):
    """Pie chart: proportion of senior citizens vs non-seniors."""
    labels = ['Non-Senior Citizens', 'Senior Citizens']
    sizes = df['SeniorCitizen'].value_counts().sort_index()
    colors = ['#66b3ff','#ff9999']

    plt.figure(figsize=(6,6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('Proportion of Senior Citizens')
    plt.axis('equal')
    plt.show()

def plot_churn_by_gender(df):
    """Bar plot: churn distribution by gender."""
    plt.figure(figsize=(6,4))
    sns.countplot(x='gender', hue='Churn', data=df, palette='pastel')
    plt.title('Churn by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.show()

def plot_partners_pie(df):
    """Pie chart: customers with and without partners."""
    partner_counts = df['Partner'].value_counts()
    labels = partner_counts.index
    sizes = partner_counts.values
    colors = ['#ffcc99','#99ff99']

    plt.figure(figsize=(6,6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('Proportion of Customers With Partners')
    plt.axis('equal')
    plt.show()

def plot_contract_type_bar(df):
    """Bar plot: number of customers by contract type."""
    plt.figure(figsize=(6,4))
    sns.countplot(x='Contract', data=df, palette='Set2')
    plt.title('Number of Customers by Contract Type')
    plt.xlabel('Contract Type')
    plt.ylabel('Count')
    plt.show()

def plot_contract_type_pie(df):
    """Pie chart: distribution of contract types."""
    contract_counts = df['Contract'].value_counts()
    labels = contract_counts.index
    sizes = contract_counts.values
    colors = sns.color_palette('pastel')[0:len(labels)]

    plt.figure(figsize=(6,6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('Contract Type Distribution')
    plt.axis('equal')
    plt.show()

def plot_correlation_heatmap(df):
    """Correlation heatmap for numerical features."""
    numeric_df = df.select_dtypes(include='number')
    plt.figure(figsize=(10,8))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

def plot_churn_by_contract_type(df):
    """Bar plot: churn counts by contract type."""
    plt.figure(figsize=(6,4))
    sns.countplot(x='Contract', hue='Churn', data=df, palette='pastel')
    plt.title('Churn by Contract Type')
    plt.xlabel('Contract Type')
    plt.ylabel('Count')
    plt.show()

def plot_target_distribution(df, target_col='Churn'):
    """Bar plot: distribution of target variable (e.g., Churn)."""
    plt.figure(figsize=(4,4))
    sns.countplot(x=target_col, data=df, hue=target_col, palette='pastel', legend=False)
    #sns.countplot(x=target_col, data=df, palette='pastel')
    plt.title(f'Distribution of Target: {target_col}')
    plt.xlabel(target_col)
    plt.ylabel('Count')
    plt.show()
