#!/usr/bin/env python
# coding: utf-8

# # Mobile Money Transactions Fraud Analysis
# 
# This notebook analyzes a dataset of mobile money transactions to identify patterns and occurrences of fraudulent activities. The dataset contains five types of transactions: CASH-IN, CASH-OUT, DEBIT, PAYMENT, and TRANSFER. The aim is to essentially analyze the distribution of these transactions, identify fraud-prone types, and evaluate the effectiveness of the fraud detection system.
# 

# In[127]:


# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[128]:


# Loading the dataset from a CSV file
df = pd.read_csv("C:\\Users\\Kevin\\Downloads\\Chase\\transactions.csv")


# In[129]:


# Displaying the first few rows of the dataset to understand its structure
print(df.head())


# ### Data Overview
# The dataset contains 200,000 rows and 11 columns. Here is a quick overview of the key columns:
# - `step`: Time step unit of the transaction (1 hour).
# - `type`: The type of transaction, which can be CASH-IN, CASH-OUT, DEBIT, PAYMENT, or TRANSFER.
# - `amount`: The amount of money transacted.
# - `nameOrig`: The ID of the customer initiating the transaction.
# - `oldbalanceOrg`: Initial balance of the customer before the transaction.
# - `newbalanceOrig`: New balance of the customer after the transaction.
# - `nameDest`: The ID of the customer receiving the transaction.
# - `oldbalanceDest`: Initial balance of the receiver before the transaction.
# - `newbalanceDest`: New balance of the receiver after the transaction.
# - `isFraud`: 1 if the transaction is fraudulent, 0 otherwise.
# - `isFlaggedFraud`: 1 if the transaction is flagged as fraudulent by the system, 0 otherwise.

# In[130]:


# Checking the shape of the dataset to understand the number of rows and columns
print(df.shape)


# In[131]:


# Generating descriptive statistics of the dataset to get an overview of data distribution
print(df.describe())


# ### Dataset Statistics
# The dataset has 200,000 transactions and 11 columns. Descriptive statistics provide insights into the data distribution. For example:
# - The average transaction amount is around 179,861.90.
# - The maximum transaction amount is 9.2 million, indicating a wide range of transaction values.
# - The majority of transactions have zero balances for `oldbalanceDest` and `newbalanceDest`, suggesting that many transactions may not involve actual account holders.

# In[132]:


transaction_counts = df['type'].value_counts()
print(transaction_counts)


# ### Fraudulent Transactions Analysis
# We filter the dataset to identify all transactions that were marked as fraud (`isFraud = 1`). This helps in understanding which transaction types are more susceptible to fraud.

# In[133]:


# Filtering the dataset to show only fraudulent transactions
fraud_transactions = df[df['isFraud'] == 1]
print(fraud_transactions)


# ### Transaction Type Distribution
# The following plot shows the distribution of different transaction types in the dataset. This helps in understanding which types are most common.

# In[134]:


# Visualizing the count of each transaction type
sns.countplot(x='type', data=df)
plt.title('Distribution of Transaction Types')
plt.show()


# ### Insights
# The plot shows that `PAYMENT` transactions are the most frequent, followed by `CASH_OUT` and `CASH_IN`. `DEBIT` transactions are relatively rare, which might indicate that they are not commonly used in this system.

# ### Transaction Types Differentiated by Fraud Status
# 
# We will visualize the number of transactions for each type, differentiated by whether they are fraudulent (`isFraud = 1`) or not (`isFraud = 0`). This count plot provides a clear view of which transaction types are more prone to fraud. If some transaction types do not have fraudulent transactions, we will annotate them to reflect this.

# In[135]:


import pandas as pd

# Count the number of transactions per type and fraud status
fraud_counts = df.groupby(['type', 'isFraud']).size().reset_index(name='count')

# Print the fraud counts for transparency
print(fraud_counts)

# Create the count plot with custom palette
sns.countplot(x='type', hue='isFraud', data=df, palette='Blues')

# Annotate the bars with their respective counts
for p in plt.gca().patches:
    plt.annotate(f'{int(p.get_height())}', (p.get_x() + 0.15, p.get_height() + 10))

# Display the plot
plt.title('Transaction Types Differentiated by Fraud Status')
plt.show()


# ### Insights
# 
# - The plot shows that most fraudulent transactions are concentrated in specific types like `TRANSFER` and `CASH_OUT`.
# - Some transaction types, such as `CASH-IN` and `PAYMENT`, have no fraudulent transactions in this dataset, as indicated by annotations on the plot.
# - This visualization helps us focus on the most critical transaction types for fraud detection.

# ### Correlation Analysis
# 
# To better understand the relationships between the key numeric features in the dataset, we calculate the correlation matrix for a subset of features. By focusing on the most relevant features such as transaction amounts, balances, and fraud occurrence, we aim to highlight the important correlations while keeping the visualization clear and interpretable.

# In[136]:


# Selecting a subset of numeric columns relevant to fraud detection for a clearer heatmap
relevant_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud']

# Calculating the correlation matrix of the selected features
correlation_subset = df[relevant_features].corr()

# Plotting the correlation heatmap with rotated axis labels for better readability
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_subset, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title('Correlation Heatmap of Relevant Features')
plt.xticks(rotation=45)  # Rotating x-axis labels
plt.yticks(rotation=0)   # Keeping y-axis labels horizontal
plt.show()


# ### Insights from the Correlation Heatmap
# 
# - `oldbalanceDest` and `newbalanceDest` have a high positive correlation (0.95), as expected, since they represent the balance before and after transactions for the destination account.
# - `oldbalanceOrg` and `newbalanceOrig` also have a perfect positive correlation (1.0), which is logical for similar reasons.
# - `amount` shows a moderate correlation with `newbalanceDest` (0.34), suggesting that transaction amounts significantly impact the destination balance.
# - `isFraud` has very low correlation with other features, reinforcing that detecting fraud may require more sophisticated techniques beyond simple linear relationships.

# ### Fraud Rate by Transaction Type
# We calculate the fraud rate for each type of transaction to identify which types have a higher likelihood of fraud. The plot below shows the fraud rates.

# In[137]:


# Calculating the fraud rate for each transaction type
fraud_rate_by_type = df.groupby('type')['isFraud'].mean()

# Plotting the fraud rate by transaction type to visualize which types have the highest rates of fraud
fraud_rate_by_type.plot(kind='bar', title='Fraud Rate by Transaction Type')
plt.show()


# ### Insights
# `TRANSFER` and `CASH_OUT` transactions have the highest fraud rates. This suggests that these transaction types are more frequently targeted by fraudulent activities.

# ### Flagged vs Actual Fraud
# The heatmap below compares transactions that were flagged as fraud by the system against the ones that were actually fraudulent. This helps assess the effectiveness of the fraud detection system.

# In[138]:


# Creating a heatmap to compare flagged fraud vs actual fraud
flagged_vs_actual = pd.crosstab(df['isFraud'], df['isFlaggedFraud'])
sns.heatmap(flagged_vs_actual, annot=True, cmap='Blues', fmt='d')
plt.title('Flagged Fraud vs Actual Fraud')
plt.show()


# ### Conclusion
# The current fraud detection system flags very few transactions as fraudulent, and there is a noticeable discrepancy between flagged fraud and actual fraud. This suggests room for improvement in the detection algorithm. 
# - `TRANSFER` and `CASH_OUT` transactions are the most prone to fraud in this dataset.
# - The fraud detection system currently has a low rate of flagging fraud, indicating it may need improvements.
# - Future work could involve creating a machine learning model to better detect fraudulent transactions and reduce false negatives.
