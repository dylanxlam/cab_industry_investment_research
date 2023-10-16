import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import datetime as dt

# Load the Datasets

cab_data = pd.read_csv("Cab_Data.csv")
customer_data = pd.read_csv("Customer_ID.csv")
transaction_data = pd.read_csv("Transaction_ID.csv")
city_data = pd.read_csv("City.csv")

# Display the first few rows of each dataset to get an initial sense of the data
print("Cab Data Sample:")
print(cab_data.head())

print("\nCustomer Data Sample:")
print(customer_data.head())

print("\nTransaction Data Sample:")
print(transaction_data.head())

print("\nCity Data Sample:")
print(city_data.head())


# List field names and data types for each dataset
def list_fields_and_data_types(df):
    print("\nField Names and Data Types:")
    for column, dtype in zip(df.columns, df.dtypes):
        print(f"{column}: {dtype}")

print("\nCab Data:")
list_fields_and_data_types(cab_data)

print("\nCustomer Data:")
list_fields_and_data_types(customer_data)

print("\nTransaction Data:")
list_fields_and_data_types(transaction_data)

print("\nCity Data:")
list_fields_and_data_types(city_data)




# Display basic statistics
print("Basic Statistics of Cab Data:")
print(cab_data.describe())

# Check for missing values
print("\nMissing Values in Cab Data:")
print(cab_data.isnull().sum())


# Load City Data
city_data = pd.read_csv("City.csv")

# Display basic statistics
print("\nBasic Statistics of City Data:")
print(city_data.describe())

# Check for missing values
print("\nMissing Values in City Data:")
print(city_data.isnull().sum())

# Descriptive Statistics

# Calculate mean and median of 'Price Charged' in Cab Data
mean_price = cab_data['Price Charged'].mean()
median_price = cab_data['Price Charged'].median()

print(f"\nMean Price Charged in Cab Data: {mean_price}")
print(f"Median Price Charged in Cab Data: {median_price}")

# Calculate mean and median of 'Income (USD/Month)' in Customer Data
customer_data = pd.read_csv("Customer_ID.csv")
mean_income = customer_data['Income (USD/Month)'].mean()
median_income = customer_data['Income (USD/Month)'].median()

print(f"\nMean Income (USD/Month) in Customer Data: {mean_income}")
print(f"Median Income (USD/Month) in Customer Data: {median_income}")

# Cross-tabulation

# Load Transaction Data
transaction_data = pd.read_csv("Transaction_ID.csv")

# Create a cross-tabulation between 'Company' and 'City' in Cab Data
cross_tab_cab = pd.crosstab(cab_data['Company'], cab_data['City'])

# Create a cross-tabulation between 'Gender' and 'Age' in Customer Data
cross_tab_customer = pd.crosstab(customer_data['Gender'], customer_data['Age'])

# Display the cross-tabulation for Cab Data
print("\nCross-Tabulation between 'Company' and 'City' in Cab Data:")
print(cross_tab_cab)

# Display the cross-tabulation for Customer Data
print("\nCross-Tabulation between 'Gender' and 'Age' in Customer Data:")
print(cross_tab_customer)

# Creating 'Travel Day', 'Price per KM', and 'Profit' columns

# Define a function to convert Excel date format to datetime
def excel_to_datetime(excel_date):
    return dt.datetime(1899, 12, 30) + dt.timedelta(days=excel_date)

# Convert the 'Date of Travel' column to datetime
cab_data['Date of Travel'] = cab_data['Date of Travel'].apply(excel_to_datetime)

# Now, you can perform the desired transformations
cab_data['Travel Day'] = cab_data['Date of Travel'].dt.day_name()
cab_data['Travel Month'] = cab_data['Date of Travel'].dt.month
cab_data['Travel Year'] = cab_data['Date of Travel'].dt.year

# Calculate price per kilometer
cab_data['Price per KM'] = cab_data['Price Charged'] / cab_data['KM Travelled']

# Calculate profit (Price Charged - Cost of Trip)
cab_data['Profit'] = cab_data['Price Charged'] - cab_data['Cost of Trip']
cab_data.head()



# Binning Age into Age Groups
bins = [0, 18, 35, 60, float("inf")]
labels = ['Young', 'Young Adult', 'Middle-aged', 'Senior']
customer_data['Age Group'] = pd.cut(customer_data['Age'], bins=bins, labels=labels)

# Min-Max Scaling for Income (USD/Month)
min_income = customer_data['Income (USD/Month)'].min()
max_income = customer_data['Income (USD/Month)'].max()
customer_data['Scaled Income'] = (customer_data['Income (USD/Month)'] - min_income) / (max_income - min_income)

# Print the updated DataFrame
customer_data.head() # Print the first few rows to verify the changes


# Convert the "Payment_Mode" variable to numerical values (0 for Cash and 1 for Card)
transaction_data['Payment_Mode'] = transaction_data['Payment_Mode'].map({'Cash': 0, 'Card': 1})

# Display the updated dataset
print('Cash is 0, card is 1')
transaction_data.head()


# Clean the "Population" column by removing commas and spaces and converting to numeric
city_data['Population'] = city_data['Population'].str.replace(',', '').str.strip().astype(float)

# Clean the "Users" column by removing commas and spaces and converting to numeric
city_data['Users'] = city_data['Users'].str.replace(',', '').str.strip().astype(float)

# Calculate the ratio of cab users to the city's population
city_data['Cab_Users_Per_Capita'] = city_data['Users'] / city_data['Population']

# Display the updated dataset
print(city_data)

# Assuming you've already loaded the datasets into variables: cab_data, transaction_data, customer_data, and city_data

# Join Cab Data and Transaction Data
master_data = cab_data.merge(transaction_data, on='Transaction ID', how='inner')

# Join Customer Data
master_data = master_data.merge(customer_data, on='Customer ID', how='left')

# Optionally, join City Data (if needed)
master_data = master_data.merge(city_data, on='City', how='left')




master_data_no_duplicates = master_data.drop_duplicates()

# If you want to reset the index after removing duplicates
master_data_no_duplicates.reset_index(drop=True, inplace=True)

# Display the first few rows of the cleaned dataset
print(master_data_no_duplicates.head())

# Check the shape to see how many duplicates were removed
print("Original Dataset Shape:", master_data.shape)
print("Cleaned Dataset Shape:", master_data_no_duplicates.shape)


# Handle outliers in the entire master data for numerical columns
numerical_columns = ['KM Travelled', 'Price Charged', 'Cost of Trip', 'Age', 'Income (USD/Month)']

# Define a function to handle outliers using IQR method
def handle_outliers(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    dataframe[column] = dataframe[column].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))

# Apply outlier handling to all numerical columns
for column in numerical_columns:
    handle_outliers(master_data, column)

# Display summary statistics after handling outliers
print("Summary Statistics After Handling Outliers:")
master_data.describe()

# - Create new features from existing ones
# - Age Groups, Income Bins, Profit Margin
master_data['Age_Group'] = pd.cut(master_data['Age'], bins=[0, 18, 35, 50, 100], labels=['<18', '18-35', '36-50', '50+'])
master_data['Income_Group'] = pd.cut(master_data['Income (USD/Month)'], bins=[0, 2000, 4000, 6000, 10000], labels=['<2000', '2000-4000', '4000-6000', '6000+'])
master_data['Profit_Margin'] = master_data['Price Charged'] - master_data['Cost of Trip']



# Define a significance level
alpha = 0.05

# Helper function for printing hypothesis test results
def print_hypothesis_result(test_name, p_value, alpha):
    print(f"\n{test_name}:")
    print(f"Null Hypothesis: {null_hypothesis}")
    print(f"Alternative Hypothesis: {alternative_hypothesis}")
    print(f"P-Value: {p_value:.4f}")
    
    if p_value < alpha:
        print("Result: Reject Null Hypothesis")
    else:
        print("Result: Fail to Reject Null Hypothesis")



# Test 1: T-test to compare the average age between Pink Cab and Yellow Cab customers
null_hypothesis = "There is no significant difference in the average age between Pink Cab and Yellow Cab customers."
alternative_hypothesis = "There is a significant difference in the average age between Pink Cab and Yellow Cab customers."

pink_cab_age = master_data[master_data['Company'] == 'Pink Cab']['Age']
yellow_cab_age = master_data[master_data['Company'] == 'Yellow Cab']['Age']

t_stat_age, p_value_age = stats.ttest_ind(pink_cab_age, yellow_cab_age)
print_hypothesis_result("T-test for Age", p_value_age, alpha)

# Test 2: Chi-square test of independence between Payment Mode and Gender
null_hypothesis = "Payment mode and gender are independent."
alternative_hypothesis = "Payment mode and gender are not independent."

observed_payment_gender = pd.crosstab(master_data['Payment_Mode'], master_data['Gender'])
chi2, p_value_payment_gender, _, _ = stats.chi2_contingency(observed_payment_gender)
print_hypothesis_result("Chi-square Test for Payment Mode and Gender", p_value_payment_gender, alpha)

# Test 3: ANOVA to compare average profit among different age groups
null_hypothesis = "There is no significant difference in average profit among age groups."
alternative_hypothesis = "There is a significant difference in average profit among age groups."

bins = [18, 25, 35, 50, 65, np.inf]
labels = ['18-24', '25-34', '35-49', '50-64', '65+']
master_data['Age Group'] = pd.cut(master_data['Age'], bins=bins, labels=labels)

age_groups = master_data['Age Group'].dropna().unique()
anova_results = []

for age_group in age_groups:
    group_data = master_data[master_data['Age Group'] == age_group]['Profit']
    anova_results.append(group_data)

f_stat_age, p_value_anova_age = stats.f_oneway(*anova_results)
print_hypothesis_result("ANOVA for Average Profit Among Age Groups", p_value_anova_age, alpha)

# Test 4: T-test to compare the profitability (profit per ride) of Pink Cab and Yellow Cab
null_hypothesis = "There is no significant difference in profitability between Pink Cab and Yellow Cab."
alternative_hypothesis = "There is a significant difference in profitability between Pink Cab and Yellow Cab."

pink_cab_profit_per_ride = pink_cab_age / pink_cab_age.shape[0]
yellow_cab_profit_per_ride = yellow_cab_age / yellow_cab_age.shape[0]

t_stat_profit, p_value_profit = stats.ttest_ind(pink_cab_profit_per_ride, yellow_cab_profit_per_ride)
print_hypothesis_result("T-test for Profitability", p_value_profit, alpha)


# Segmentation
# - Segment data based on relevant criteria (e.g., Age Groups, Income Groups)
# - Analyze each segment separately
age_groups = master_data.groupby('Age_Group')
for age_group, group_data in age_groups:
    # Analyze each age group separately
    print(f"Age Group: {age_group}")
    print(group_data.describe())




# Analyze seasonality or trends over time
# Cab usage by month
monthly_usage = master_data.groupby('Travel Month')['Transaction ID'].count()

# Visualize monthly cab usage
plt.figure(figsize=(10, 6))
monthly_usage.plot(kind='line')
plt.title('Monthly Cab Usage Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Transactions')
plt.show()

# Revenue Trend Analysis
# Analyze the revenue generated by each cab company over time
monthly_revenue = master_data.groupby(['Travel Month', 'Company'])['Price Charged'].sum().unstack()

# Create a line plot for monthly revenue for each cab company
plt.figure(figsize=(10, 6))
monthly_revenue.plot(kind='line')
plt.title('Monthly Revenue by Cab Company')
plt.xlabel('Month')
plt.ylabel('Total Revenue')
plt.legend(loc='upper right')
plt.show()

# Customer Demographics Analysis

# Analyze how the average age of customers changes over time
monthly_avg_age = master_data.groupby('Travel Month')['Age'].mean()

# Create a line plot for monthly average customer age
plt.figure(figsize=(10, 6))
monthly_avg_age.plot(kind='line')
plt.title('Average Customer Age Over Time')
plt.xlabel('Month')
plt.ylabel('Average Age')
plt.show()

# City-Based Analysis

# Analyze cab usage trends in different cities over time
monthly_city_usage = master_data.groupby(['Travel Month', 'City'])['Transaction ID'].count().unstack()

# Create a line plot for monthly cab usage in different cities
plt.figure(figsize=(12, 8))
ax = monthly_city_usage.plot(kind='line')
plt.title('Monthly Cab Usage by City Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Transactions')

# Customize the legend
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title='City', ncol=2)
plt.show()


# Customer Segment Analysis

# Analyze how the usage patterns differ for various customer segments over time
monthly_age_group_usage = master_data.groupby(['Travel Month', 'Age Group'])['Transaction ID'].count().unstack()

# Create a line plot for monthly cab usage by age group
plt.figure(figsize=(12, 8))
monthly_age_group_usage.plot(kind='line')
plt.title('Monthly Cab Usage by Age Group Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Transactions')
plt.legend(loc='upper right')
plt.show()





# Analyze factors affecting profit margins
# Company-wise profit analysis
company_profit = master_data.groupby('Company')['Profit_Margin'].mean()

# Visualize profit margins by company
plt.figure(figsize=(10, 6))
company_profit.plot(kind='bar')
plt.title('Average Profit Margin by Company')
plt.xlabel('Company')
plt.ylabel('Average Profit Margin')
plt.show()

# Payment Mode Profit Analysis

# Analyze profit margins based on payment modes
payment_mode_profit = master_data.groupby('Payment_Mode')['Profit_Margin'].mean()

# Define custom labels for the x-axis
payment_mode_labels = {0: 'Card', 1: 'Cash'}

# Map the labels to the payment_mode_profit index
payment_mode_profit.index = payment_mode_profit.index.map(payment_mode_labels)

# Visualize profit margins by payment mode with custom labels
plt.figure(figsize=(10, 6))
payment_mode_profit.plot(kind='bar')
plt.title('Average Profit Margin by Payment Mode')
plt.xlabel('Payment Mode')
plt.ylabel('Average Profit Margin')
plt.show()


# Age Group Profit Analysis

# Analyze profit margins based on age groups
age_group_profit = master_data.groupby('Age Group')['Profit_Margin'].mean()

# Visualize profit margins by age group
plt.figure(figsize=(10, 6))
age_group_profit.plot(kind='bar')
plt.title('Average Profit Margin by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Average Profit Margin')
plt.show()

# City-Based Profit Analysis

# Analyze profit margins in different cities
city_profit = master_data.groupby('City')['Profit_Margin'].mean()

# Visualize profit margins by city
plt.figure(figsize=(12, 8))
city_profit.plot(kind='bar')
plt.title('Average Profit Margin by City')
plt.xlabel('City')
plt.ylabel('Average Profit Margin')
plt.xticks(rotation=90)
plt.show()

# Income Group Profit Analysis

# Analyze profit margins based on income groups
income_group_profit = master_data.groupby('Income_Group')['Profit_Margin'].mean()

# Visualize profit margins by income group
plt.figure(figsize=(10, 6))
income_group_profit.plot(kind='bar')
plt.title('Average Profit Margin by Income Group')
plt.xlabel('Income Group')
plt.ylabel('Average Profit Margin')
plt.show()
