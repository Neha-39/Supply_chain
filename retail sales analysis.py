# ==========================================
# 📦 1. IMPORT LIBRARIES
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# ==========================================
# 📂 2. LOAD DATA
# ==========================================
# Replace with your dataset path
df = pd.read_csv("retail_sales.csv")

print("Initial Data:")
print(df.head())

# ==========================================
# 🧹 3. DATA CLEANING
# ==========================================

# Convert to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')

# Drop missing values
df = df.dropna()

# Remove duplicates
df = df.drop_duplicates()

# Create Revenue column
df['Revenue'] = df['Quantity'] * df['Price']

# Extract date features
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Month_Name'] = df['Order Date'].dt.strftime('%b')

print("\nCleaned Data:")
print(df.head())



# Total Sales
total_sales = df['Revenue'].sum()
print("\nTotal Revenue:", total_sales)

# Total Orders
total_orders = df['Order ID'].nunique()
print("Total Orders:", total_orders)

# Average Order Value
aov = total_sales / total_orders
print("Average Order Value:", round(aov, 2))

# ==========================================
# 📦 5. SALES BY CATEGORY
# ==========================================
category_sales = df.groupby('Category')['Revenue'].sum().sort_values(ascending=False)
print("\nSales by Category:\n", category_sales)

plt.figure(figsize=(8,5))
category_sales.plot(kind='bar', color='skyblue')
plt.title("Sales by Category")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==========================================
# 📅 6. MONTHLY SALES TREND
# ==========================================
monthly_sales = df.groupby(['Year','Month'])['Revenue'].sum().reset_index()

# Create proper date for plotting
monthly_sales['Date'] = pd.to_datetime(monthly_sales[['Year','Month']].assign(DAY=1))

plt.figure(figsize=(10,5))
plt.plot(monthly_sales['Date'], monthly_sales['Revenue'], marker='o')
plt.title("Monthly Sales Trend")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.grid()
plt.show()

# ==========================================
# 🛍 7. TOP PRODUCTS
# ==========================================
top_products = df.groupby('Product')['Revenue'].sum().nlargest(10)

plt.figure(figsize=(8,5))
top_products.plot(kind='barh', color='green')
plt.title("Top 10 Products")
plt.xlabel("Revenue")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ==========================================
# 👥 8. TOP CUSTOMERS
# ==========================================
top_customers = df.groupby('Customer ID')['Revenue'].sum().nlargest(10)
print("\nTop Customers:\n", top_customers)

# ==========================================
# 🌍 9. REGION / CITY ANALYSIS
# ==========================================
if 'City' in df.columns:
    city_sales = df.groupby('City')['Revenue'].sum().nlargest(10)

    plt.figure(figsize=(8,5))
    city_sales.plot(kind='bar', color='orange')
    plt.title("Top Cities by Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ==========================================
# 🧠 10. RFM ANALYSIS (Customer Segmentation)
# ==========================================

snapshot_date = df['Order Date'].max() + pd.Timedelta(days=1)

rfm = df.groupby('Customer ID').agg({
    'Order Date': lambda x: (snapshot_date - x.max()).days,
    'Order ID': 'nunique',
    'Revenue': 'sum'
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

print("\nRFM Table:\n", rfm.head())

# RFM Score (simple scaling)
rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4,3,2,1])
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1,2,3,4])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1,2,3,4])

rfm['RFM_Score'] = rfm[['R_Score','F_Score','M_Score']].sum(axis=1)

print("\nRFM Scores:\n", rfm.head())

# ==========================================
# 📉 11. CORRELATION HEATMAP
# ==========================================
plt.figure(figsize=(6,4))
sns.heatmap(df[['Quantity','Price','Revenue']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# ==========================================
# 💾 12. EXPORT CLEAN DATA FOR BI TOOLS
# ==========================================
df.to_csv("cleaned_retail_data.csv", index=False)
rfm.to_csv("rfm_analysis.csv")

print("\n✅ Data exported successfully!")

# ==========================================
# 🚀 END OF PROJECT
# ==========================================
