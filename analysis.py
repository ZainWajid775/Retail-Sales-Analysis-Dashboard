"""
E-Commerce Data Mining Analysis
================================
A simple, step-by-step analysis of e-commerce transaction data

Requirements:
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Association Rule Mining
- Classification Models
- Customer Segmentation (Clustering)
- Model Evaluation
- Business Recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Create output directories
os.makedirs('outputs', exist_ok=True)
os.makedirs('outputs/visualizations', exist_ok=True)

print("="*80)
print("E-COMMERCE DATA MINING ANALYSIS")
print("="*80)

# ============================================================================
# STEP 1: DATA LOADING AND INITIAL EXPLORATION
# ============================================================================
print("\n" + "="*80)
print("STEP 1: DATA LOADING")
print("="*80)

print("\nLoading data.csv...")
df = pd.read_csv('data.csv', low_memory=False)

print(f"✓ Data loaded successfully!")
print(f"  - Total rows: {len(df):,}")
print(f"  - Total columns: {len(df.columns)}")

print("\nFirst few rows of the dataset:")
print(df.head(3))

print("\nColumn names:")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

print("\nBasic statistics:")
print(df.describe())

# ============================================================================
# STEP 2: DATA CLEANING AND PREPROCESSING
# ============================================================================
print("\n" + "="*80)
print("STEP 2: DATA CLEANING AND PREPROCESSING")
print("="*80)

print("\n[2.1] Checking for missing values...")
missing = df.isnull().sum()
print("\nMissing values per column:")
print(missing[missing > 0])

print("\n[2.2] Removing empty/irrelevant columns...")
# Drop last 5 empty columns and other irrelevant columns
columns_to_drop = [col for col in df.columns if 'Unnamed' in col or col.strip() == '']
# Also drop columns with too many missing values
for col in df.columns:
    if df[col].isnull().sum() / len(df) > 0.5:  # More than 50% missing
        if col not in columns_to_drop:
            columns_to_drop.append(col)

if columns_to_drop:
    df = df.drop(columns=columns_to_drop)
    print(f"✓ Dropped {len(columns_to_drop)} irrelevant columns")

print("\n[2.3] Standardizing column names...")
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
print("✓ Column names standardized")

print("\n[2.4] Handling missing values in key columns...")
# Keep only rows with essential information
essential_cols = ['item_id', 'created_at', 'sku', 'price', 'qty_ordered', 'increment_id']
existing_essential = [col for col in essential_cols if col in df.columns]
before_rows = len(df)
df = df.dropna(subset=existing_essential)
print(f"✓ Removed {before_rows - len(df):,} rows with missing essential data")

print("\n[2.5] Converting data types...")
# Convert date column
if 'created_at' in df.columns:
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df = df.dropna(subset=['created_at'])
    print("✓ Converted 'created_at' to datetime")

# Convert numeric columns
numeric_cols = ['price', 'qty_ordered']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=[col for col in numeric_cols if col in df.columns])
print("✓ Converted price and quantity to numeric")

print("\n[2.6] Creating derived features...")
# Extract time features
if 'created_at' in df.columns:
    df['year'] = df['created_at'].dt.year
    df['month'] = df['created_at'].dt.month
    df['day_of_week'] = df['created_at'].dt.day_name()
    df['hour'] = df['created_at'].dt.hour
    print("✓ Extracted time features (year, month, day, hour)")

# Calculate total sales
if 'price' in df.columns and 'qty_ordered' in df.columns:
    df['total_sales'] = df['price'] * df['qty_ordered']
    print("✓ Calculated total sales per transaction")

# Clean price and quantity (remove negatives/zeros)
df = df[(df['price'] > 0) & (df['qty_ordered'] > 0)]

print(f"\n✓ Data cleaning complete!")
print(f"  - Final dataset: {len(df):,} rows × {len(df.columns)} columns")
print(f"\nCleaned dataset info:")
print(df.info())

# Save cleaned data
df.to_csv('outputs/cleaned_data.csv', index=False)
print("\n✓ Cleaned data saved to 'outputs/cleaned_data.csv'")

# ============================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n" + "="*80)
print("STEP 3: EXPLORATORY DATA ANALYSIS (EDA)")
print("="*80)

print("\n[3.1] Sales Trends Over Time...")

# Monthly sales trend
df['year_month'] = df['created_at'].dt.to_period('M').astype(str)
monthly_sales = df.groupby('year_month')['total_sales'].sum().reset_index()

plt.figure(figsize=(12, 5))
plt.plot(monthly_sales['year_month'], monthly_sales['total_sales'], marker='o', linewidth=2, markersize=6)
plt.title('Monthly Sales Trend', fontsize=14, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Total Sales (PKR)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/visualizations/1_monthly_sales_trend.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Visualization saved: 1_monthly_sales_trend.png")

print("\n[3.2] Top Product Categories...")

if 'category_name_1' in df.columns:
    top_categories = df['category_name_1'].value_counts().head(10)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(top_categories)), top_categories.values, color='steelblue')
    plt.xticks(range(len(top_categories)), top_categories.index, rotation=45, ha='right')
    plt.title('Top 10 Product Categories', fontsize=14, fontweight='bold')
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Number of Orders', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('outputs/visualizations/2_top_categories.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Visualization saved: 2_top_categories.png")
    
    print(f"\nTop 5 Categories:")
    for i, (cat, count) in enumerate(top_categories.head(5).items(), 1):
        print(f"  {i}. {cat}: {count:,} orders")

print("\n[3.3] Sales by Day of Week...")

daily_sales = df.groupby('day_of_week')['total_sales'].sum()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_sales = daily_sales.reindex([day for day in day_order if day in daily_sales.index])

plt.figure(figsize=(10, 6))
colors = ['#FF6B6B' if day in ['Saturday', 'Sunday'] else '#4ECDC4' for day in daily_sales.index]
bars = plt.bar(daily_sales.index, daily_sales.values, color=colors)
plt.title('Total Sales by Day of Week', fontsize=14, fontweight='bold')
plt.xlabel('Day of Week', fontsize=12)
plt.ylabel('Total Sales (PKR)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height/1e6:.1f}M', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/visualizations/3_sales_by_day.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Visualization saved: 3_sales_by_day.png")

print("\n[3.4] Price Distribution...")

# Create meaningful price bins
max_price = df['price'].quantile(0.95)  # Focus on 95% of data
price_bins = [0, 500, 1000, 2000, 5000, 10000, 20000, max_price]
price_labels = ['0-500', '500-1K', '1K-2K', '2K-5K', '5K-10K', '10K-20K', f'20K+']

df_temp = df[df['price'] <= max_price].copy()
df_temp['price_range'] = pd.cut(df_temp['price'], bins=price_bins, labels=price_labels, include_lowest=True)
price_counts = df_temp['price_range'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
bars = plt.bar(range(len(price_counts)), price_counts.values, color='skyblue', edgecolor='black', linewidth=1.5)
plt.xticks(range(len(price_counts)), price_counts.index, rotation=45)
plt.title('Product Price Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Price Range (PKR)', fontsize=12)
plt.ylabel('Number of Products', fontsize=12)
plt.grid(axis='y', alpha=0.3)

# Add value labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/visualizations/4_price_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Visualization saved: 4_price_distribution.png")

print(f"\n  💰 Price Insights:")
print(f"     Most common price range: {price_counts.idxmax()}")
print(f"     {price_counts.max():,} products in this range")

print("\n[3.5] Best Selling Category Year-Wise...")

# Category sales by year
df['year'] = df['created_at'].dt.year
category_year = df.groupby(['year', 'category_name_1'])['total_sales'].sum().reset_index()

# Get top category per year
top_per_year = category_year.sort_values('total_sales', ascending=False).groupby('year').first().reset_index()

plt.figure(figsize=(12, 6))
colors_list = plt.cm.Set3(range(len(top_per_year)))
bars = plt.bar(top_per_year['year'].astype(str), top_per_year['total_sales'], color=colors_list, edgecolor='black', linewidth=1.5)
plt.title('Best Selling Category Year-Wise', fontsize=14, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Total Sales (PKR)', fontsize=12)
plt.grid(axis='y', alpha=0.3)

# Add category labels on bars
for i, (bar, cat) in enumerate(zip(bars, top_per_year['category_name_1'])):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{cat}\n{height/1e6:.1f}M', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/visualizations/5_category_year.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Visualization saved: 5_category_year.png")

print("\n[3.6] Payment Methods Distribution...")

# Payment method analysis
payment_counts = df['payment_method'].value_counts().head(10)

plt.figure(figsize=(12, 6))
colors_payment = plt.cm.Pastel1(range(len(payment_counts)))
bars = plt.barh(range(len(payment_counts)), payment_counts.values, color=colors_payment, edgecolor='black', linewidth=1.5)
plt.yticks(range(len(payment_counts)), payment_counts.index)
plt.title('Top 10 Payment Methods Used', fontsize=14, fontweight='bold')
plt.xlabel('Number of Orders', fontsize=12)
plt.ylabel('Payment Method', fontsize=12)
plt.grid(axis='x', alpha=0.3)

# Add value labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2.,
            f'{int(width):,}', ha='left', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/visualizations/6_payment_methods.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Visualization saved: 6_payment_methods.png")

print("\n[3.7] Cancelled/Returned Orders by Category...")

# Analyze status by category
if 'status' in df.columns:
    # Get cancelled/returned orders
    cancelled_status = df[df['status'].isin(['canceled', 'cancelled', 'returned', 'refunded', 'closed'])].copy()
    
    if len(cancelled_status) > 0:
        cancel_by_cat = cancelled_status.groupby('category_name_1').size().sort_values(ascending=False).head(10)
        
        plt.figure(figsize=(12, 6))
        colors_cancel = ['#FF6B6B' if i == 0 else '#FFA07A' for i in range(len(cancel_by_cat))]
        bars = plt.barh(range(len(cancel_by_cat)), cancel_by_cat.values, color=colors_cancel, edgecolor='black', linewidth=1.5)
        plt.yticks(range(len(cancel_by_cat)), cancel_by_cat.index)
        plt.title('Cancelled/Returned Orders by Category', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Cancelled Orders', fontsize=12)
        plt.ylabel('Category', fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{int(width):,}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('outputs/visualizations/7_cancellations_by_category.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Visualization saved: 7_cancellations_by_category.png")
        
        print("\n[3.8] Cancellations Over Time...")
        
        # Cancellations over time
        cancelled_status['month'] = cancelled_status['created_at'].dt.to_period('M').astype(str)
        cancel_monthly = cancelled_status.groupby('month').size()
        
        plt.figure(figsize=(14, 6))
        plt.plot(cancel_monthly.index, cancel_monthly.values, marker='o', linewidth=2.5, color='#E74C3C', markersize=6)
        plt.fill_between(range(len(cancel_monthly)), cancel_monthly.values, alpha=0.3, color='#E74C3C')
        plt.title('Cancelled/Returned Orders Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Number of Cancellations', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/visualizations/8_cancellations_over_time.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Visualization saved: 8_cancellations_over_time.png")
        
        print(f"\n  ⚠️ Cancellation Insights:")
        print(f"     Total cancelled/returned orders: {len(cancelled_status):,}")
        print(f"     Cancellation rate: {len(cancelled_status)/len(df)*100:.2f}%")
        print(f"     Category with most cancellations: {cancel_by_cat.index[0]}")
    else:
        print("  ℹ️ No cancelled/returned orders found in dataset")
else:
    print("  ℹ️ Status column not available for cancellation analysis")

print("\n[3.9] Additional Trends...")

# Average order value over time
monthly_aov = df.groupby('year_month')['total_sales'].mean()

plt.figure(figsize=(14, 6))
plt.plot(monthly_aov.index, monthly_aov.values, marker='o', linewidth=2.5, color='#3498DB', markersize=6)
plt.fill_between(range(len(monthly_aov)), monthly_aov.values, alpha=0.3, color='#3498DB')
plt.title('Average Order Value (AOV) Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Average Order Value (PKR)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/visualizations/9_aov_trend.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Visualization saved: 9_aov_trend.png")

# Customer acquisition trend
monthly_customers = df.groupby('year_month')['customer_id'].nunique()

plt.figure(figsize=(14, 6))
plt.plot(monthly_customers.index, monthly_customers.values, marker='s', linewidth=2.5, color='#2ECC71', markersize=6)
plt.fill_between(range(len(monthly_customers)), monthly_customers.values, alpha=0.3, color='#2ECC71')
plt.title('Unique Customers Per Month', fontsize=14, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Number of Unique Customers', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/visualizations/10_customer_trend.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Visualization saved: 10_customer_trend.png")

print("\n[3.10] Key Statistics:")
print(f"  • Average Order Value: {df['total_sales'].mean():,.2f} PKR")
print(f"  • Median Order Value: {df['total_sales'].median():,.2f} PKR")
print(f"  • Total Revenue: {df['total_sales'].sum():,.2f} PKR")
print(f"  • Total Orders: {df['increment_id'].nunique():,}")
print(f"  • Unique Customers: {df['customer_id'].nunique():,}")
print(f"  • Average Items per Order: {df['qty_ordered'].mean():.2f}")
print(f"  • Most Popular Payment: {df['payment_method'].value_counts().index[0]} ({df['payment_method'].value_counts().values[0]:,} orders)")

# ============================================================================
# STEP 4: ASSOCIATION RULE MINING
# ============================================================================
print("\n" + "="*80)
print("STEP 4: ASSOCIATION RULE MINING (Product Co-Purchase Patterns)")
print("="*80)

from mlxtend.frequent_patterns import apriori, association_rules

print("\n[4.1] Preparing transaction data...")

# Use top 50 products to keep it simple and fast
top_products = df['sku'].value_counts().head(50).index.tolist()
df_basket = df[df['sku'].isin(top_products)].copy()

print(f"✓ Analyzing top 50 products (covering {len(df_basket)/len(df)*100:.1f}% of transactions)")

# Create basket format: rows=orders, columns=products, values=0/1
basket = df_basket.groupby(['increment_id', 'sku'])['qty_ordered'].sum().unstack().fillna(0)
basket = basket.applymap(lambda x: 1 if x > 0 else 0).astype(bool)

print(f"✓ Created basket with {len(basket)} orders and {len(basket.columns)} products")

print("\n[4.2] Finding frequent itemsets...")
frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
print(f"✓ Found {len(frequent_itemsets)} frequent itemsets")

if len(frequent_itemsets) > 0:
    print("\n[4.3] Generating association rules...")
    
    # Try with lower lift threshold first
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    rules = rules.sort_values('lift', ascending=False)
    
    print(f"✓ Generated {len(rules)} total association rules")
    
    # Show breakdown by lift strength
    if len(rules) > 0:
        strong_rules = rules[rules['lift'] > 1.5]
        moderate_rules = rules[(rules['lift'] > 1.2) & (rules['lift'] <= 1.5)]
        weak_rules = rules[rules['lift'] <= 1.2]
        
        print(f"\n📊 Rules Breakdown:")
        print(f"   • Strong rules (Lift > 1.5): {len(strong_rules)}")
        print(f"   • Moderate rules (Lift 1.2-1.5): {len(moderate_rules)}")
        print(f"   • Weak rules (Lift < 1.2): {len(weak_rules)}")
        
        # Show best available rules
        display_rules = strong_rules if len(strong_rules) > 0 else (moderate_rules if len(moderate_rules) > 0 else rules)
        num_to_show = min(5, len(display_rules))
        
        print(f"\n📊 Top {num_to_show} Product Co-Purchase Patterns:")
        print("-" * 80)
        for idx, row in display_rules.head(num_to_show).iterrows():
            ant = ', '.join(list(row['antecedents']))
            con = ', '.join(list(row['consequents']))
            print(f"\n{idx+1}. When customers buy: {ant}")
            print(f"   They also buy: {con}")
            print(f"   → Support: {row['support']:.2%} | Confidence: {row['confidence']:.1%} | Lift: {row['lift']:.2f}x")
        
        # Save rules
        rules.to_csv('outputs/association_rules.csv', index=False)
        print(f"\n✓ All {len(rules)} rules saved to 'outputs/association_rules.csv'")
        
        if len(strong_rules) == 0:
            print("\n💡 Note: No strong co-purchase patterns found. This suggests:")
            print("   • Customers tend to buy single items per order")
            print("   • Products are purchased independently")
            print("   • Consider analyzing by category instead of individual SKUs")
    else:
        print("\n⚠️  No association rules generated.")
        print("💡 This means products are rarely purchased together in the same order.")
else:
    print("\n⚠️  Not enough frequent itemsets found. Try lowering min_support.")
    rules = pd.DataFrame()

# ============================================================================
# STEP 5: CLASSIFICATION MODELS - MULTIPLE SEGMENTATION APPROACHES
# ============================================================================
print("\n" + "="*80)
print("STEP 5: CLASSIFICATION MODELS (4 Segmentation Approaches)")
print("="*80)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Sample data for performance (use max 100k rows for classification)
df_sample = df.sample(n=min(100000, len(df)), random_state=42)

# Store all results
all_segmentation_results = []

# =============================================================================
# SEGMENTATION 1: HIGH-VALUE PREDICTION (ORDER VALUE)
# =============================================================================
print("\n[5.1] SEGMENTATION 1: High-Value Customer Prediction")
print("-" * 80)

# Aggregate to order level
order_features = df_sample.groupby('increment_id').agg({
    'total_sales': 'sum',
    'qty_ordered': 'sum',
    'category_name_1': 'nunique'
}).reset_index()

order_features.columns = ['order_id', 'total_spent', 'total_items', 'unique_categories']

# Define target: High-value orders (>12,000 PKR)
order_features['segment'] = (order_features['total_spent'] > 12000).astype(int)
order_features['segment_name'] = order_features['segment'].map({0: 'Low-Value', 1: 'High-Value'})

print(f"✓ Created order-level features for {len(order_features):,} orders")
print(f"  - High-value orders: {order_features['segment'].sum():,} ({order_features['segment'].mean():.1%})")

X = order_features[['total_items', 'unique_categories']]
y = order_features['segment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale for KNN
scaler1 = StandardScaler()
X_train_scaled = scaler1.fit_transform(X_train)
X_test_scaled = scaler1.transform(X_test)

models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Naive Bayes': GaussianNB(),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

for name, model in models.items():
    if name == 'K-Nearest Neighbors':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  ✓ {name}: {accuracy:.2%}")
    
    all_segmentation_results.append({
        'Segmentation': 'High-Value Prediction',
        'Model': name,
        'Accuracy': accuracy,
        'Samples': len(X),
        'Classes': 'Low-Value, High-Value'
    })

# =============================================================================
# SEGMENTATION 2: CATEGORY PREFERENCE
# =============================================================================
print("\n[5.2] SEGMENTATION 2: Category Preference Segmentation")
print("-" * 80)

# Get customer's primary category
customer_cat = df_sample.groupby(['customer_id', 'category_name_1']).agg({
    'total_sales': 'sum',
    'increment_id': 'nunique'
}).reset_index()

primary_cat = customer_cat.sort_values('total_sales', ascending=False).groupby('customer_id').first().reset_index()

# Customer features
customer_features = df_sample.groupby('customer_id').agg({
    'total_sales': 'sum',
    'increment_id': 'nunique',
    'qty_ordered': 'sum'
}).reset_index()

merged = customer_features.merge(primary_cat[['customer_id', 'category_name_1']], on='customer_id')

# Get top 5 categories for classification
top_categories = merged['category_name_1'].value_counts().head(5).index.tolist()
merged_filtered = merged[merged['category_name_1'].isin(top_categories)]

print(f"✓ Top 5 categories: {', '.join(top_categories)}")
print(f"✓ Customers analyzed: {len(merged_filtered):,}")

X = merged_filtered[['total_sales', 'increment_id', 'qty_ordered']]
y = merged_filtered['category_name_1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler2 = StandardScaler()
X_train_scaled = scaler2.fit_transform(X_train)
X_test_scaled = scaler2.transform(X_test)

for name, model in models.items():
    if name == 'K-Nearest Neighbors':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  ✓ {name}: {accuracy:.2%}")
    
    all_segmentation_results.append({
        'Segmentation': 'Category Preference',
        'Model': name,
        'Accuracy': accuracy,
        'Samples': len(X),
        'Classes': ', '.join(top_categories)
    })

# =============================================================================
# SEGMENTATION 3: PAYMENT METHOD
# =============================================================================
print("\n[5.3] SEGMENTATION 3: Payment Method Segmentation")
print("-" * 80)

df_payment = df_sample[df_sample['payment_method'].notna()].copy()

# Group payment methods into meaningful categories
def categorize_payment(method):
    if pd.isna(method):
        return None
    method_lower = str(method).lower()
    if method_lower == 'cod' or 'door' in method_lower:
        return 'COD'
    elif 'card' in method_lower or 'bankalfalah' in method_lower or 'mcb' in method_lower:
        return 'Card'
    else:
        return 'Online Payment'

df_payment['payment_category'] = df_payment['payment_method'].apply(categorize_payment)
df_payment_filtered = df_payment[df_payment['payment_category'].notna()].copy()

payment_counts = df_payment_filtered['payment_category'].value_counts()
print(f"✓ Payment categories:")
for cat, count in payment_counts.items():
    print(f"  - {cat}: {count:,} orders ({count/len(df_payment_filtered):.1%})")
print(f"✓ Orders analyzed: {len(df_payment_filtered):,}")

X = df_payment_filtered[['total_sales', 'hour', 'qty_ordered']]
y = df_payment_filtered['payment_category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler4 = StandardScaler()
X_train_scaled = scaler4.fit_transform(X_train)
X_test_scaled = scaler4.transform(X_test)

for name, model in models.items():
    if name == 'K-Nearest Neighbors':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  ✓ {name}: {accuracy:.2%}")
    
    all_segmentation_results.append({
        'Segmentation': 'Payment Method',
        'Model': name,
        'Accuracy': accuracy,
        'Samples': len(X),
        'Classes': 'COD, Online Payment, Card'
    })

# =============================================================================
# SAVE RESULTS TO CSV
# =============================================================================
results_df = pd.DataFrame(all_segmentation_results)
results_df.to_csv('outputs/classification_results.csv', index=False)
print(f"\n✓ Saved classification results to 'outputs/classification_results.csv'")

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Classification Models - All Segmentation Approaches', fontsize=16, fontweight='bold')

segmentations = ['High-Value Prediction', 'Category Preference', 'Payment Method']

for idx, seg in enumerate(segmentations):
    seg_data = results_df[results_df['Segmentation'] == seg]
    
    models_list = seg_data['Model'].tolist()
    accuracies = seg_data['Accuracy'].tolist()
    colors = ['#2ECC71' if acc == max(accuracies) else '#3498DB' for acc in accuracies]
    
    bars = axes[idx].bar(range(len(models_list)), accuracies, color=colors, edgecolor='black', linewidth=1.5)
    axes[idx].set_title(seg, fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Accuracy', fontsize=10)
    axes[idx].set_ylim(0, 1)
    axes[idx].set_xticks(range(len(models_list)))
    axes[idx].set_xticklabels([m.replace(' ', '\n') for m in models_list], fontsize=9)
    axes[idx].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[idx].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/visualizations/5_model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n✓ Visualization saved: 5_model_comparison.png")

# Get best performing model overall
best_model_row = results_df.loc[results_df['Accuracy'].idxmax()]
print(f"\n🏆 Best Model Overall: {best_model_row['Model']} ({best_model_row['Segmentation']}) with {best_model_row['Accuracy']:.2%} accuracy")

# ============================================================================
# STEP 6: CLUSTERING (Customer Segmentation)
# ============================================================================
print("\n" + "="*80)
print("STEP 6: CLUSTERING (Customer Segmentation)")
print("="*80)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("\n[6.1] Preparing RFM features...")

# Calculate RFM (Recency, Frequency, Monetary)
snapshot_date = df['created_at'].max() + pd.Timedelta(days=1)

rfm = df.groupby('increment_id').agg({
    'created_at': lambda x: (snapshot_date - x.max()).days,  # Recency
    'sku': 'count',  # Frequency (count of items)
    'total_sales': 'sum'  # Monetary
}).reset_index()

rfm.columns = ['order_id', 'recency', 'frequency', 'monetary']

print(f"✓ Prepared RFM features for {len(rfm):,} orders")

# OPTIMIZATION: Sample data if too large (faster clustering)
if len(rfm) > 50000:
    print(f"  ⚡ Sampling 50,000 orders for faster clustering (from {len(rfm):,})...")
    rfm_sample = rfm.sample(n=50000, random_state=42)
else:
    rfm_sample = rfm.copy()

# Use log transformation to handle skewness
rfm_sample['recency_log'] = np.log1p(rfm_sample['recency'])
rfm_sample['frequency_log'] = np.log1p(rfm_sample['frequency'])
rfm_sample['monetary_log'] = np.log1p(rfm_sample['monetary'])

# Scale the features
scaler_cluster = StandardScaler()
rfm_scaled = scaler_cluster.fit_transform(rfm_sample[['recency_log', 'frequency_log', 'monetary_log']])

print(f"✓ Using {len(rfm_sample):,} orders for clustering analysis")

print("\n[6.2] Finding optimal number of clusters...")
print("  (Testing K=2 to K=6 - this will take ~30 seconds)")

# Elbow method
sse = []
silhouette_scores = []
K_range = range(2, 7)  # Reduced range for faster computation

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    sse.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(rfm_scaled, kmeans.labels_))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(K_range, sse, marker='o', linewidth=2, markersize=8)
plt.title('Elbow Method', fontsize=12, fontweight='bold')
plt.xlabel('Number of Clusters (K)', fontsize=11)
plt.ylabel('SSE', fontsize=11)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, marker='o', linewidth=2, markersize=8, color='orange')
plt.title('Silhouette Score', fontsize=12, fontweight='bold')
plt.xlabel('Number of Clusters (K)', fontsize=11)
plt.ylabel('Silhouette Score', fontsize=11)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/visualizations/6_optimal_clusters.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Visualization saved: 6_optimal_clusters.png")

# Choose optimal K (highest silhouette score)
optimal_k = K_range[silhouette_scores.index(max(silhouette_scores))]
print(f"\n✓ Optimal K = {optimal_k} (Silhouette Score: {max(silhouette_scores):.3f})")

print(f"\n[6.3] Applying K-Means with K={optimal_k}...")

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
rfm_sample['cluster'] = kmeans_final.fit_predict(rfm_scaled)

# If we sampled, apply the model to the full dataset
if len(rfm_sample) < len(rfm):
    print("  Applying clusters to full dataset...")
    rfm['recency_log'] = np.log1p(rfm['recency'])
    rfm['frequency_log'] = np.log1p(rfm['frequency'])
    rfm['monetary_log'] = np.log1p(rfm['monetary'])
    rfm_full_scaled = scaler_cluster.transform(rfm[['recency_log', 'frequency_log', 'monetary_log']])
    rfm['cluster'] = kmeans_final.predict(rfm_full_scaled)
else:
    rfm = rfm_sample.copy()

print(f"✓ Clustering complete!")

# Analyze clusters
print("\n📊 Customer Segment Profile:")
print("-" * 80)
cluster_summary = rfm.groupby('cluster')[['recency', 'frequency', 'monetary']].mean().round(2)
cluster_counts = rfm['cluster'].value_counts().sort_index()

for cluster_id in range(optimal_k):
    print(f"\n🔹 Segment {cluster_id + 1} ({cluster_counts[cluster_id]:,} customers)")
    print(f"   Avg Recency: {cluster_summary.loc[cluster_id, 'recency']:.0f} days")
    print(f"   Avg Frequency: {cluster_summary.loc[cluster_id, 'frequency']:.1f} orders")
    print(f"   Avg Monetary: {cluster_summary.loc[cluster_id, 'monetary']:,.2f} PKR")

# Visualize clusters
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter = plt.scatter(rfm['frequency'], rfm['monetary'], c=rfm['cluster'], 
                     cmap='viridis', alpha=0.6, edgecolors='black', linewidth=0.5)
plt.xlabel('Frequency', fontsize=11)
plt.ylabel('Monetary Value (PKR)', fontsize=11)
plt.title('Customer Segments: Frequency vs Monetary', fontsize=12, fontweight='bold')
plt.colorbar(scatter, label='Segment')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
cluster_counts.plot(kind='bar', color='steelblue', edgecolor='black', linewidth=1.5)
plt.title('Customer Distribution by Segment', fontsize=12, fontweight='bold')
plt.xlabel('Segment', fontsize=11)
plt.ylabel('Number of Customers', fontsize=11)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/visualizations/7_customer_segments.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n✓ Visualization saved: 7_customer_segments.png")

# ============================================================================
# STEP 7: MODEL EVALUATION SUMMARY
# ============================================================================
print("\n" + "="*80)
print("STEP 7: EVALUATION SUMMARY")
print("="*80)

print("\n📈 PERFORMANCE METRICS:")
print("-" * 80)

print("\n[Classification Models - All Segmentations]")
for seg in results_df['Segmentation'].unique():
    print(f"\n  {seg}:")
    seg_data = results_df[results_df['Segmentation'] == seg]
    for _, row in seg_data.iterrows():
        best_marker = "⭐" if row['Accuracy'] == seg_data['Accuracy'].max() else "  "
        print(f"    {best_marker} {row['Model']}: {row['Accuracy']:.2%}")

print("\n[Clustering]")
print(f"  • Optimal Clusters: {optimal_k}")
print(f"  • Silhouette Score: {max(silhouette_scores):.3f}")
print(f"  • Largest Segment: {cluster_counts.max():,} customers")
print(f"  • Smallest Segment: {cluster_counts.min():,} customers")

if len(rules) > 0:
    print("\n[Association Rules]")
    print(f"  • Total Rules: {len(rules)}")
    print(f"  • Avg Confidence: {rules['confidence'].mean():.1%}")
    print(f"  • Avg Lift: {rules['lift'].mean():.2f}x")
    print(f"  • Strong Rules (Lift > 2): {len(rules[rules['lift'] > 2])}")

# ============================================================================
# STEP 8: BUSINESS RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("STEP 8: BUSINESS RECOMMENDATIONS")
print("="*80)

print("\n💡 KEY INSIGHTS & ACTIONABLE RECOMMENDATIONS:")
print("-" * 80)

print("\n[1] PRODUCT STRATEGY")
if len(rules) > 0 and len(rules) >= 3:
    print("  📦 Product Bundling Opportunities:")
    for idx, row in rules.head(3).iterrows():
        ant = ', '.join(list(row['antecedents']))
        con = ', '.join(list(row['consequents']))
        print(f"     • Bundle '{ant}' with '{con}' (Lift: {row['lift']:.2f}x)")
    print("\n  ✅ ACTION: Create combo deals and cross-sell recommendations")
else:
    print("  📦 Focus on category-level bundling based on top categories")

if 'category_name_1' in df.columns:
    print(f"\n  🎯 Top Performing Category: {df['category_name_1'].value_counts().index[0]}")
    print("  ✅ ACTION: Expand inventory and marketing for this category")

print("\n[2] CUSTOMER TARGETING")
print(f"  🎯 Best Overall Model: {best_model_row['Model']} on {best_model_row['Segmentation']} ({best_model_row['Accuracy']:.1%} accuracy)")
print("  ✅ ACTION: Use models to identify and target customers based on different segmentation approaches")
print("  ✅ ACTION: High-Value: Offer loyalty rewards to predicted high-value customers")
print("  ✅ ACTION: Category Preference: Send personalized category-specific promotions")
print("  ✅ ACTION: Payment Method: Optimize checkout experience based on predicted payment preference")

print("\n[3] CUSTOMER SEGMENTATION STRATEGIES")
for cluster_id in range(optimal_k):
    recency = cluster_summary.loc[cluster_id, 'recency']
    monetary = cluster_summary.loc[cluster_id, 'monetary']
    
    print(f"\n  📊 Segment {cluster_id + 1} ({cluster_counts[cluster_id]:,} customers):")
    
    if recency < rfm['recency'].median() and monetary > rfm['monetary'].median():
        print("     Type: CHAMPIONS (Recent & High-Value)")
        print("     ✅ Reward with VIP perks and exclusive early access")
    elif recency > rfm['recency'].median() and monetary > rfm['monetary'].median():
        print("     Type: AT RISK (Inactive but Valuable)")
        print("     ✅ Win-back campaign with personalized discounts")
    elif monetary < rfm['monetary'].median():
        print("     Type: POTENTIAL (Low Spenders)")
        print("     ✅ Nurture with targeted offers to increase spending")
    else:
        print("     Type: REGULAR CUSTOMERS")
        print("     ✅ Maintain engagement with regular promotions")

print("\n[4] OPERATIONAL INSIGHTS")
best_day = daily_sales.idxmax()
print(f"  📅 Peak Sales Day: {best_day}")
print("  ✅ ACTION: Schedule major promotions and ensure stock availability on this day")

print(f"\n  💵 Average Order Value: {df['total_sales'].mean():,.0f} PKR")
print("  ✅ ACTION: Implement strategies to increase AOV (free shipping threshold, upsells)")

print("\n[5] KEY PERFORMANCE INDICATORS TO TRACK")
print("  📊 Monitor these metrics regularly:")
print("     • Customer Lifetime Value by segment")
print("     • Conversion rate of high-value predictions")
print("     • Cross-sell success rate from association rules")
print("     • Segment migration (e.g., Potential → Champions)")
print("     • Churn rate (customers with high recency)")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE! ✅")
print("="*80)

print("\n📁 OUTPUT FILES GENERATED:")
print("  1. outputs/cleaned_data.csv - Preprocessed dataset")
if len(rules) > 0:
    print("  2. outputs/association_rules.csv - Product co-purchase rules")
print("\n📊 VISUALIZATIONS CREATED:")
print("  EDA Visualizations:")
print("    1. 1_monthly_sales_trend.png - Sales over time")
print("    2. 2_top_categories.png - Popular product categories")
print("    3. 3_sales_by_day.png - Weekly sales pattern")
print("    4. 4_price_distribution.png - Product pricing")
print("    5. 5_category_year.png - Best selling category year-wise")
print("    6. 6_payment_methods.png - Payment methods distribution")
print("    7. 7_cancellations_by_category.png - Cancelled orders by category")
print("    8. 8_cancellations_over_time.png - Cancellation trends")
print("    9. 9_aov_trend.png - Average order value over time")
print("    10. 10_customer_trend.png - Customer acquisition trend")
print("  Model Visualizations:")
print("    11. 5_model_comparison.png - Classification results (3 segmentations)")
print("    12. 6_optimal_clusters.png - Clustering analysis")
print("    13. 7_customer_segments.png - Customer segmentation")

print("\n🎯 NEXT STEPS:")
print("  1. Review all visualizations in 'outputs/visualizations/' folder")
print("  2. Validate business recommendations with stakeholders")
print("  3. Implement A/B tests for proposed strategies")
print("  4. Set up automated reporting for KPIs")
print("  5. Retrain models monthly with new data")

print("\n" + "="*80)
print("Thank you for using this analysis tool!")
print("="*80)
