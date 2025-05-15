# iris_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Set Seaborn style
sns.set(style="whitegrid")

# Task 1: Load and Explore the Dataset
try:
    # Load Iris dataset
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].apply(lambda x: iris.target_names[x])
    print("✅ Dataset loaded successfully.\n")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

# Display first few rows
print("📌 First 5 rows of the dataset:")
print(df.head())

# Check structure and missing values
print("\n📋 Dataset Info:")
print(df.info())

print("\n🧼 Missing Values:")
print(df.isnull().sum())

# Clean data (if needed)
df.dropna(inplace=True)

# Task 2: Basic Data Analysis
print("\n📊 Descriptive Statistics:")
print(df.describe())

print("\n📈 Mean values grouped by species:")
grouped = df.groupby('species').mean()
print(grouped)

# Task 3: Data Visualization

# Add a fake time index column to simulate time-series data
df['index'] = df.index

# Line chart: Sepal length over time
plt.figure(figsize=(10, 5))
sns.lineplot(x='index', y='sepal length (cm)', data=df)
plt.title('Trend of Sepal Length Over Time')
plt.xlabel('Index (Simulated Time)')
plt.ylabel('Sepal Length (cm)')
plt.tight_layout()
plt.show()

# Bar chart: Average petal length per species
plt.figure(figsize=(8, 5))
sns.barplot(x='species', y='petal length (cm)', data=df, ci=None)
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.tight_layout()
plt.show()

# Histogram: Sepal width distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['sepal width (cm)'], kde=True, bins=20, color='skyblue')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Scatter plot: Sepal length vs Petal length
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title('Sepal Length vs Petal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.tight_layout()
plt.show()
