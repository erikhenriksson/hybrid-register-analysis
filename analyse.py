import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("hybrid_discreteness_results.csv")

# Create multiple visualizations
plt.figure(figsize=(15, 5))

# 1. Histogram
plt.subplot(131)
sns.histplot(data=df, x="max_discreteness", bins=30)
plt.title("Distribution of Discreteness Scores")
plt.xlabel("Discreteness Score")
plt.ylabel("Count")

# 2. Box Plot
plt.subplot(132)
sns.boxplot(y=df["max_discreteness"])
plt.title("Box Plot of Discreteness Scores")
plt.ylabel("Discreteness Score")

# 3. Violin Plot
plt.subplot(133)
sns.violinplot(y=df["max_discreteness"])
plt.title("Violin Plot of Discreteness Scores")
plt.ylabel("Discreteness Score")

plt.tight_layout()
plt.show()

# Print summary statistics
print("\nSummary Statistics:")
print(df["max_discreteness"].describe())

# Check for potential outliers
Q1 = df["max_discreteness"].quantile(0.25)
Q3 = df["max_discreteness"].quantile(0.75)
IQR = Q3 - Q1
outliers = df[
    (df["max_discreteness"] < (Q1 - 1.5 * IQR))
    | (df["max_discreteness"] > (Q3 + 1.5 * IQR))
]
print(f"\nNumber of potential outliers: {len(outliers)}")
