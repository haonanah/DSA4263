import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport


file_path = r"C:\Users\yusheng.leow\OneDrive - Birkenstock Group B.V. & Co. KG\Desktop\Fraudulent_online_shops_dataset.csv"
df = pd.read_csv(file_path, delimiter = ";")

print(df.info())
print(df.head())
print(df.describe())
df.isna().any()


# Ensure 'Label' is categorical for proper visualization
df["Label"] = df["Label"].astype("category")

# Plot Fraudulent vs Legitimate shop distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Label", palette="coolwarm")
plt.title("Distribution of Fraudulent vs Legitimate Online Shops")
plt.xlabel("Shop Type")
plt.ylabel("Count")
plt.show()

# Histogram of Domain Length
plt.figure(figsize=(10, 5))
sns.histplot(df, x="Domain length", bins=30, kde=True, hue="Label", palette="coolwarm", element="step")
plt.title("Domain Length Distribution (Fraudulent vs Legitimate)")
plt.xlabel("Domain Length")
plt.ylabel("Frequency")
plt.show()

# Boxplot for TrustPilot Score by Label
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="Label", y="TrustPilot score", palette="coolwarm")
plt.title("TrustPilot Score by Shop Type")
plt.xlabel("Shop Type")
plt.ylabel("TrustPilot Score")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Print report
#report = df.profile_report(title="EDA Report")
#report.to_file("EDA_Report.html")