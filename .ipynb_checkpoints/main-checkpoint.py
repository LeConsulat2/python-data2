import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

xls = pd.ExcelFile("University_Data.xlsx")

# Display sheet names to understand the structure
xls.sheet_names

df_raw = pd.read_excel(xls, sheet_name="Raw")
df_table = pd.read_excel(xls, sheet_name="Table")

# Convert start date into a numeric format for trend analysis
df_table["Start Date"] = pd.to_numeric(df_table["Start Date"], errors="coerce")

# Set the figure size for multiple plots
plt.figure(figsize=(15, 10))

# 1. Bar Chart - Number of programs per department
plt.subplot(2, 2, 1)
dept_counts = df_table["Department"].value_counts()
sns.barplot(x=dept_counts.index, y=dept_counts.values, palette="viridis")
plt.xticks(rotation=90)
plt.title("Number of Programs per Department")
plt.xlabel("Department")
plt.ylabel("Count")

# 2. Pie Chart - Distribution of Qualification Levels
plt.subplot(2, 2, 2)
qual_counts = df_table["Qualification Level"].value_counts()
plt.pie(
    qual_counts,
    labels=qual_counts.index,
    autopct="%1.1f%%",
    colors=sns.color_palette("pastel"),
)
plt.title("Distribution of Qualification Levels")

# 3. Line Chart - Trend of Program Sizes Over Time
plt.subplot(2, 2, 3)
df_table_sorted = df_table.sort_values(by="Start Date")
sns.lineplot(
    data=df_table_sorted, x="Start Date", y="Total Size", marker="o", color="b"
)
plt.title("Trend of Program Sizes Over Time")
plt.xlabel("Start Date")
plt.ylabel("Total Size")

# 4. Box Plot - Distribution of Total Sizes Across Qualification Levels
plt.subplot(2, 2, 4)
sns.boxplot(data=df_table, x="Qualification Level", y="Total Size", palette="coolwarm")
plt.xticks(rotation=90)
plt.title("Total Size Distribution Across Qualification Levels")
plt.xlabel("Qualification Level")
plt.ylabel("Total Size")

# Display the plots
plt.tight_layout()
plt.show()
