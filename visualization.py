import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def advanced_plots(df):
    """
    Generates all required plots for the EDA project:
    - Line Chart
    - Bar Plot
    - Distribution Plot
    - Correlation Heatmap
    - Violin Plot
    """
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    if len(numeric_cols) < 2:
        print("Not enough numeric columns to generate plots.")
        return

    # -----------------------
    # 1️⃣ LINE CHART
    # -----------------------
    plt.figure(figsize=(8, 5))
    plt.plot(df[numeric_cols[0]], df[numeric_cols[1]], marker='o')
    plt.title("Line Chart")
    plt.xlabel(numeric_cols[0])
    plt.ylabel(numeric_cols[1])
    plt.grid(True)
    plt.show()

    # -----------------------
    # 2️⃣ BAR PLOT
    # -----------------------
    plt.figure(figsize=(8, 5))
    sns.barplot(x=df[numeric_cols[0]], y=df[numeric_cols[1]])
    plt.title("Bar Plot")
    plt.xlabel(numeric_cols[0])
    plt.ylabel(numeric_cols[1])
    plt.xticks(rotation=90)
    plt.show()

    # -----------------------
    # 3️⃣ DISTRIBUTION PLOT
    # -----------------------
    plt.figure(figsize=(8, 5))
    sns.histplot(df[numeric_cols[1]], kde=True)
    plt.title("Distribution Plot")
    plt.xlabel(numeric_cols[1])
    plt.ylabel("Frequency")
    plt.show()

    # -----------------------
    # 4️⃣ CORRELATION HEATMAP
    # -----------------------
    plt.figure(figsize=(10, 6))
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Heatmap")
    plt.show()

    # -----------------------
    # 5️⃣ VIOLIN PLOT
    # -----------------------
    plt.figure(figsize=(8, 5))
    sns.violinplot(y=df[numeric_cols[1]])
    plt.title("Violin Plot")
    plt.ylabel(numeric_cols[1])
    plt.show()
