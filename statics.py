import pandas as pd
import numpy as np

def calculate_mean_median(df):

    numeric_cols = df.select_dtypes(include=np.number).columns
    
    mean_values = df[numeric_cols].mean()
    median_values = df[numeric_cols].median()
    
    stats_df = pd.DataFrame({
        'Mean': mean_values,
        'Median': median_values
    })
    print(stats_df)
    return stats_df
