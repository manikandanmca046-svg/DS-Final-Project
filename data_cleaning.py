import pandas as pd
import numpy as np

def clean_data(df):
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Iterate over numeric columns
    for col in df.select_dtypes(include=np.number):
        # Fill missing values with median
        df[col].fillna(df[col].median(), inplace=True)
        
        # Calculate IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Remove outliers
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    
    return df
