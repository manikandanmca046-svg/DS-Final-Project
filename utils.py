import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def transform_data(df):
   
    numeric_cols = df.select_dtypes(include=np.number).columns

    
    minmax_scaler = MinMaxScaler()
    df_normalized = df.copy()
    df_normalized[numeric_cols] = minmax_scaler.fit_transform(df[numeric_cols])
    print(df_normalized)
    print("\nNormalization applied to numeric columns")

    
    standard_scaler = StandardScaler()
    df_standardized = df.copy()
    df_standardized[numeric_cols] = standard_scaler.fit_transform(df[numeric_cols])
    print(df_standardized)
    print("Standardization applied to numeric columns")

    return df_normalized,df_standardized