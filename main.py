from Data import load_data
from data_cleaning import clean_data
from visualization import advanced_plots
from utils import transform_data
from kmeans import kmeans_models
from statics import calculate_mean_median
df=load_data()
print(df.head(5))

clean_data(df)
advanced_plots(df)
transform_data(df)
kmeans_models(df)
calculate_mean_median(df)