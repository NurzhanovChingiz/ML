import numpy as np

def optimize_data_types(df):
    optimized_df = df.copy()

    for column in optimized_df.columns:
        current_dtype = optimized_df[column].dtype

        if current_dtype == object:
            # Convert object columns to categorical if unique values are less than 50% of total values
            unique_count = len(optimized_df[column].unique())
            total_count = len(optimized_df[column])
            if unique_count / total_count < 0.5:
                optimized_df[column] = optimized_df[column].astype("category")
        elif current_dtype == int:
            # Downcast int columns to smaller types if possible
            optimized_df[column] = pd.to_numeric(optimized_df[column], downcast="integer")
        elif current_dtype == float:
            # Downcast float columns to smaller types if possible
            optimized_df[column] = pd.to_numeric(optimized_df[column], downcast="float")
        elif current_dtype == bool:
            # Convert bool columns to integers
            optimized_df[column] = optimized_df[column].astype(int)
        elif "datetime" in str(current_dtype):
            # Convert datetime columns to datetime type
            optimized_df[column] = pd.to_datetime(optimized_df[column])

    return optimized_df
