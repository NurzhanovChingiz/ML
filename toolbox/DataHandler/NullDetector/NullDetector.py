import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class NullDetector:
    """
    A class for detecting and visualizing null values in a DataFrame.
    """
    
    def __init__(self):
        pass

    def detect_nulls(self, df):
        """
        Detects and visualizes null values in a DataFrame.
        
        Args:
            df (pandas.DataFrame): The DataFrame to analyze.
        """
        # Count of all null values in dataframe columns
        null_count = df.isnull().sum()
        
        # Calculate the percentage of null values in each column
        null_percentage = (df.isnull().sum() / df.shape[0]) * 100
    
        # Create a dataframe for output
        null_df = pd.DataFrame({'Null Count': null_count, 'Null Percentage': null_percentage})
        
        print("Number and percentage of null values in df columns:")
        print(null_df)
    
        # Nullity matrix to find null value in dataframe
        plt.figure(figsize=(15, 10))
        sns.heatmap(df.isnull(), cmap="YlGnBu", cbar_kws={'label': 'Missing Data'})
        plt.title("Nullity Matrix")
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.show()

