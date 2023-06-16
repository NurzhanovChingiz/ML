import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from collections import Counter

class OutliersDetector:
    """Class to detect and handle outliers in a dataframe."""
    
    def __init__(self, df, method="iqr"):
        """
        Initialize OutliersDetector with dataframe and method.
        
        :param df: input DataFrame
        :param method: method for outlier detection; "iqr" for Interquartile Range, "sd" for Standard Deviation, "zscore" for Z-Score
        """
        self.df = df
        self.method = method
        # Select numerical columns in the dataframe
        self.numerical_features = df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns

    def detect_outliers_iqr(self):
        """Detect outliers using the Interquartile Range (IQR) method."""
        outlier_indices = []
        for c in self.numerical_features:
            # Calculate Q1, Q3, and IQR for each numerical column
            Q1 = np.percentile(self.df[c], 25)
            Q3 = np.percentile(self.df[c], 75)
            IQR = Q3 - Q1
            # Identify outlier step
            outlier_step = IQR * 1.5
            # Get indices of outliers
            outlier_list_col = self.df[(self.df[c] < Q1 - outlier_step) | (self.df[c] > Q3 + outlier_step)].index
            outlier_indices.extend(outlier_list_col)
        # Count indices appearing more than twice
        outlier_indices = Counter(outlier_indices)
        self.multiple_outliers = [i for i, v in outlier_indices.items() if v > 2]
        return self.multiple_outliers

    def detect_outliers_sd(self):
        """Detect outliers using the Standard Deviation (SD) method."""
        outlier_indices = []
        for c in self.numerical_features:
            # Calculate mean and standard deviation for each numerical column
            mean, std_dev = self.df[c].mean(), self.df[c].std()
            # Identify cutoff for outliers (mean ± 3*standard_deviation)
            cutoff = std_dev * 3
            lower, upper = mean - cutoff, mean + cutoff
            # Get indices of outliers
            outlier_list_col = self.df[(self.df[c] < lower) | (self.df[c] > upper)].index
            outlier_indices.extend(outlier_list_col)
        # Count indices appearing more than twice
        outlier_indices = Counter(outlier_indices)
        self.multiple_outliers = [i for i, v in outlier_indices.items() if v > 2]
        return self.multiple_outliers

    def detect_outliers_zscore(self):
        """Detect outliers using the Z-Score method."""
        # Calculate z-scores for each numerical column
        z_scores = self.df[self.numerical_features].apply(zscore)
        # Get indices of outliers (z-score > 3 or z-score < -3)
        self.multiple_outliers = z_scores[np.abs(z_scores) > 3].dropna(how='all').index.to_list()
        return self.multiple_outliers

    def detect_outliers(self):
        """Detect outliers using the chosen method."""
        if self.method == "iqr":
            return self.detect_outliers_iqr()
        elif self.method == "sd":
            return self.detect_outliers_sd()
        elif self.method == "zscore":
            return self.detect_outliers_zscore()
        else:
            raise ValueError("Invalid method. Expected one of: 'iqr', 'sd', 'zscore'")

    def drop_outliers(self):
        """Drop the detected outliers from the dataframe."""
        self.df = self.df.drop(self.detect_outliers(), axis = 0)
        return self.df

    def show_outliers(self):
        """Show the detected outliers."""
        if self.multiple_outliers:
            print(f"Detected {len(self.multiple_outliers)} outliers.")
        else:
            print("No outliers detected.")
        return self.multiple_outliers

    def visualize_outliers(self):
        """Visualize the outliers for each numerical feature using boxplots."""
        for feature in self.numerical_features:
            plt.figure(figsize=(10,4))
            sns.boxplot(x=self.df[feature])
            plt.title(f'Boxplot of {feature}')
            plt.show()
