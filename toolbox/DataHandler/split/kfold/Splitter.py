import pandas as pd
from sklearn.model_selection import KFold

class Splitter:
    """
    A class used to split data into train/test sets.
    """
    def __init__(self, X, y, n_splits=5, random_state=42, shuffle=True):
        """
        The constructor for Splitter class.

        Parameters:
           X (DataFrame): Features
           y (Series/DataFrame): Target variable
           n_splits (int): Number of folds. Must be at least 2.
           random_state (int): Random state for reproducibility.
           shuffle (bool): Whether to shuffle the data before splitting into batches.
        """
        # Ensure the data size is a multiple of 2*n_splits
        total_samples = len(X)
        surplus_samples = total_samples % (2 * n_splits)
    
        if surplus_samples > 0:
            X = X[:-surplus_samples]
            y = y[:-surplus_samples]
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle
        self.kfold = KFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=self.shuffle)
        
    def split_data(self):
        X_train, X_test, y_train, y_test = [], [], [], []
        
        for train_index, test_index in self.kfold.split(self.X):
            X_train_fold, X_test_fold = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train_fold, y_test_fold = self.y.iloc[train_index], self.y.iloc[test_index]
            
            X_train.append(X_train_fold)
            X_test.append(X_test_fold)
            y_train.append(y_train_fold)
            y_test.append(y_test_fold)
        
        return  X_train, X_test, y_train, y_test
