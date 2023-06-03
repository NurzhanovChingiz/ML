from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn_pandas import DataFrameMapper

class DataFramePreprocessing(TransformerMixin, BaseEstimator):
    '''DataFramePreprocessing class fits and transforms all features and returns a Pandas DataFrame'''

    def __init__(self):
        '''
        Constructor for DataFramePreprocessing class.

        Parameters:
        - X (pandas.DataFrame): Input DataFrame.
        '''
        self.X = None
        self.numerical_features = []
        self.categorical_features = []
        self.boolean_features = []
    def get_features(self):
        '''
        Extracts the numeric, categorical, and boolean features from the input DataFrame.

        Parameters:
        - X (pandas.DataFrame): Input DataFrame.

        Returns:
        - None
        '''
        self.numerical_features = self.X.select_dtypes(include=['int16', 'float16', 'int32', 'float32', 'int64', 'float64']).columns
        self.categorical_features = self.X.select_dtypes(include=['object']).columns
        self.boolean_features = self.X.select_dtypes(include=['bool']).columns
        
    def categorical_transformer(self):
        '''
        Creates a list of tuples specifying the transformations for categorical features.

        Returns:
        - list: List of tuples, where each tuple contains a feature name and a transformation pipeline.
        '''
        return [([feature], [SimpleImputer(strategy='most_frequent'), OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-999)]) for feature in self.categorical_features]

    def numerical_transformer(self):
        '''
        Creates a list of tuples specifying the transformations for numerical features.

        Returns:
        - list: List of tuples, where each tuple contains a feature name and a transformation pipeline.
        '''
        return [([feature], [SimpleImputer(strategy='most_frequent'), StandardScaler()]) for feature in self.numerical_features]

    def boolean_transformer(self):
        '''
        Creates a list of tuples specifying the transformations for boolean features.

        Returns:
        - list: List of tuples, where each tuple contains a feature name and a transformation pipeline.
        '''
        return [([feature], [SimpleImputer(strategy='most_frequent'), OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-999)]) for feature in self.boolean_features]

    def mapper(self):
        '''
        Creates a DataFrameMapper object that combines all the feature transformations.

        Returns:
        - DataFrameMapper: DataFrameMapper object that applies the specified transformations to the input data.
        '''
        
        return DataFrameMapper(self.numerical_transformer() + self.categorical_transformer() + self.boolean_transformer(), df_out=True)

    def fit(self, X, y=None):
        '''
        Fits the DataFramePreprocessing transformer on the input data.

        Parameters:
        - X (array-like or DataFrame): Input data to fit the transformer on.
        - y (array-like or None): Target values (ignored).

        Returns:
        - self: Returns the instance itself.
        '''
        X = X.copy()
        self.X = X
        self.get_features()
        self.mapper()
        X = check_array(X, accept_sparse=False)

        self.n_features_in_ = X.shape[1]
        self.n_features_ = X.shape[1]
        self.is_fitted_ = True

        return self

    def transform(self, X):
        '''
        Transforms the input data using the fitted DataFramePreprocessing transformer.
    
        Parameters:
        - X (array-like or DataFrame): Input data to transform.
    
        Returns:
        - array-like or DataFrame: Transformed data.
        '''
        X = X.copy()
        
        check_is_fitted(self, ['is_fitted_'])
        X = check_array(X, accept_sparse=True)
    
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen in `fit`')
        return X
