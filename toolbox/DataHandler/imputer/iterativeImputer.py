from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from category_encoders import CatBoostEncoder
import numpy as np
from feature_engine.selection import DropHighPSIFeatures, DropDuplicateFeatures, DropCorrelatedFeatures, DropConstantFeatures
from lightgbm import LGBMRegressor

class IterativeImputer(BaseEstimator, TransformerMixin):

    def __init__(self, max_iterations=10, model_params={}):
        self.max_iterations = max_iterations
        self.model_params = model_params
        self.features = None
        self.mapper_ = self.mapper()
        self.rows_miss = None
        self.error_minimize = None
        self.lgb_params = {
            'n_estimators': 100,
            'max_depth': 5,
            "num_leaves": 16,
            'learning_rate': 0.05,
            'subsample': 0.7,
            'colsample_bytree': 0.8,
            #'reg_alpha': 0.25,
            'reg_lambda': 5e-07,
            'objective': 'regression_l2',
            'metric': 'mean_squared_error',
#             'boosting_type': 'gbdt',
#             'random_state': 42,
        }
    def fit(self, X, y=None):
        """Fit the imputer to the data and identify features with missing values."""
        self.features = [f for f in X.columns if X[f].isna().sum() > 0]
#         print(self.features)
        return self

    def transform(self, X):
        X_temp=X.copy()
        """Impute missing values using iterative imputation."""
        
        missing_rows = self.store_missing_rows(X_temp, self.features)
        for f in self.features:
            X_temp[f]=X_temp[f].fillna(X_temp[f].mean())
        cat_features = [f for f in X_temp.columns if not pd.api.types.is_numeric_dtype(X_temp[f])]
        dictionary = {feature: [] for feature in self.features}
        if len(self.features)>0:
            for iteration in tqdm(range(self.max_iterations), desc="Iterations"):
    #             print(1)
                for feature in self.features:
            #                 # Skip features with no missing values
                    self.rows_miss =  missing_rows[feature].index
                    missing_temp = X_temp.loc[self.rows_miss].copy()
                    non_missing_temp = X_temp.drop(index=self.rows_miss).copy()
                    y_pred_prev=missing_temp[feature]
                    missing_temp = missing_temp.drop(columns=[feature])
                    # Step 3: Use the remaining features to predict missing values using Random Forests
                    X_train = non_missing_temp.drop(columns=[feature])
                    y_train = non_missing_temp[[feature]]
                    mapper_pipe = self.mapper_
                    
                    X_train = mapper_pipe.fit_transform(X_train, y_train)
                    model= lgb.LGBMRegressor(**self.lgb_params,random_state=42,boosting_type='dart',verbose=0,
            force_row_wise=True,)
                    model.fit(X_train, y_train)
                    # Step 4: Predict missing values for the feature and update all N features
                    y_pred = model.predict(mapper_pipe.transform(missing_temp))
                    X_temp.loc[self.rows_miss, feature] = y_pred
                    self.error_minimize=self.rmse(y_pred,y_pred_prev)
                    dictionary[feature].append(self.error_minimize)  # Append the error_minimize value
#                     print(self.error_minimize)
    #                 print(2)
            X_temp[self.features] = np.array(X_temp.iloc[:X_temp.shape[0]][self.features])
            X_temp = X_temp.drop(columns=cat_features)    
            return X_temp
        return X
    def store_missing_rows(self, df, features):
        """Function stores where missing values are located for given set of features."""
        missing_rows = {}

        for feature in features:
            missing_rows[feature] = df[df[feature].isnull()]
        
        return missing_rows
    def get_feature_names_out(self):
        pass
    def rmse(self, y1, y2):
        from sklearn.metrics import mean_squared_error
        """RMSE Evaluator"""
        return (np.sqrt(mean_squared_error(np.array(y1), np.array(y2))))

    def mapper(self):
        self.cat_imputer =  SimpleImputer(strategy='most_frequent')
        # Scale and encoding
        self.scaler = RobustScaler(quantile_range=(10.0, 90.0))
        self.encoder =  CatBoostEncoder(random_state = 42, drop_invariant=True)
        num_cols = make_column_selector(dtype_include=np.number)
        cat_cols = make_column_selector(dtype_include=object)
        categorical_imputer = Pipeline([
            ('Imputer', self.cat_imputer),
            ('Encoder', self.encoder)  # Adding encoding for categorical data
        ])
        imput = ColumnTransformer([
            ('categorical_imputer', categorical_imputer, cat_cols),
            
        ],
            remainder='passthrough' # remainder='passthrough' to keep columns not specified
        )  
    
        pipe = Pipeline([
            
            ('Imputer', imput),
            ('DropDuplicateFeatures', DropDuplicateFeatures()),
            ('DropConstantFeatures', DropConstantFeatures()),
            ('DropCorrelatedFeatures', DropCorrelatedFeatures(threshold=0.95)),
            ('scaler', self.scaler),  # Applies scaling to numerical features only
        ])

        pipe.set_output(transform="pandas")
        return pipe
