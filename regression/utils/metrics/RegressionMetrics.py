from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np

class RegressionMetrics:
    """
    A class for computing and printing regression evaluation metrics.

    Args:
        pipeline_or_model: The regression pipeline or model.
        X_test: The test features.
        y_test: The test labels.
        X_val: The validation features.
        y_val: The validation labels.
        style: Flag indicating whether to apply styling to the output. Default is False.
        is_model: Flag indicating whether the provided object is a regression model. Default is False.
    """

    def __init__(self, pipeline_or_model, X_test, y_test, X_val, y_val, style=False, is_model=False):
        if is_model:
            self.model = pipeline_or_model
        else:
            self.pipeline = pipeline_or_model
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val
        self.y_pred = None
        self.style = style

    def mean_absolute_error(self):
        """
        Calculates the mean absolute error (MAE).

        Returns:
            The mean absolute error value.
        """
        return mean_absolute_error(self.y_val, self.y_pred)

    def mean_absolute_percentage_error(self):
        """
        Calculates the mean absolute percentage error (MAPE).

        Returns:
            The mean absolute percentage error value.
        """
        return mean_absolute_percentage_error(self.y_val, self.y_pred)

    def mean_squared_error(self):
        """
        Calculates the mean squared error (MSE).

        Returns:
            The mean squared error value.
        """
        return mean_squared_error(self.y_val, self.y_pred)

    def mean_root_mean_squared_error(self):
        """
        Calculates the root mean squared error (RMSE).

        Returns:
            The root mean squared error value.
        """
        return mean_squared_error(self.y_val, self.y_pred, squared=False)

    def r2_test(self):
        """
        Calculates the R-squared score for the test set.

        Returns:
            The R-squared score for the test set.
        """
        if hasattr(self, 'model'):
            return self.model.score(self.X_test, self.y_test)
        else:
            return self.pipeline.score(self.X_test, self.y_test)

    def r2_val(self):
        """
        Calculates the R-squared score for the validation set.

        Returns:
            The R-squared score for the validation set.
        """
        return r2_score(self.y_val, self.y_pred)

    def aic(self):
        """
        Calculates the Akaike Information Criterion (AIC).

        Returns:
            The Akaike Information Criterion value.
        """
        y_pred = self.predict()
        n_params = len(self.pipeline.named_steps) if hasattr(self, 'pipeline') else 0
        n = len(self.y_val)
        mse = mean_squared_error(self.y_val, y_pred)
        aic = 2 * n_params - 2 * np.log(mse) + n_params * np.log(n)
        return aic

    def bic(self):
        """
        Calculates the Bayesian Information Criterion (BIC).

        Returns:
            The Bayesian Information Criterion value.
        """
        y_pred = self.predict()
        n_params = len(self.pipeline.named_steps) if hasattr(self, 'pipeline') else 0
        n = len(self.y_val)
        mse = mean_squared_error(self.y_val, y_pred)
        bic = -2 * np.log(mse) + n_params * np.log(n)
        return bic

    def std(self):
        """
        Calculates the standard deviation of the predicted values.

        Returns:
            The standard deviation value.
        """
        return self.y_pred.std()

    def mean(self):
        """
        Calculates the mean of the predicted values.

        Returns:
            The mean value.
        """
        return self.y_pred.mean()

    def predict(self):
        """
        Performs prediction on the validation set.

        Returns:
            The predicted values.
        """
        if hasattr(self, 'model'):
            self.y_pred = self.model.predict(self.X_val)
        else:
            self.y_pred = self.pipeline.predict(self.X_val)
        return self.y_pred

    def set_frame_style(self, df, caption=""):
        """
        Helper function to set dataframe presentation style.

        Args:
            df: The DataFrame to style.
            caption: The caption for the styled DataFrame. Default is an empty string.

        Returns:
            The styled DataFrame.
        """
        return df.style.background_gradient(
            cmap='coolwarm').set_caption(caption).set_table_styles([{
                'selector':
                'caption',
                'props': [('color', 'Blue'), ('font-size', '28px'),
                          ('font-weight', 'bold')]
            }])

    def run(self):
        """
        Runs the regression metrics calculation and printing.

        Returns:
            The DataFrame containing the metric values.
        """
        try:
            self.predict()
            metrics = {
                "Test R-squared": self.r2_test(),
                "Val R-squared": self.r2_val(),
                "MAE": self.mean_absolute_error(),
                "MSE": self.mean_squared_error(),
                "RMSE": self.mean_root_mean_squared_error(),
                "MAPE": self.mean_absolute_percentage_error(),
                "AIC": self.aic(),
                "BIC": self.bic(),
                "Std Deviation": self.std(),
                "Mean": self.mean()
            }

            df_metrics = pd.DataFrame.from_dict(metrics,
                                                orient="index",
                                                columns=["Value"])
            df_metrics.index.name = "Metric"

            if self.style:
                return self.set_frame_style(df_metrics)
            else:
                return df_metrics
        except Exception as e:
            print("An error occurred:", str(e))

