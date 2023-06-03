The RegressionMetrics class is designed to compute and print regression evaluation metrics.

The class takes the following arguments in its constructor:

pipeline: The regression pipeline model.
X_test: The test features.
y_test: The test labels.
X_val: The validation features.
y_val: The validation labels.
style: A flag indicating whether to apply styling to the output. Default is False.
The class has the following methods:

mean_absolute_error: Calculates the mean absolute error (MAE).
mean_absolute_percentage_error: Calculates the mean absolute percentage error (MAPE).
mean_squared_error: Calculates the mean squared error (MSE).
mean_root_mean_squared_error: Calculates the root mean squared error (RMSE).
r2_test: Calculates the R-squared score for the test set.
r2_val: Calculates the R-squared score for the validation set.
aic: Calculates the Akaike Information Criterion (AIC).
bic: Calculates the Bayesian Information Criterion (BIC).
std: Calculates the standard deviation of the predicted values.
mean: Calculates the mean of the predicted values.
predict: Performs prediction on the validation set.
set_frame_style: A helper function to set dataframe presentation style.
run: Runs the regression metrics calculation and printing.
The predict method is called internally by the run method to perform prediction on the validation set before calculating the metrics.

The run method calculates various regression evaluation metrics using the methods described above and prints them in a DataFrame format. The metrics include test R-squared, validation R-squared, MAE, MSE, RMSE, MAPE, AIC, BIC, standard deviation, and mean.

If the style flag is set to True, the DataFrame is displayed with styling using the set_frame_style helper function. Otherwise, the DataFrame is printed as is.

If any exception occurs during the calculation or printing of metrics, an error message is displayed.

This class provides a convenient way to calculate and display regression evaluation metrics using the scikit-learn library and pandas DataFrame.
