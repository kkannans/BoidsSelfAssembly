import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import config
from utils import load_pairwise_distances
import os

class TimeSeriesRegression:
    """
    A class that implements linear and SV regression for time series data
    using SciPy and scikit-learn.
    """
    
    def __init__(self, data, train_size=config.END_FRAME):
        """
        Initialize with time series data.
        
        Parameters:
        -----------
        data : array-like
            The complete time series data
        train_size : int, default=576
            Number of time steps to use for training
        """
        self.data = np.array(data)
        self.train_size = train_size
        self.test_size = len(data) - train_size
        
        # Split data into train and test sets
        self.train_data = self.data[:train_size]
        self.test_data = self.data[train_size:]
        
        # Create time indices
        self.train_indices = np.arange(train_size)
        self.test_indices = np.arange(train_size, len(data))
        self.all_indices = np.arange(len(data))
        
        # Initialize scaler for SVR
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
    
    def linear_model(self, x, a, b):
        """Simple linear function: y = ax + b"""
        return a * x + b
    
    def fit_linear_regression(self):
        """
        Fit a linear regression model using SciPy's curve_fit.
        
        Returns:
        --------
        params : tuple
            The fitted parameters (a, b) of the linear model
        """
        # Fit the model to training data
        params, covariance = curve_fit(
            self.linear_model, 
            self.train_indices, 
            self.train_data
        )
        
        self.linear_params = params
        return params
    
    def predict_linear(self, indices=None):
        """
        Make predictions using the fitted linear model.
        
        Parameters:
        -----------
        indices : array-like, optional
            The indices for which to make predictions
            
        Returns:
        --------
        predictions : ndarray
            Predicted values
        """
        if not hasattr(self, 'linear_params'):
            raise ValueError("Model not fitted. Call fit_linear_regression() first.")
            
        if indices is None:
            indices = self.all_indices
            
        return self.linear_model(indices, *self.linear_params)
    
    def fit_custom_nonlinear(self, model_func, p0=None):
        """
        Fit a custom nonlinear regression model using SciPy's curve_fit.
        
        Parameters:
        -----------
        model_func : callable
            A function that takes x and parameters and returns predicted y
        p0 : array-like, optional
            Initial parameter guesses
            
        Returns:
        --------
        params : tuple
            The fitted parameters of the nonlinear model
        """
        # Fit the model to training data
        params, covariance = curve_fit(
            model_func, 
            self.train_indices, 
            self.train_data,
            p0=p0
        )
        
        self.nonlinear_func = model_func
        self.nonlinear_params = params
        return params
    
    def predict_nonlinear(self, indices=None):
        """
        Make predictions using the fitted nonlinear model.
        
        Parameters:
        -----------
        indices : array-like, optional
            The indices for which to make predictions
            
        Returns:
        --------
        predictions : ndarray
            Predicted values
        """
        if not hasattr(self, 'nonlinear_params') or not hasattr(self, 'nonlinear_func'):
            raise ValueError("Model not fitted. Call fit_custom_nonlinear() first.")
            
        if indices is None:
            indices = self.all_indices
            
        return self.nonlinear_func(indices, *self.nonlinear_params)
    
    def fit_svr(self, kernel='rbf', C=100, epsilon=0.1, gamma='scale'):
        """
        Fit a Support Vector Regression model.
        
        Parameters:
        -----------
        kernel : str, default='rbf'
            The kernel type to be used in the algorithm
        C : float, default=100
            Regularization parameter
        epsilon : float, default=0.1
            Epsilon in the epsilon-SVR model
        gamma : str or float, default='scale'
            Kernel coefficient
            
        Returns:
        --------
        model : SVR
            The fitted SVR model
        """
        # Reshape and scale the data
        X_train = self.train_indices.reshape(-1, 1)
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(self.train_data.reshape(-1, 1)).ravel()
        
        # Create and fit the SVR model
        self.svr_model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
        self.svr_model.fit(X_train_scaled, y_train_scaled)
        
        return self.svr_model
    
    def predict_svr(self, indices=None):
        """
        Make predictions using the fitted SVR model.
        
        Parameters:
        -----------
        indices : array-like, optional
            The indices for which to make predictions
            
        Returns:
        --------
        predictions : ndarray
            Predicted values
        """
        if not hasattr(self, 'svr_model'):
            raise ValueError("Model not fitted. Call fit_svr() first.")
            
        if indices is None:
            indices = self.all_indices
            
        # Reshape and scale the test data
        X_scaled = self.scaler_X.transform(indices.reshape(-1, 1))
        
        # Make predictions and inverse transform to original scale
        y_pred_scaled = self.svr_model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        return y_pred
    
    def evaluate_model(self, predictions, actual=None):
        """
        Evaluate a model's predictions.
        
        Parameters:
        -----------
        predictions : array-like
            Predicted values
        actual : array-like, optional
            Actual values to compare against
            
        Returns:
        --------
        metrics : dict
            Dictionary containing error metrics
        """
        if actual is None:
            # Evaluate on test set by default
            test_preds = predictions[self.train_size:]
            actual = self.test_data
        else:
            # If actual is provided, use that for comparison
            test_preds = predictions
            
        rmse = np.sqrt(mean_squared_error(actual, test_preds))
        mae = mean_absolute_error(actual, test_preds)
        mape = np.mean(np.abs((actual - test_preds) / actual)) * 100
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
    
    def plot_results(self, predictions_dict, save_path="./regression_bw_longer/regression_results.png"):
        """
        Plot the original data and model predictions.
        
        Parameters:
        -----------
        predictions_dict : dict
            Dictionary with model names as keys and predictions as values
        """
        plt.figure(figsize=(12, 6))
        
        # Plot original data
        plt.plot(self.all_indices, self.data, 'k-', label='Original Data')
        
        # Plot training/test split
        plt.axvline(x=self.train_size, color='r', linestyle='--', alpha=0.5)
        plt.text(self.train_size + 5, np.max(self.data), 'Train/Test Split', 
                 color='r', alpha=0.7)
        
        # Plot model predictions
        for name, preds in predictions_dict.items():
            plt.plot(self.all_indices, preds, '--', label=f'{name} Predictions')
        
        plt.legend()
        plt.title('Time Series Regression Results')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path)

    def compute_losses(self, predictions_dict):
        """
        Compute MAE losses for each model's predictions at each time step.
        
        Parameters:
        -----------
        predictions_dict : dict
            Dictionary with model names as keys and predictions as values
            
        Returns:
        --------
        losses : dict
            Dictionary with model names as keys and lists of MAE losses as values
        """
        losses = {}
        num_steps = len(self.data)
        
        for model_name, preds in predictions_dict.items():
            model_losses = []
            for t in range(num_steps):
                # Compute MAE between model prediction and actual data at time t
                mae = mean_absolute_error([self.data[t]], [preds[t]])
                model_losses.append(mae)
            losses[model_name] = model_losses
            
        return losses
