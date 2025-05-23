import os
import numpy as np
import regression
from utils import load_pairwise_distances
from regression import TimeSeriesRegression

def main():
    pairwise_distances_video = load_pairwise_distances()
    # compute the avg pair across all frames
    avg_pairwise_distances = [np.mean(pairwise_distances_video[i]) for i in range(len(pairwise_distances_video))]
    avg_pairwise_distances = np.array(avg_pairwise_distances) # (721,1)
    # Initialize model
    model = TimeSeriesRegression(avg_pairwise_distances, train_size=576)
    
    # 1. Linear Regression
    print("Fitting Linear Regression...")
    params = model.fit_linear_regression()
    linear_preds = model.predict_linear()
    linear_metrics = model.evaluate_model(linear_preds)
    print(f"Linear Regression parameters: {params}")
    print(f"Linear Regression metrics: {linear_metrics}")
    
    # 2. Nonlinear Regression - Example with quadratic function
    print("\nFitting Nonlinear Regression (Quadratic)...")
    def quadratic_model(x, a, b, c):
        return a * x**2 + b * x + c
    
    nonlinear_params = model.fit_custom_nonlinear(quadratic_model, p0=[0.001, 0.01, 0])
    nonlinear_preds = model.predict_nonlinear()
    nonlinear_metrics = model.evaluate_model(nonlinear_preds)
    print(f"Nonlinear Regression parameters: {nonlinear_params}")
    print(f"Nonlinear Regression metrics: {nonlinear_metrics}")
    
    # 3. Support Vector Regression
    print("\nFitting Support Vector Regression...")
    model.fit_svr(kernel='rbf', C=100, epsilon=0.1)
    svr_preds = model.predict_svr()
    svr_metrics = model.evaluate_model(svr_preds)
    print(f"SVR metrics: {svr_metrics}")
    
    os.makedirs('./regression_models_bw_longer', exist_ok=True)
    # Plot results
    model.plot_results({
        'Linear': linear_preds,
        'Quadratic': nonlinear_preds,
        'SVR': svr_preds
    }, save_path='./regression_models_bw_longer/regression_results.png')

    # get model output from both training and testing.
    linear_preds_train = model.predict_linear(indices=model.train_indices)
    linear_preds_test = model.predict_linear(indices=model.test_indices)
    linear_combined = np.concatenate([linear_preds_train, linear_preds_test])
    nonlinear_preds_train = model.predict_nonlinear(indices=model.train_indices)
    nonlinear_preds_test = model.predict_nonlinear(indices=model.test_indices)
    nonlinear_combined = np.concatenate([nonlinear_preds_train, nonlinear_preds_test])
    svr_preds_train = model.predict_svr(indices=model.train_indices)
    svr_preds_test = model.predict_svr(indices=model.test_indices)
    svr_combined = np.concatenate([svr_preds_train, svr_preds_test])

    # Compute losses
    losses = model.compute_losses({
        'Linear': linear_combined,
        'Quadratic': nonlinear_combined,
        'SVR': svr_combined
    })
    # save losses to regression_models_bw_longer/linear_model_avg_pairwise_distances_losses.npy
    os.makedirs('./regression_models_bw_longer', exist_ok=True)
    np.save('./regression_models_bw_longer/linear_model_avg_pairwise_distances_losses.npy', losses['Linear'])
    np.save('./regression_models_bw_longer/quadratic_model_avg_pairwise_distances_losses.npy', losses['Quadratic'])
    np.save('./regression_models_bw_longer/svr_model_avg_pairwise_distances_losses.npy', losses['SVR'])

if __name__ == "__main__":
    main()