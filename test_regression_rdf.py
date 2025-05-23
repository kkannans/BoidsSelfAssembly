import numpy as np
import matplotlib.pyplot as plt
from utils import load_target_rdfs, get_first_peak_from_rdf
from loss_functions import loss_rdf
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import jensenshannon
import os
import config

def reconstruct_rdf_gaussian(mu, sigma, radii , peak_position):
    """
    Reconstruct an RDF using a Gaussian shape centered on the typical peak position.
    
    Parameters:
    -----------
    mu : float
        Mean parameter (controls overall height)
    sigma : float
        Standard deviation parameter (controls width/spread)
    radii : array-like
        Radii values for the RDF
    peak_position : float
        Position of the peak in the RDF
        
    Returns:
    --------
    reconstructed_rdf : ndarray
        Reconstructed RDF with shape matching radii
    """
    # Create a Gaussian peak
    peak_height = mu 
    peak_width = sigma
    
    # Generate Gaussian distribution
    gaussian = peak_height * np.exp(-0.5 * ((radii - peak_position) / peak_width) ** 2)
    
    # Add baseline to ensure proper mean
    baseline = mu - np.mean(gaussian)
    reconstructed_rdf = gaussian + baseline
    
    # Ensure RDF doesn't go below zero and has correct mean
    reconstructed_rdf = np.maximum(reconstructed_rdf, 0)
    
    # Fine-tune to get exact mean
    current_mean = np.mean(reconstructed_rdf)
    scale_factor = mu / current_mean if current_mean > 0 else 1.0
    reconstructed_rdf *= scale_factor
    
    # Fine-tune to get exact standard deviation
    current_std = np.std(reconstructed_rdf)
    if current_std > 0:
        normalized = (reconstructed_rdf - np.mean(reconstructed_rdf)) / current_std
        reconstructed_rdf = normalized * sigma + mu
    
    return reconstructed_rdf

# Create output directories
os.makedirs("./regression_rdf_bw_longer", exist_ok=True)

# Load RDF data
g_r, radii, N_pairs = load_target_rdfs()

# Extract mean and standard deviation of RDFs
mu = np.array([np.mean(g_r[i]) for i in range(len(g_r))])
sigma = np.array([np.std(g_r[i]) for i in range(len(g_r))])

# Compute μ and σ for each timestep in training data
num_train = config.END_FRAME
mu_train = np.array([np.mean(g_r[i]) for i in range(num_train)])
sigma_train = np.array([np.std(g_r[i]) for i in range(num_train)])

# Compute μ and σ for each timestep in test data
mu_test = np.array([np.mean(g_r[i]) for i in range(num_train, len(g_r))])
sigma_test = np.array([np.std(g_r[i]) for i in range(num_train, len(g_r))])

# Linear regression for μ
model_mu = LinearRegression()
X_mu = np.arange(len(mu_train)).reshape(-1, 1)
model_mu.fit(X_mu, mu_train)

# Linear regression for σ
model_sigma = LinearRegression()
X_sigma = np.arange(len(sigma_train)).reshape(-1, 1)
model_sigma.fit(X_sigma, sigma_train)

# Predict for test timepoints
X_test = np.arange(len(mu_train), len(mu_train) + len(mu_test)).reshape(-1, 1)
mu_pred = model_mu.predict(X_test)
sigma_pred = model_sigma.predict(X_test)

# Predict for training timepoints
X_train = np.arange(len(mu_train)).reshape(-1, 1)
mu_pred_train = model_mu.predict(X_train)
sigma_pred_train = model_sigma.predict(X_train)

# Calculate average peak position from training data only
avg_peak_position = np.mean([get_first_peak_from_rdf(g_r[i]) for i in range(num_train)])

# Reconstruct RDFs for training set using Gaussian approach
reconstructed_rdfs_train = []
for i in range(len(mu_pred_train)):
    # Use the same average peak position for all training reconstructions
    rdf = reconstruct_rdf_gaussian(mu_pred_train[i], sigma_pred_train[i], radii, avg_peak_position)
    reconstructed_rdfs_train.append(rdf)
reconstructed_rdfs_train = np.array(reconstructed_rdfs_train)

# Reconstruct RDFs for test set using Gaussian approach
reconstructed_rdfs = []
for i in range(len(mu_pred)):
    # Use the same average peak position for all test reconstructions
    rdf = reconstruct_rdf_gaussian(mu_pred[i], sigma_pred[i], radii, avg_peak_position)
    reconstructed_rdfs.append(rdf)
reconstructed_rdfs = np.array(reconstructed_rdfs)

# compute loss for training set
loss_train = []
for t in range(len(reconstructed_rdfs_train)):
    loss_train.append(jensenshannon(reconstructed_rdfs_train[t], g_r[t]))

# compute loss for test set
loss_test = []
for t in range(len(reconstructed_rdfs)):
    loss_test.append(jensenshannon(reconstructed_rdfs[t], g_r[t + len(reconstructed_rdfs_train)]))

# plot losses
plt.figure(figsize=(10, 5))
plt.plot(loss_train, label='Training Loss')
plt.plot(range(len(loss_train), len(loss_train) + len(loss_test)), loss_test, label='Test Loss')
plt.xlabel('Time Step')
plt.ylabel('Loss (Jensen-Shannon Distance)')
plt.title('RDF Reconstruction Loss')
plt.legend()
# save plot as png in regression_rdf folder
plt.savefig('regression_rdf_bw_longer/losses_linear_regression.png')
plt.close()
# save losses as losses_linear_regression.npy in regression_rdf folder
# combine losses into one array
losses = np.concatenate((loss_train, loss_test))
np.save('regression_rdf_bw_longer/losses_linear_regression.npy', losses)

