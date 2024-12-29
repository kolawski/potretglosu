import pandas as pd

from utils.statistical_tests import calculate_spearman_correlation, scatterplot, calculate_correlation, calculate_covariance, regplot


PATH = "/app/results/other_tests/does_embedding_and_latent_mse_correlate_with_parameters_mse.csv"

# Read the CSV file
data = pd.read_csv(PATH)

# Extract columns into arrays
emb_mse = data['embedding mse'].values
lat_mse = data['latents mse'].values

emb_mse_normalized = (emb_mse - emb_mse.min()) / (emb_mse.max() - emb_mse.min())
lat_mse_normalized = (lat_mse - lat_mse.min()) / (lat_mse.max() - lat_mse.min())

normalized_mse_emb_plus_lat = emb_mse_normalized + lat_mse_normalized

param_mse = data['parameters_mse'].values
param_mse_normalized = (param_mse - param_mse.min()) / (param_mse.max() - param_mse.min())

gender = data['gender'].values
normalized_gender = (gender - gender.min()) / (gender.max() - gender.min())

# # correlation between sum and parameters_mse
# print("Correlation between sum of emb and lat mse and parameters mse")
# correlation = calculate_correlation(normalized_mse_emb_plus_lat, param_mse_normalized)
# print(f"Correlation between normalized sum of emb and lat mse and parameters mse: {correlation}")
# covariance = calculate_covariance(normalized_mse_emb_plus_lat, param_mse_normalized)
# print(f"Covariance between normalized sum of emb and lat mse and parameters mse: {covariance}")
# coef, p_value = calculate_spearman_correlation(normalized_mse_emb_plus_lat, param_mse_normalized)
# print(f"Spearman correlation coefficient: {coef}, p-value: {p_value}")

# scatterplot(normalized_mse_emb_plus_lat, param_mse_normalized, "Normalized sum of emb and lat mse", "Normalized parameters mse", "/app/results/other_tests/emb_lat_sum_mse_params_mse.png")
# regplot(normalized_mse_emb_plus_lat, param_mse_normalized, "Normalized sum of emb and lat mse", "Normalized parameters mse", "/app/results/other_tests/emb_lat_sum_mse_params_mse_regplot.png")

# # correlation between embedding mse and parameters_mse
# print("Correlation between embedding mse and parameters mse")
# correlation = calculate_correlation(emb_mse, param_mse)
# print(f"Correlation between embedding mse and parameters mse: {correlation}")
# covariance = calculate_covariance(emb_mse, param_mse)
# print(f"Covariance between embedding mse and parameters mse: {covariance}")
# coef, p_value = calculate_spearman_correlation(emb_mse, param_mse)
# print(f"Spearman correlation coefficient: {coef}, p-value: {p_value}")

# scatterplot(emb_mse, param_mse, "Embedding mse", "Parameters mse", "/app/results/other_tests/emb_mse_params_mse.png")
# regplot(emb_mse, param_mse, "Embedding mse", "Parameters mse", "/app/results/other_tests/emb_mse_params_mse_regplot.png")

# # correlation between latents mse and parameters_mse
# print("Correlation between latents mse and parameters mse")
# correlation = calculate_correlation(lat_mse, param_mse)
# print(f"Correlation between latents mse and parameters mse: {correlation}")
# covariance = calculate_covariance(lat_mse, param_mse)
# print(f"Covariance between latents mse and parameters mse: {covariance}")
# coef, p_value = calculate_spearman_correlation(lat_mse, param_mse)
# print(f"Spearman correlation coefficient: {coef}, p-value: {p_value}")

# scatterplot(lat_mse, param_mse, "Latents mse", "Parameters mse", "/app/results/other_tests/lat_mse_params_mse.png")
# regplot(lat_mse, param_mse, "Latents mse", "Parameters mse", "/app/results/other_tests/lat_mse_params_mse_regplot.png")

# correlation between normalized params and normalized gender
print("Correlation between normalized parameters mse and normalized gender")
correlation = calculate_correlation(param_mse_normalized, normalized_gender)
print(f"Correlation between normalized parameters mse and normalized gender: {correlation}")
covariance = calculate_covariance(param_mse_normalized, normalized_gender)
print(f"Covariance between normalized parameters mse and normalized gender: {covariance}")
coef, p_value = calculate_spearman_correlation(param_mse_normalized, normalized_gender)
print(f"Spearman correlation coefficient: {coef}, p-value: {p_value}")

regplot(param_mse_normalized, normalized_gender, "Normalized parameters mse", "Normalized gender abs", "/app/results/other_tests/norm_gender_abs_norm_params_mse_regplot.png")
