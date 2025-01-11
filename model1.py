import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t

data = {
    't': [
        0.01, 0.03, 0.04, 0.05, 0.07, 0.08, 0.09, 0.11, 0.12, 0.13, 0.15, 0.16, 0.17, 0.19,
        0.20, 0.21, 0.23, 0.24, 0.25, 0.27, 0.28, 0.29, 0.31, 0.32, 0.33, 0.35, 0.36, 0.37,
        0.39, 0.40, 0.41, 0.43, 0.44, 0.45, 0.47, 0.48, 0.49, 0.51, 0.52, 0.53, 0.55, 0.56,
        0.57, 0.59, 0.60, 0.61, 0.63, 0.64, 0.65, 0.67, 0.68, 0.69, 0.71, 0.72, 0.73, 0.75,
        0.76, 0.77, 0.79, 0.80, 0.81, 0.83, 0.84, 0.85, 0.87, 0.88, 0.89, 0.91, 0.92, 0.93,
        0.95, 0.96, 0.97, 0.99, 1.00
    ],
    'y': [
        8.4201, 8.5624, 8.6420, 8.7681, 8.8110, 8.9750, 9.0337, 9.1791, 9.3122, 9.3899, 9.5486, 9.6763, 
        9.7820, 9.9307, 10.0614, 10.2165, 10.3552, 10.4393, 10.5861, 10.7517, 10.8842, 11.0062, 11.1657, 
        11.3202, 11.5593, 11.6538, 11.8317, 11.9710, 12.1655, 12.3367, 12.5526, 12.6882, 12.9224, 13.0763, 
        13.3090, 13.4396, 13.6745, 13.8529, 14.0600, 14.2940, 14.5535, 14.7239, 14.9380, 15.1779, 15.4129, 
        15.6788, 15.9082, 16.1593, 16.4322, 16.6636, 16.9461, 17.2595, 17.4573, 17.7848, 18.0744, 18.3205, 
        18.6276, 19.0191, 19.3157, 19.6301, 19.9632, 20.2039, 20.5709, 20.9546, 21.2796, 21.6742, 21.9976, 
        22.3817, 22.8038, 23.1386, 23.5531, 23.9421, 24.4370, 24.8160, 25.2312
    ]
}

        df = pd.DataFrame(data)
        time_values = df['t'].values
        measurement_values = df['y'].values
        scaled_time = (time_values - np.min(time_values)) / (np.max(time_values) - np.min(time_values)) 


def exponential_model(time, params):
    alpha0, alpha1, beta1, alpha2, beta2 = params
    beta1_exp = np.exp(np.clip(beta1 * time, -50, 50))  
    beta2_exp = np.exp(np.clip(beta2 * time, -50, 50))
    return alpha0 + alpha1 * beta1_exp + alpha2 * beta2_exp


def jacobian_exponential(time, params):
    alpha0, alpha1, beta1, alpha2, beta2 = params
    J = np.zeros((len(time), len(params)))
    J[:, 0] = 1  
    J[:, 1] = np.exp(np.clip(beta1 * time, -50, 50))  
    J[:, 2] = alpha1 * time * np.exp(np.clip(beta1 * time, -50, 50))  
    J[:, 3] = np.exp(np.clip(beta2 * time, -50, 50))  
    J[:, 4] = alpha2 * time * np.exp(np.clip(beta2 * time, -50, 50))  
    return J

def gauss_newton_exponential(time, measurements, initial_params, tol=1e-6, max_iter=100, reg_lambda=1e-4):
    params = np.array(initial_params, dtype=float)
    for iteration in range(max_iter):
        residuals = measurements - exponential_model(time, params)
        J = jacobian_exponential(time, params)
        delta = np.linalg.inv(J.T @ J + reg_lambda * np.eye(J.shape[1])) @ J.T @ residuals
        params += delta
        params = np.clip(params, -10, 10)
        if np.linalg.norm(delta) < tol:
            break
    return params


    initial_guess = [1, 1, 0.5, 1, 0.5]
    fitted_params = gauss_newton_exponential(scaled_time, measurement_values, initial_guess)

    residuals = measurement_values - exponential_model(scaled_time, fitted_params)
    lse = np.sum(residuals**2)
    num_data_points = len(measurement_values)
    num_parameters = 5
    sigma_squared = lse / (num_data_points - num_parameters)


    J = jacobian_exponential(scaled_time, fitted_params)
    fim = np.linalg.inv(J.T @ J)
    covariance_matrix = fim * sigma_squared
    std_errors = np.sqrt(np.diag(covariance_matrix))
    alpha = 0.05
    t_critical = t.ppf(1 - alpha / 2, num_data_points - num_parameters)

    confidence_intervals = []
    for i, std_err in enumerate(std_errors):
        ci_lower = fitted_params[i] - t_critical * std_err
        ci_upper = fitted_params[i] + t_critical * std_err
        confidence_intervals.append((ci_lower, ci_upper))


    print("Fitted Parameters:")
    param_names = ["α₀", "α₁", "β₁", "α₂", "β₂"]
    for i, param in enumerate(fitted_params):
        print(f"{param_names[i]} = {param:.4f}")

    print(f"Least Squares Error (LSE): {lse:.4f}")
    print(f"Estimated Variance (σ²): {sigma_squared:.4f}")
    print("\nConfidence Intervals (95%):")
    for i, (lower, upper) in enumerate(confidence_intervals):
        print(f"{param_names[i]} = {fitted_params[i]:.4f}: ({lower:.4f}, {upper:.4f})")


    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    axs[0].scatter(time_values, measurement_values, label='Observed Data', color='blue')
    axs[0].plot(time_values, exponential_model(scaled_time, fitted_params), label='Fitted Model', color='red')
    axs[0].set_xlabel('Time (t)')
    axs[0].set_ylabel('Measurement (y)')
    axs[0].set_title('Observed Data and Fitted Exponential Model')
    axs[0].legend()


    axs[1].scatter(time_values, residuals, color='green', marker='o', s=30)
    axs[1].axhline(y=0, color='black', linestyle='--')  
    axs[1].set_xlabel('Time (t)')
    axs[1].set_ylabel('Residuals')
    axs[1].set_title('Residuals of the Fitted Model')

    plt.tight_layout()  
    plt.show()
