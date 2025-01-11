import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t

data = {
    't': [
        0.01, 0.03, 0.04, 0.05, 0.07, 0.08, 0.09, 0.11, 0.12, 0.13, 0.15, 0.16, 
        0.17, 0.19, 0.20, 0.21, 0.23, 0.24, 0.25, 0.27, 0.28, 0.29, 0.31, 0.32, 
        0.33, 0.35, 0.36, 0.37, 0.39, 0.40, 0.41, 0.43, 0.44, 0.45, 0.47, 0.48, 
        0.49, 0.51, 0.52, 0.53, 0.55, 0.56, 0.57, 0.59, 0.60, 0.61, 0.63, 0.64, 
        0.65, 0.67, 0.68, 0.69, 0.71, 0.72, 0.73, 0.75, 0.76, 0.77, 0.79, 0.80, 
        0.81, 0.83, 0.84, 0.85, 0.87, 0.88, 0.89, 0.91, 0.92, 0.93, 0.95, 0.96, 
        0.97, 0.99, 1.00
    ],
    'y': [
        8.4201, 8.5624, 8.6420, 8.7681, 8.8110, 8.9750, 9.0337, 9.1791, 9.3122, 
        9.3899, 9.5486, 9.6763, 9.7820, 9.9307, 10.0614, 10.2165, 10.3552, 10.4393, 
        10.5861, 10.7517, 10.8842, 11.0062, 11.1657, 11.3202, 11.5593, 11.6538, 
        11.8317, 11.9710, 12.1655, 12.3367, 12.5526, 12.6882, 12.9224, 13.0763, 
        13.3090, 13.4396, 13.6745, 13.8529, 14.0600, 14.2940, 14.5535, 14.7239, 
        14.9380, 15.1779, 15.4129, 15.6788, 15.9082, 16.1593, 16.4322, 16.6636, 
        16.9461, 17.2595, 17.4573, 17.7848, 18.0744, 18.3205, 18.6276, 19.0191, 
        19.3157, 19.6301, 19.9632, 20.2039, 20.5709, 20.9546, 21.2796, 21.6742, 
        21.9976, 22.3817, 22.8038, 23.1386, 23.5531, 23.9421, 24.4370, 24.8160, 
        25.2312
    ]
}

df = pd.DataFrame(data)

def polynomial_model(t, beta0, beta1, beta2, beta3, beta4):
    return beta0 + beta1 * t + beta2 * t**2 + beta3 * t**3 + beta4 * t**4

def compute_jacobian(t, params):
    J = np.zeros((len(t), len(params)))
    J[:, 0] = 1  
    J[:, 1] = t 
    J[:, 2] = t**2  
    J[:, 3] = t**3  
    J[:, 4] = t**4  
    return J

def gauss_newton_method(t, y, model, jacobian, initial_guess, max_iterations=100, tolerance=1e-6):
    params = np.array(initial_guess, dtype=float)
    
    for iteration in range(max_iterations):
        
        residuals = y - model(t, *params)
        
        
        J = jacobian(t, params)
        
        
        delta = np.linalg.inv(J.T @ J) @ J.T @ residuals
        params += delta
        
        if np.linalg.norm(delta) < tolerance:
            break
    
    return params

initial_guess = [0.1, 0.1, 0.1, 0.1, 0.1]

optimized_params = gauss_newton_method(df['t'].values, df['y'].values, polynomial_model, compute_jacobian, initial_guess)

y_predicted = polynomial_model(df['t'].values, *optimized_params)

residuals = df['y'].values - y_predicted
lse = np.sum(residuals**2)

num_data_points = len(df['y'].values)
num_params = len(optimized_params)

variance_estimate = lse / (num_data_points - num_params)

J = compute_jacobian(df['t'].values, optimized_params)
fim = np.linalg.inv(J.T @ J)

cov_matrix = fim * variance_estimate

standard_errors = np.sqrt(np.diag(cov_matrix))

alpha = 0.05
t_critical_value = t.ppf(1 - alpha / 2, df=num_data_points - num_params)
confidence_intervals = [(optimized_params[i] - t_critical_value * standard_errors[i], 
                         optimized_params[i] + t_critical_value * standard_errors[i]) 
                        for i in range(len(optimized_params))]

print(f"Estimated Parameters:")
for i, param in enumerate(optimized_params):
    print(f"β{i} = {param:.4f}")

print(f"Least Squares Error (LSE): {lse:.4f}")
print(f"Estimated Variance (σ²): {variance_estimate:.4f}")

print("\nConfidence Intervals (95%):")
for i, (lower, upper) in enumerate(confidence_intervals):
    print(f"β{i} = ({lower:.4f}, {upper:.4f})")

plt.figure(figsize=(10, 6))
plt.scatter(df['t'], residuals, color='green', label='Residuals')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('t')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(df['t'], df['y'], label='Observed Data', color='blue')
plt.plot(df['t'], y_predicted, label='Fitted Model', color='red')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Observed Data and Fitted Polynomial Model with Confidence Intervals')
plt.legend()
plt.show()
