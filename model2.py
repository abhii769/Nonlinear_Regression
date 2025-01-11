import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = {
    't': [0.01, 0.03, 0.04, 0.05, 0.07, 0.08, 0.09, 0.11, 0.12, 0.13, 0.15, 0.16, 0.17, 0.19,
          0.20, 0.21, 0.23, 0.24, 0.25, 0.27, 0.28, 0.29, 0.31, 0.32, 0.33, 0.35, 0.36, 0.37,
          0.39, 0.40, 0.41, 0.43, 0.44, 0.45, 0.47, 0.48, 0.49, 0.51, 0.52, 0.53, 0.55, 0.56,
          0.57, 0.59, 0.60, 0.61, 0.63, 0.64, 0.65, 0.67, 0.68, 0.69, 0.71, 0.72, 0.73, 0.75,
          0.76, 0.77, 0.79, 0.80, 0.81, 0.83, 0.84, 0.85, 0.87, 0.88, 0.89, 0.91, 0.92, 0.93,
          0.95, 0.96, 0.97, 0.99, 1.00],
    'y': [8.4201, 8.5624, 8.6420, 8.7681, 8.8110, 8.9750, 9.0337, 9.1791, 9.3122, 9.3899, 9.5486, 9.6763, 
          9.7820, 9.9307, 10.0614, 10.2165, 10.3552, 10.4393, 10.5861, 10.7517, 10.8842, 11.0062, 11.1657, 
          11.3202, 11.5593, 11.6538, 11.8317, 11.9710, 12.1655, 12.3367, 12.5526, 12.6882, 12.9224, 13.0763, 
          13.3090, 13.4396, 13.6745, 13.8529, 14.0600, 14.2940, 14.5535, 14.7239, 14.9380, 15.1779, 15.4129, 
          15.6788, 15.9082, 16.1593, 16.4322, 16.6636, 16.9461, 17.2595, 17.4573, 17.7848, 18.0744, 18.3205, 
          18.6276, 19.0191, 19.3157, 19.6301, 19.9632, 20.2039, 20.5709, 20.9546, 21.2796, 21.6742, 21.9976, 
          22.3817, 22.8038, 23.1386, 23.5531, 23.9421, 24.4370, 24.8160, 25.2312]
}

df = pd.DataFrame(data)

t = (df['t'] - df['t'].mean()) / df['t'].std()
y = (df['y'] - df['y'].mean()) / df['y'].std()


def rat_model(t, p):
    a0, a1, b0, b1 = p
    return (a0 + a1 * t) / (b0 + b1 * t)

def jac_rat(t, p):
    a0, a1, b0, b1 = p
    J = np.zeros((len(t), len(p)))
    denom = (b0 + b1 * t) ** 2
    J[:, 0] = 1 / (b0 + b1 * t)
    J[:, 1] = t / (b0 + b1 * t)
    J[:, 2] = -(a0 + a1 * t) / denom
    J[:, 3] = -(a0 + a1 * t) * t / denom
    return J

def gauss_newton(t, y, p_init, tol=1e-6, max_iter=100, lam=1e-1):
    p = np.array(p_init, dtype=float)
    for _ in range(max_iter):
        res = y - rat_model(t, p)
        J = jac_rat(t, p)
        dp = np.linalg.inv(J.T @ J + lam * np.eye(J.shape[1])) @ J.T @ res
        p += dp
        if np.linalg.norm(dp) < tol:
            break
    return p

def grid_search(t, y, a0_init, a1_init, b0_rng, b1_rng, tol=1e-6, max_iter=100, lam=1e-1):
    best_p, best_lse = None, float('inf')
    for b0 in b0_rng:
        for b1 in b1_rng:
            p0 = [a0_init, a1_init, b0, b1]
            p_opt = gauss_newton(t, y, p0, tol, max_iter, lam)
            res = y - rat_model(t, p_opt)
            lse = np.sum(res ** 2)
            if lse < best_lse:
                best_p, best_lse = p_opt, lse
    return best_p, best_lse


b0_rng = np.linspace(-5, 5, 20)
b1_rng = np.linspace(-2, 2, 20)


opt_p, opt_lse = grid_search(t, y, 0, 0, b0_rng, b1_rng)


n, k = len(y), 4
sigma2_est = opt_lse / (n - k)


def conf_int(J, sigma2):
    FIM = J.T @ J / sigma2 + np.eye(J.shape[1]) * 1e-6
    cov_mat = np.linalg.inv(FIM)
    intervals = [(opt_p[i] - 1.96 * np.sqrt(cov_mat[i, i]), opt_p[i] + 1.96 * np.sqrt(cov_mat[i, i])) for i in range(len(opt_p))]
    return intervals

conf_intervals = conf_int(jac_rat(t, opt_p), sigma2_est)

print("Optimal parameters:")
for i, name in enumerate(['a0', 'a1', 'b0', 'b1']):
    print(f"{name} = {opt_p[i]:.4f} (95% CI: {conf_intervals[i][0]:.4f} to {conf_intervals[i][1]:.4f})")
print(f"LSE: {opt_lse:.4f}")
print(f"Variance (σ²): {sigma2_est:.4f}")


residuals = y - rat_model(t, opt_p)
plt.figure(figsize=(10, 6))
plt.scatter(t, residuals, color='purple')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('t (scaled)')
plt.ylabel('Residuals')
plt.title('Residuals of the Fitted Rational Model')
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(t, y, label='Observed', color='blue')
plt.plot(t, rat_model(t, opt_p), label='Fitted Model', color='red')
plt.xlabel('t (scaled)')
plt.ylabel('y (scaled)')
plt.title('Observed vs. Fitted Model')
plt.legend()
plt.show()
