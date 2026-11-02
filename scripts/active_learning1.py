import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

rng = np.random.default_rng()
mu1, sigma1 = 0, 1
mu2, sigma2 = 1, 2
n_samples = 1000

x1, x2 = rng.normal(mu1, sigma2, n_samples), rng.normal(mu1, sigma2, n_samples)
X = np.column_stack((x1, x2))


def fun_y(x1, x2):
    return np.sin(x1) + x2


y = fun_y(x1, x2)

# Create 1D arrays for X and Y axes
x1_grid = np.linspace(-10, 10, 200)
x2_grid = np.linspace(-10, 10, 200)

# Combine them into 2D matrices
x1v, x2v = np.meshgrid(x1_grid, x2_grid)
X_val = np.column_stack((x1v.ravel(), x2v.ravel()))
y_val = fun_y(X_val[:, 0], X_val[:, 1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(f"Initial training set size: {X_train.shape[0]}")
print("-" * 40)

n_mlp = 2
test_errors = []  # List to track MSE at each iteration

for iter_num in range(10):
    print(f"Starting Iteration {iter_num + 1}...")

    preds = np.zeros((X_val.shape[0], n_mlp))
    test_preds = np.zeros((X_test.shape[0], n_mlp))  # Array for test set predictions

    for i in range(n_mlp):
        X_new, _, y_new, _ = train_test_split(X_train, y_train, test_size=0.3)

        mlp = MLPRegressor(hidden_layer_sizes=(16, 16), activation="tanh", max_iter=500)
        mlp = mlp.fit(X_new, y_new)

        # Predict on both the validation grid (for uncertainty) and test set (for evaluation)
        preds[:, i] = mlp.predict(X_val)
        test_preds[:, i] = mlp.predict(X_test)

    # Evaluate current ensemble performance on the holdout test set
    mean_test_preds = np.mean(test_preds, axis=1)
    current_mse = mean_squared_error(y_test, mean_test_preds)
    test_errors.append(current_mse)
    print(f"  > Test MSE: {current_mse:.4f}")

    # Calculate uncertainty (variance across models)
    std = np.std(preds, axis=1)

    # Active learning acquisition
    points2add = 50
    max_std_idx = np.argpartition(std, -points2add)[-points2add:]

    X_max = X_val[max_std_idx]
    y_max = fun_y(X_max[:, 0], X_max[:, 1])

    X_train = np.vstack((X_train, X_max))
    y_train = np.concatenate((y_train, y_max))

    print(f"  > Acquired {points2add} new points with highest uncertainty.")
    print("-" * 40)

# Plotting the evaluation
plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(test_errors) + 1), test_errors, marker="o", linestyle="-", color="b"
)
plt.title("Active Learning Performance Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error (MSE) on Test Set")
plt.xticks(range(1, len(test_errors) + 1))
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()
