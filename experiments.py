import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression, generate_data

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data
num_samples = 1000
X, y = generate_data(num_samples, a=2, b=1, noise_level=0.5)

# Define optimizers and their parameters
optimizers = ['sgd', 'batch', 'mini_batch', 'momentum', 'adam', 'rmsprop']
learning_rates = [0.1, 0.01, 0.001]
n_iterations = 500

# Run experiments
results = {}
for opt in optimizers:
    for lr in learning_rates:
        model = LinearRegression(learning_rate=lr, n_iterations=n_iterations, optimizer=opt)
        model.fit(X, y)
        key = f"{opt}_lr{lr}"
        results[key] = {
            'cost_history': model.cost_history,
            'final_cost': model.cost_history[-1],
            'final_params': model.theta.flatten()
        }

# Plotting
plt.figure(figsize=(20, 15))

# Plot 1: Cost History for all optimizers (best learning rate)
plt.subplot(2, 2, 1)
for opt in optimizers:
    best_lr = min([lr for lr in learning_rates], key=lambda lr: results[f"{opt}_lr{lr}"]['final_cost'])
    plt.plot(results[f"{opt}_lr{best_lr}"]['cost_history'], label=f"{opt} (lr={best_lr})")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost History (Best learning rate for each optimizer)")
plt.legend()
plt.yscale('log')

# Plot 2: Final Cost Comparison
plt.subplot(2, 2, 2)
x = np.arange(len(optimizers))
width = 0.25
for i, lr in enumerate(learning_rates):
    costs = [results[f"{opt}_lr{lr}"]['final_cost'] for opt in optimizers]
    plt.bar(x + i*width, costs, width, label=f'lr={lr}')
plt.xlabel("Optimizers")
plt.ylabel("Final Cost")
plt.title("Final Cost Comparison")
plt.xticks(x + width, optimizers)
plt.legend()
plt.yscale('log')

# Plot 3: Convergence Speed Comparison
plt.subplot(2, 2, 3)
threshold = 1e-2  # Define convergence threshold
for opt in optimizers:
    best_lr = min([lr for lr in learning_rates], key=lambda lr: results[f"{opt}_lr{lr}"]['final_cost'])
    cost_history = results[f"{opt}_lr{best_lr}"]['cost_history']
    convergence_iter = next((i for i, cost in enumerate(cost_history) if cost < threshold), n_iterations)
    plt.bar(opt, convergence_iter)
plt.xlabel("Optimizers")
plt.ylabel("Iterations to Converge")
plt.title(f"Convergence Speed (Threshold: {threshold})")

# Plot 4: Parameter Space Trajectories
plt.subplot(2, 2, 4)
for opt in optimizers:
    best_lr = min([lr for lr in learning_rates], key=lambda lr: results[f"{opt}_lr{lr}"]['final_cost'])
    model = LinearRegression(learning_rate=best_lr, n_iterations=n_iterations, optimizer=opt)
    model.fit(X, y)
    theta_history = np.array(model.theta_history)
    plt.plot(theta_history[:, 0], theta_history[:, 1], label=opt)
plt.xlabel("theta_0")
plt.ylabel("theta_1")
plt.title("Parameter Space Trajectories")
plt.legend()

plt.tight_layout()
plt.show()

print("Final Results:")
for opt in optimizers:
    best_lr = min([lr for lr in learning_rates], key=lambda lr: results[f"{opt}_lr{lr}"]['final_cost'])
    result_key = f"{opt}_lr{best_lr}"
    print(f"{opt} (lr={best_lr}):")
    print(f"  Final Cost: {results[result_key]['final_cost']:.6f}")
    print(f"  Final Parameters: {results[result_key]['final_params']}")
    print()

