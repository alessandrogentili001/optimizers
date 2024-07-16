import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, optimizer='sgd'):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.optimizer = optimizer
        self.theta = None
        self.cost_history = []
        self.theta_history = []

    def add_bias(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X]

    def compute_cost(self, X, y):
        m = len(y)
        return np.sum((X.dot(self.theta) - y)**2) / (2*m)

    def compute_gradient(self, X, y):
        m = len(y)
        return X.T.dot(X.dot(self.theta) - y) / m

    def sgd(self, X, y):
        m = len(y)
        for _ in range(self.n_iterations):
            for i in range(m):
                random_index = np.random.randint(m)
                xi = X[random_index:random_index+1]
                yi = y[random_index:random_index+1]
                gradient = xi.T.dot(xi.dot(self.theta) - yi)
                self.theta -= self.learning_rate * gradient
            self.record_progress(X, y)

    def batch_gd(self, X, y):
        for _ in range(self.n_iterations):
            gradient = self.compute_gradient(X, y)
            self.theta -= self.learning_rate * gradient
            self.record_progress(X, y)

    def mini_batch_gd(self, X, y, batch_size=32):
        m = len(y)
        for _ in range(self.n_iterations):
            for i in range(0, m, batch_size):
                xi = X[i:i+batch_size]
                yi = y[i:i+batch_size]
                gradient = self.compute_gradient(xi, yi)
                self.theta -= self.learning_rate * gradient
            self.record_progress(X, y)

    def momentum_gd(self, X, y, momentum=0.9):
        velocity = np.zeros_like(self.theta)
        for _ in range(self.n_iterations):
            gradient = self.compute_gradient(X, y)
            velocity = momentum * velocity - self.learning_rate * gradient
            self.theta += velocity
            self.record_progress(X, y)

    def adam(self, X, y, beta1=0.9, beta2=0.999, epsilon=1e-8):
        m = np.zeros_like(self.theta)
        v = np.zeros_like(self.theta)
        for t in range(1, self.n_iterations + 1):
            gradient = self.compute_gradient(X, y)
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            self.theta -= self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            self.record_progress(X, y)

    def rmsprop(self, X, y, decay_rate=0.9, epsilon=1e-8):
        cache = np.zeros_like(self.theta)
        for _ in range(self.n_iterations):
            gradient = self.compute_gradient(X, y)
            cache = decay_rate * cache + (1 - decay_rate) * (gradient ** 2)
            self.theta -= self.learning_rate * gradient / (np.sqrt(cache) + epsilon)
            self.record_progress(X, y)

    def fit(self, X, y):
        X = self.add_bias(X)
        self.theta = np.random.randn(X.shape[1], 1)
        
        if self.optimizer == 'sgd':
            self.sgd(X, y)
        elif self.optimizer == 'batch':
            self.batch_gd(X, y)
        elif self.optimizer == 'mini_batch':
            self.mini_batch_gd(X, y)
        elif self.optimizer == 'momentum':
            self.momentum_gd(X, y)
        elif self.optimizer == 'adam':
            self.adam(X, y)
        elif self.optimizer == 'rmsprop':
            self.rmsprop(X, y)
        else:
            raise ValueError("Unknown optimizer")

    def predict(self, X):
        X = self.add_bias(X)
        return X.dot(self.theta)

    def record_progress(self, X, y):
        self.cost_history.append(self.compute_cost(X, y))
        self.theta_history.append(self.theta.copy())

    def plot_progress(self, X, y):
        plt.figure(figsize=(20, 5))
        
        # Plot 1: Data and Regression Line
        plt.subplot(131)
        plt.scatter(X, y, alpha=0.6)
        X_range = np.array([[X.min()], [X.max()]])
        X_range_bias = self.add_bias(X_range)
        y_pred = self.predict(X_range)
        plt.plot(X_range, y_pred, color='r', label='Regression Line')
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title("Linear Regression Fit")
        plt.legend()

        # Plot 2: Cost History
        plt.subplot(132)
        plt.plot(range(len(self.cost_history)), self.cost_history)
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title("Cost History")

        # Plot 3: Parameter Space
        plt.subplot(133)
        theta_history = np.array(self.theta_history)
        plt.plot(theta_history[:, 0], theta_history[:, 1], 'r-')
        plt.plot(theta_history[:, 0], theta_history[:, 1], 'bo')
        plt.xlabel("theta_0")
        plt.ylabel("theta_1")
        plt.title("Gradient Descent in Parameter Space")

        plt.tight_layout()
        plt.show()

# Generate sample data
def generate_data(num_samples, a, b, noise_level=0.1):
    X = np.random.rand(num_samples, 1)
    y = a * X + b + np.random.randn(num_samples, 1) * noise_level
    return X, y

# Example usage
# num_samples = 100
# X, y = generate_data(num_samples, a=2, b=1)

# Create and train the model using RMSprop optimizer
# model = LinearRegression(learning_rate=0.01, n_iterations=200, optimizer='rmsprop')
# model.fit(X, y)

# Plot the results
# model.plot_progress(X, y)

# print("Final parameters:", model.theta.flatten())