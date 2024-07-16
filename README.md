# Optimizer Comparison for Linear Regression

## Introduction

This project provides a comprehensive comparison of various optimization algorithms for linear regression. It aims to demonstrate the performance differences between popular optimization techniques in machine learning, including:

- Stochastic Gradient Descent (SGD)
- Batch Gradient Descent
- Mini-batch Gradient Descent
- Momentum
- Adam (Adaptive Moment Estimation)
- RMSprop (Root Mean Square Propagation)

By implementing these optimizers from scratch and comparing their performance on a simple linear regression task, this project serves as an educational tool for understanding the strengths and weaknesses of each method.

## Project Structure

The project consists of two main Python files and an additional folder:

1. `linear_regression.py`: Contains the `LinearRegression` class implementation with various optimizer options.
2. `experiments.py`: Runs experiments to compare the performance of different optimizers.
3. `understanding GD`: this folder contains python notebooks that explain the gradient descent method applied to the general problem of function minimization.

## Features

- Custom implementation of six popular optimization algorithms
- Flexible `LinearRegression` class that allows easy switching between optimizers
- Comprehensive experiment setup to compare optimizer performance
- Visualizations of cost history, final costs, convergence speed, and parameter space trajectories

## Getting Started

### Prerequisites

- Python 3.x
- NumPy
- Matplotlib

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/alessandrogentili001/optimizers.git
   cd optimizers
   ```

2. Install the required packages:
   ```
   pip install numpy matplotlib
   ```

### Running the Experiments

To run the experiments and see the comparison results:

```
python experiments.py
```

This will generate plots comparing the performance of different optimizers and print the final results.

## Customization

You can easily modify the `experiments.py` file to:
- Change the dataset size or noise level
- Adjust the learning rates or number of iterations
- Add new optimizers or modify existing ones

## Contributing

Contributions to improve the project are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- This project was inspired by the need for a clear, hands-on comparison of optimization algorithms in machine learning.
- Special thanks to all the researchers and practitioners who have developed and refined these optimization techniques.