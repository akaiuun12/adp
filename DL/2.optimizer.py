# %% 2. Optimizer.py
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')


# %% Ordinary Least Squares

def ordinary_least_squares(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = 0
    denominator = 0

    for i in range(len(x)):
        numerator += (x[i] - x_mean) * (y[i] - y_mean)
        denominator += (x[i] - x_mean) ** 2

    m = numerator / denominator
    c = y_mean - m * x_mean

    return m, c


x = np.array([2, 4, 6, 8])
y = np.array([81, 93, 91, 97])

m, c = ordinary_least_squares(x, y)
y_pred = m * x + c

# SSE
sse = sum((y - y_pred) ** 2)

plt.scatter(x, y, color='red')
plt.plot(x, y_pred, color='blue')

# Draw vertical lines for actual vs predicted
plt.vlines(x, y_pred, y, colors='gray', linestyles='dashed')

# Draw squares representing the error
for xi, yi, ypi in zip(x, y, y_pred):
    plt.plot([xi, xi, xi+1, xi+1, xi], [yi, ypi, ypi, yi, yi], 'k-')

plt.title('Ordinary Least Squares with SSE')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('lecture/images/ols_sse.png')
plt.show()

# %% Gradient Descent

def gradient_descent(x, y, learning_rate, epochs):
    m = 0
    c = 0
    n = len(x)

    for _ in range(epochs):
        y_pred = m * x + c
        dm = (-2 / n) * sum(x * (y - y_pred))
        dc = (-2 / n) * sum(y - y_pred)
        m -= learning_rate * dm
        c -= learning_rate * dc

    return m, c


x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 4, 5, 6])

m, c = gradient_descent(x, y, 0.01, 1000)
y_pred = m * x + c

plt.scatter(x, y, color='red')
plt.plot(x, y_pred, color='blue')
plt.title('Gradient Descent')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('lecture/images/gd.png')
plt.show()


# %% Stochastic Gradient Descent

def stochastic_gradient_descent(x, y, learning_rate, epochs):
    m = 0
    c = 0
    n = len(x)

    for _ in range(epochs):
        for i in range(n):
            y_pred = m * x[i] + c
            dm = -2 * x[i] * (y[i] - y_pred)
            dc = -2 * (y[i] - y_pred)
            m -= learning_rate * dm
            c -= learning_rate * dc

    return m, c


x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 4, 5, 6])

m, c = stochastic_gradient_descent(x, y, 0.01, 1000)
y_pred = m * x + c

plt.scatter(x, y, color='red')
plt.plot(x, y_pred, color='blue')
plt.title('Stochastic Gradient Descent')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('lecture/images/sgd.png')
plt.show()




# %% 
import tensorflow as tf

print(tf.__version__)