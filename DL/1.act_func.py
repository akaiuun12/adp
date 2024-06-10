# %% 1. Activation Function
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# %% 0. Step Function
def step_function(x):
    return np.array(x > 0, dtype=np.int)

x = np.arange(-10, 10, 0.1)
plt.plot(x, step_function(x))
plt.vlines(0, -1, 2, colors='b', linestyles='dashed')
plt.hlines(0, -10, 10, colors='b', linestyles='dashed')

plt.title('Step Function')
plt.xlabel('x')
plt.ylabel('step_function(x)')
plt.savefig('images/step_function.png')
plt.show()

# %% 1. Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-10, 10, 0.1)
plt.plot(x, sigmoid(x))
plt.vlines(0, -10, 1, colors='b', linestyles='dashed')
plt.hlines(0, -10, 10, colors='b', linestyles='dashed')

plt.title('Sigmoid Function')
plt.xlabel('x')
plt.ylabel('sigmoid(x)')

plt.xlim(-10, 10)
plt.ylim(-1, 1)
plt.savefig('images/sigmoid.png')
plt.show()


# %% 2. ReLU
def relu(x):
    return np.maximum(0, x)

x = np.arange(-10, 10, 0.1)
plt.plot(x, relu(x))
plt.vlines(0, -10, 10, colors='b', linestyles='dashed')
plt.hlines(0, -10, 10, colors='b', linestyles='dashed')

plt.title('ReLU Function')
plt.xlabel('x')
plt.ylabel('relu(x)')

plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.savefig('images/relu.png')
plt.show()


# %% 3. Tanh
def tanh(x):
    return np.tanh(x)

x = np.arange(-10, 10, 0.1)
plt.plot(x, tanh(x))
plt.vlines(0, -1, 1, colors='b', linestyles='dashed')
plt.hlines(0, -10, 10, colors='b', linestyles='dashed')

plt.title('Tanh Function')
plt.xlabel('x')
plt.ylabel('tanh(x)')

plt.xlim(-10, 10)
plt.ylim(-1, 1)
plt.savefig('images/tanh.png')
plt.show()
