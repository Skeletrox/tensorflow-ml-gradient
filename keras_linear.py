from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.callbacks import TensorBoard
import numpy as np


def generate_data(num_vals):
    x1 = np.asarray([rnd.random()*d_range for i in range(num_vals)])
    x2 = np.asarray([rnd.random()*d_range for i in range(num_vals)])
    x3 = np.asarray([rnd.random()*d_range for i in range(num_vals)])
    x4 = np.asarray([rnd.random()*d_range for i in range(num_vals)])
    bias = np.asarray([1 for i in range(num_vals)])
    y = [3*x_1 + 2*x_2 + 7*x_3 + 8*x_4 + 10*b
         for (x_1, x_2, x_3, x_4, b) in zip(x1, x2, x3, x4, bias)]
    x1 = x1.reshape((num_vals, 1))
    x2 = x2.reshape((num_vals, 1))
    x3 = x3.reshape((num_vals, 1))
    x4 = x4.reshape((num_vals, 1))
    bias = bias.reshape((num_vals, 1))
    print (x1.shape)
    x = np.concatenate((x1, x2, x3, x4, bias), axis=1)
    split_point = int(num_vals*0.8)
    x_train = x[:split_point]
    y_train = y[:split_point]
    x_test = x[split_point:]
    y_test = y[split_point:]
    return ((x_train, y_train), (x_test, y_test))


rnd = np.random

# Define the limits of our dataset
d_range = 100
num_vals = 100
epochs = 500

# Multi class classifier
# Let's assume y = 3x1 + 2x2 + 7x3 + 8x4 + 10
(train_x, train_y), (test_x, test_y) = generate_data(num_vals)
train_size = len(train_x)

model = Sequential()
model.add(Dense(units=1, activation='linear', input_shape=(5,)))
tb_callback = TensorBoard(log_dir='./logs/Keras_Linear')
model.compile(loss='mse', optimizer='adam')
model.fit(train_x, train_y, epochs=epochs, batch_size=1,
          callbacks=[tb_callback])
score = model.evaluate(test_x, test_y, batch_size=1)
print (score)
values = model.predict(test_x, batch_size=1)
for (obtained, actual) in zip(values, test_y):
    print (obtained, actual)
