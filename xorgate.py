import tensorflow as tf
import numpy as np
import sys
#XOR requires a multilayer neural network

#The inputs are as follows:

T, F = 1.0, -1.0
bias = 1.0
bias_2 = tf.ones([4,1])

train_in = [
    [T, T, bias],
    [T, F, bias],
    [F, T, bias],
    [F, F, bias]
]

#The expected outputs are

train_out = [
    [F],
    [T],
    [T],
    [F]
]

#Since this is a multilayer neural network, we deal with intermediate outputs as:

train_out_int_1 = [
    [F],
    [F],
    [T],
    [F]
]

train_out_int_2 = [
    [F],
    [T],
    [F],
    [F]
]

#We define the step function as follows:

def step(x):
    greater = tf.greater(x, 0)
    as_float = tf.to_float(greater)
    doubled = tf.multiply(as_float, 2)
    return tf.subtract(doubled, 1)

#Now we look at the weights. The first layer of the neural network calculates the value for the following perceptrons (Neural network units):
#  p1 = ~A.B, p2 = A.~B
# Similar to single layer neural networks, we shall assume a bias and two inputs per perceptron

wp1 = tf.Variable(tf.random_normal([3,1]), dtype = tf.float32)
wp2 = tf.Variable(tf.random_normal([3,1]), dtype = tf.float32)

#The next layer of our neural network will include two inputs [we shall skip the bias component, but you can include it if you are inclined to do so]

wp3 = tf.Variable(tf.random_normal([3,1]), dtype = tf.float32)

#Moving on to the output phases, we calculate the intermediate outputs
#The intermediate outputs are stored in out_int_1 and out_int_2, and correspond to train_out_int_1 and train_out_int_2 respectively

out_int_1 = step(tf.matmul(train_in, wp1))
out_int_2 = step(tf.matmul(train_in, wp2))

#This combination will not hold for other aspects, such as errors and mses, because the sole purpose of combining them is to calculate the final output layer
out_int_combined = tf.squeeze(tf.stack([out_int_1, out_int_2, bias_2]))

out = step(tf.matmul(out_int_combined, wp3, transpose_a = True))

#Calculating errors

error_int_1 = tf.subtract(train_out_int_1, out_int_1)
error_int_2 = tf.subtract(train_out_int_2, out_int_2)
error_out = tf.subtract(train_out, out)

#Reducing the errors

mse_int_1 = tf.reduce_mean(tf.square(error_int_1))
mse_int_2 = tf.reduce_mean(tf.square(error_int_2))
mse_out = tf.reduce_mean(tf.square(error_out))

#Calculating delta to recalculate weights

delta_int_1 = tf.matmul(train_in, error_int_1, transpose_a=True)
delta_int_2 = tf.matmul(train_in, error_int_2, transpose_a=True)
delta_out = tf.matmul(out_int_combined, error_out)

#Assigning the recalculated weights

train_int_1 = tf.assign(wp1, tf.add(wp1, delta_int_1))
train_int_2 = tf.assign(wp2, tf.add(wp2, delta_int_2))
train_out = tf.assign(wp3, tf.add(wp3, delta_out))

#Initializing session and variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

err_sum, target = 1,0

epoch, MAX = 0, 100

while err_sum > target and epoch < MAX:
    epoch += 1
    err_int_1, _ = sess.run([mse_int_1, train_int_1])
    err_int_2, _ = sess.run([mse_int_2, train_int_2])
    err_out, _ = sess.run([mse_out, train_out])
    print ("Epoch:", epoch, "MSE at perceptron 1:", err_int_1, ", MSE at perceptron 2:", err_int_2, ", MSE at output", err_out)
    err_sum = err_int_1 + err_int_2 + err_out

print(sess.run([wp1, wp2, wp3]))
