import tensorflow as tf

#Define True, False and Bias
T, F = 1.0, -1.0
bias = 1.0

#Training Data
train_in = [
    [T, T, bias],
    [T, F, bias],
    [F, T, bias],
    [F, F, bias],
]

train_out = [
    [T],
    [F],
    [F],
    [F],
]

#Weights are of the shape 3, 1 because 3 inputs give a single output
w = tf.Variable(tf.random_normal([3,1]), dtype=tf.float32)

'''
    step(x) is essentially -1 if x < 0 and 1 if x >0
    since it is betweeen 0 and 1, we double it and subtract 1,
    giving us -1 [for 0] and 1 [for 1]
'''
def step(x):
    greater = tf.greater(x, 0)
    as_float = tf.to_float(greater)
    doubled = tf.multiply(as_float, 2)
    return tf.subtract(doubled, 1)

#Output is defined by A and B = w1A + w2B + w3
# w3 is the bias

output = step(tf.matmul(train_in, w))

#Error is the difference between the expected and the actual outputs
error = tf.subtract(train_out, output)

#We try to reduce the error
mse = tf.reduce_mean(tf.square(error))

#the delta is given by the product of the trained data and the error
#This helps recalculate the weights for a better answer
delta = tf.matmul(train_in, error, transpose_a=True)

#We assign the recalculated weights to the weights
train = tf.assign(w, tf.add(w, delta))

#Initialize session
sess = tf.Session()

#Initialize them variables
sess.run(tf.global_variables_initializer())


err, target = 1, 0

epoch, MAX = 0, 10

while err > target and epoch < MAX:
    epoch += 1
    err, _ = sess.run([mse, train])
    print('epoch:', epoch, 'mse:', err)

print (sess.run(w))
