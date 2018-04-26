import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
rnd = np.random

learning_rate = 0.5
training_epochs = 1000
display = 50

#t_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
#t_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])
t_X = np.asarray([rnd.random()*100 for i in range(100)])
W_real = 2
b_real = 3
t_Y = np.asarray([(W_real*x + b_real + rnd.normal(scale=2.0)) for x in t_X])
n_samples = t_X.shape[0]

X = tf.placeholder('float')
Y = tf.placeholder('float')

W = tf.Variable(rnd.randn(), name="weight")
b = tf.Variable(rnd.randn(), name="bias")

pred = tf.add(tf.multiply(W, X), b)

cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        for (x, y) in zip(t_X, t_Y):
            sess.run(optimizer, feed_dict={X : t_X, Y : t_Y})
        if (epoch+1) % display == 0:
            c = sess.run(cost, feed_dict={X : t_X, Y: t_Y})
            print ("Epoch:", "%04d" %(epoch+1), "Cost:", "{:.9f}".format(c), "W=", sess.run(W), "b=", sess.run(b))
        if abs(sess.run(W)-W_real) < 0.0001 and abs(sess.run(b)-b_real) < 0.0001:
            print ("Optimal solution obtained at Epoch:", "%04d" %(epoch+1), "Cost:", "{:.9f}".format(c), "W=", sess.run(W), "b=", sess.run(b))
            break

    print ("Finished Optimization")
    train_cost = sess.run(cost, feed_dict = {X: t_X, Y: t_Y})
    print ("Training Cost:", train_cost, "W=", sess.run(W), "b=", sess.run(b), "\n")

    plt.plot(t_X, t_Y, "ro", label="Original Data");
    plt.plot(t_X, sess.run(W)*t_X + sess.run(b), label="Fitted Line")
    plt.legend()
    plt.show()
