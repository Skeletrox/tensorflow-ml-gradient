import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
rnd = np.random

learning_rate = 0.5
training_epochs = 50
display = 50

t_X = np.asarray([rnd.random()*100 for i in range(100)])
a_real = 2
b_real = 3
c_real = 12

t_Y = np.asarray([a_real*x*x + b_real*x + c_real + rnd.normal(scale=2.0)for x in t_X])
n_samples = t_X.shape[0]

X = tf.placeholder('float')
Y = tf.placeholder('float')

a = tf.Variable(rnd.randn(), name='square_coeff')
b = tf.Variable(rnd.randn(), name='linear_coeff')
c = tf.Variable(rnd.randn(), name='constant')

pred = tf.add(tf.add(tf.multiply(a, tf.multiply(X, X)), tf.multiply(b, X)), c)

cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        for (x, y) in zip(t_X, t_Y):
            sess.run(optimizer, feed_dict={X : t_X, Y : t_Y})
        if (epoch+1) % display == 0:
            cst = sess.run(cost, feed_dict={X : t_X, Y : t_Y})
            print ("Epoch:", "%04d" %(epoch+1), "Cost:", "{:.9f}".format(cst), "a=", sess.run(a), "b=", sess.run(b), "c=", sess.run(c))

    print ("Finished Optimization")
    train_cost = sess.run(cost, feed_dict = {X : t_X, Y : t_Y})
    print ("Training Cost:", train_cost, "a=", sess.run(a), "b=", sess.run(b), "c=", sess.run(c))
    #plt.plot(t_X, t_Y,  "ro", label="Original Data")
    yplt = []
    xplt = [x for x in t_X]
    for x in xplt:
        yplt.append(sess.run(a)*(x**2) + sess.run(b)*x + sess.run(c))
    print (xplt)
    print (yplt)
    plt.plot(xplt, yplt, label="Fitted Curve")
    plt.legend()
    plt.show()
