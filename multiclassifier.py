import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
rnd = np.random

#Let Y = 2 if X > 200, 1 if X > 100, 0 otherwise
def lower_predict(x):
    return 1 if x > 100 else 0

def upper_predict(x):
    return 1 if x > 200 else 0

def pseudo_multiclass_classifier():
    num_epochs = 1000
    alpha = 0.3

    X_t = [rnd.rand()*300 for i in range(100)]
    X_in = [[1, x] for x in X_t]
    Y_lower = [[lower_predict(x)] for x in X_t]
    Y_upper = [[upper_predict(x)] for x in X_t]
    m = len(X_in)
    n = len(X_in[0])

    X = tf.placeholder(tf.float32, shape=[m,n], name='Inputs')
    Y_L = tf.placeholder(tf.float32, shape=[m,1], name='Outputs_Lower')
    Y_U = tf.placeholder(tf.float32, shape=[m,1], name='Outputs_Upper')
    w_lower = tf.Variable(tf.zeros([n,1]), name='Weights_Lower')
    w_upper = tf.Variable(tf.zeros([n,1]), name='Weights_Upper')

    pred_lower = tf.matmul(X, w_lower)
    pred_upper = tf.matmul(X, w_upper)

    lower_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_lower, labels=Y_L))
    upper_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_upper, labels=Y_U))

    low_optimizer = tf.train.AdamOptimizer(alpha).minimize(lower_cost)
    up_optimizer = tf.train.AdamOptimizer(alpha).minimize(upper_cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print("Initialized Variables")

        for epoch in range(num_epochs):
            sess.run(low_optimizer, feed_dict = {X : X_in, Y_L : Y_lower})
            sess.run(up_optimizer, feed_dict = {X : X_in, Y_U : Y_upper})

            if epoch % 100 == 0:
                print ("Finished %s epochs" %(epoch))
                print ("Cost at lower classifier: " , sess.run(lower_cost, feed_dict = {X : X_in, Y_L : Y_lower}))
                print ("Cost at upper classifier: " , sess.run(upper_cost, feed_dict = {X : X_in, Y_U : Y_upper}))
                print ("======================================")

        print ("Over")
        print ("---------------------------------")
        print ("Cost at lower classifier: " , sess.run(lower_cost, feed_dict = {X : X_in, Y_L : Y_lower}))
        print ("Cost at upper classifier: " , sess.run(upper_cost, feed_dict = {X : X_in, Y_U : Y_upper}))
        print ("Weights at lower: " , sess.run(w_lower))
        print ("Weights at upper: " , sess.run(w_upper))
        print ("Testing....")
        weights_lower = sess.run(w_lower)
        weights_upper = sess.run(w_upper)
        accurate = inaccurate = 0
        ones = []
        zeros = []
        twos = []
        i = 0
        for (x, y_l, y_u) in zip(X_in, Y_lower, Y_upper):
            i+=1
            obtained_lower = 1/(1+np.exp(-np.matmul(x, weights_lower)))
            obtained_upper = 1/(1+np.exp(-np.matmul(x, weights_upper)))
            actual_lower = y_l
            actual_upper = y_u
            if abs(actual_upper - obtained_upper) < 0.5 and abs(actual_lower - obtained_lower) < 0.5:
                accurate += 1
            else:
                inaccurate += 1
            if x[1] < 100:
                zeros.append([x[1], obtained_lower + obtained_upper])
            elif x[1] < 200:
                ones.append([x[1], obtained_lower + obtained_upper])
            else:
                twos.append([x[1], obtained_lower + obtained_upper])

        print ("Accurate predictions: ", accurate)
        print ("Inaccurate predictions: " , inaccurate)

        plt.scatter([o[0] for o in zeros], [o[1] for o in zeros], c='#3949AB', marker='x', label='Classified as 0')
        plt.scatter([o[0] for o in ones], [o[1] for o in ones], c='#5E35B1', marker='x', label='Classified as 1')
        plt.scatter([o[0] for o in twos], [o[1] for o in twos], c='#00b0ff', marker='x', label='Classified as 2')
        plt.plot([a for a in range(300)], [0.5 for a in range(300)], '#3f51b5', label='Lower Limit for Class 1')
        plt.plot([a for a in range(300)], [1.5 for a in range(300)], '#673ab7', label='Lower Limit for Class 2')
        plt.legend()
        plt.grid(True)
        plt.show()


pseudo_multiclass_classifier()
