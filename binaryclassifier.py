import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
rnd = np.random

def basic_class_actual(x):
    #assume output is 1 if x > 5 and 0 otherwise.
    return 1 if x > 5 else 0

def basic_classifier():
    num_epochs=1000
    alpha=0.1

    X_t = [rnd.random_sample()*10 for i in range(100)]
    X_in = np.asarray([[1, x] for x in X_t])
    Y_actual = np.asarray([[basic_class_actual(x)] for x in X_t])
    print (Y_actual.shape)

    m = len(X_in)
    n = len(X_in[0])

    X = tf.placeholder(tf.float32, shape=[m, n], name="Inputs")
    Y = tf.placeholder(tf.float32, shape=[m, 1], name="Outputs")
    w = tf.Variable(tf.zeros([n,1]), name='Weights')

    pred = tf.matmul(X, w) #This is the value that will be sent to the sigmoid function
    cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(cost)


    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print ("Initialized variables")
        for epoch in range(1, num_epochs+1):
            sess.run(optimizer, feed_dict = {X : X_in, Y : Y_actual})
            if epoch % 50 == 0:
                print ("Finished %s epochs" %(epoch))
                print ("Cost: " , sess.run(cost, feed_dict = {X : X_in, Y : Y_actual}))
                print ("Weights: " , sess.run(w))
                print ("-------------")
        print ("Over")
        print ("Cost: " , sess.run(cost, feed_dict = {X : X_in, Y : Y_actual}))
        print ("Weights: " , sess.run(w))
        print ("-------------")
        print ("Testing....")
        weights = sess.run(w)
        accurate = inaccurate = 0
        ones = []
        zeros = []
        i = 0
        for (x, y) in zip(X_in, Y_actual):
            i+=1
            obtained = 1/(1+np.exp(-np.matmul(x, weights)))
            actual = y
            if abs(actual - obtained) < 0.5:
                accurate += 1
            else:
                inaccurate += 1
            if 1 - obtained < 0.5:
                ones.append([i, x[1]])
            else:
                zeros.append([i, x[1]])
        print ("Accurate predictions: ", accurate)
        print ("Inaccurate predictions: " , inaccurate)
        plt.scatter([o[0] for o in ones], [o[1] for o in ones], c='r', marker='x')
        plt.scatter([o[0] for o in zeros], [o[1] for o in zeros], c='b', marker='x')
        plt.plot([a for a in range(i)], [5 for a in range(i)], 'g')
        plt.show()

basic_classifier()
