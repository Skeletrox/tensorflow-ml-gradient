import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt


def generate_x_y_data(batch_size):
    '''
    returns (X, Y)
    X is training data, Y is test data, which follows X
    '''
    seq_length = 24

    batch_x = []
    batch_y = []

    for _ in range(batch_size):
        vals = np.sin(np.linspace(0.0*math.pi, 3.0*math.pi, seq_length*2)
                      ) + np.random.rand()
        # vals = [i + np.random.rand() for i in range(0, 2*seq_length)]
        # vals = [5, 5, 6, 5, 5, 6, 7, 6, 10, 12, 10, 14,
        #        15, 12, 6, 19, 18, 15, 5, 5, 6, 2, 1, 6]*2
        x = vals[:seq_length]
        y = vals[seq_length:]
        x_ = np.array([x])
        y_ = np.array([y])
        x_, y_ = x_.T, y_.T

        batch_x.append(x_)
        batch_y.append(y_)

    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)

    batch_x = np.array(batch_x).transpose((1, 0, 2))
    batch_y = np.array(batch_y).transpose((1, 0, 2))

    return batch_x, batch_y


x, y = generate_x_y_data(100)
print (x.shape)
print (y.shape)

# print (x)

seq_length = x.shape[0]
batch_size = 5

output_dim = input_dim = 1
hidden_dim = 10
layers_stacked = 2

alpha = 0.05
iterations = 300
lr_decay = 0.92
momentum = 0.5
lambda_l2_reg = 0.003

try:
    tf.nn.seq2seq = tf.contrib.legacy_seq2seq
    tf.nn.rnn_cell = tf.contrib.rnn
    tf.nn.rnn_cell.GRUCell = tf.contrib.rnn.GRUCell
    print("TensorFlow's version : 1.0 (or more)")
except Exception as e:
    print("TensorFlow's version : 0.12")


tf.reset_default_graph()
sess = tf.Session()

with tf.variable_scope('seq2seq'):

    enc_inp = [
        tf.placeholder(tf.float32,
                       shape=(None, input_dim), name="inp_{}".format(t))
        for t in range(seq_length)
    ]
    expected_sparse_output = [
        tf.placeholder(tf.float32,
                       shape=(None, output_dim),
                       name="expected_sparse_output_{}".format(t))
        for t in range(seq_length)
    ]
    dec_inp = [tf.zeros_like(enc_inp[0],
               dtype=np.float32, name="GO")] + enc_inp[:-1]

    cells = []
    for i in range(layers_stacked):
        with tf.variable_scope('RNN_{}'.format(i)):
            cells.append(tf.nn.rnn_cell.GRUCell(hidden_dim))
            # cells.append(tf.nn.rnn_cell.BasicLSTMCell(...))
    cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    dec_outputs, dec_memory = tf.nn.seq2seq.basic_rnn_seq2seq(
        enc_inp,
        dec_inp,
        cell
    )

    w_out = tf.Variable(tf.random_normal([hidden_dim, output_dim]))
    b_out = tf.Variable(tf.random_normal([output_dim]))

    output_scale_factor = tf.Variable(1.0, name="Output_ScaleFactor")
    reshaped_outputs = [output_scale_factor*(tf.matmul(i, w_out) + b_out)
                        for i in dec_outputs]

with tf.variable_scope('Loss'):
    # L2 loss
    output_loss = 0
    for _y, _Y in zip(reshaped_outputs, expected_sparse_output):
        output_loss += tf.reduce_mean(tf.nn.l2_loss(_y - _Y))

    # L2 regularization (to avoid overfitting and to have a better generalizat
    # ion capacity)
    reg_loss = 0
    for tf_var in tf.trainable_variables():
        if not ("Bias" in tf_var.name or "Output_" in tf_var.name):
            reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

    loss = output_loss + lambda_l2_reg * reg_loss

with tf.variable_scope('Optimizer'):
    optimizer = tf.train.RMSPropOptimizer(alpha, decay=lr_decay,
                                          momentum=momentum)
    train_op = optimizer.minimize(loss)


def train_batch(batch_size):
    """
    Training step that optimizes the weights
    provided some batch_size X and Y examples from the dataset.
    """
    X, Y = generate_x_y_data(batch_size)
    feed_dict = {enc_inp[t]: X[t] for t in range(len(enc_inp))}
    feed_dict.update({expected_sparse_output[t]:
                      Y[t] for t in range(len(expected_sparse_output))})
    _, loss_t = sess.run([train_op, loss], feed_dict)
    return loss_t


def test_batch(batch_size):
    """
    Test step, does NOT optimize. Weights are frozen by not
    doing sess.run on the train_op.
    """
    X, Y = generate_x_y_data(batch_size)
    feed_dict = {enc_inp[t]: X[t] for t in range(len(enc_inp))}
    feed_dict.update({expected_sparse_output[t]:
                      Y[t] for t in range(len(expected_sparse_output))})
    loss_t = sess.run([loss], feed_dict)
    return loss_t[0]


train_losses = []
test_losses = []

sess.run(tf.global_variables_initializer())
for t in range(iterations+1):
    train_loss = train_batch(batch_size)
    train_losses.append(train_loss)

    if t % 10 == 0:
        # Tester
        test_loss = test_batch(batch_size)
        test_losses.append(test_loss)
        print("Step {}/{}, train loss: {}, \tTEST loss: {}".format(t,
                                                                   iterations,
                                                                   train_loss,
                                                                   test_loss))

print("Fin. train loss: {}, \tTEST loss: {}".format(train_loss, test_loss))
plt.figure(figsize=(12, 6))
plt.plot(np.array(range(0,
                        len(test_losses))
                  )/float(len(test_losses)-1)*(len(train_losses)-1),
         np.log(test_losses),
         label="Test loss"
         )
plt.plot(
    np.log(train_losses),
    label="Train loss"
)
plt.title("Training errors over time (on a logarithmic scale)")
plt.xlabel('Iteration')
plt.ylabel('log(Loss)')
plt.legend(loc='best')
plt.show()

nb_predictions = 3
print("Let's visualize {} predictions with\
 our signals:".format(nb_predictions))

X, Y = generate_x_y_data(batch_size=nb_predictions)
feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
outputs = np.array(sess.run([reshaped_outputs], feed_dict)[0])

for j in range(nb_predictions):
    plt.figure(figsize=(12, 3))

    for k in range(output_dim):
        past = X[:, j, k]
        expected = Y[:, j, k]
        pred = outputs[:, j, k]

        label1 = "Seen (past) values"
        label2 = "True future values"
        label3 = "Predictions"
        plt.plot(range(len(past)), past, "o--b", label=label1)
        plt.plot(range(len(past), len(expected)+len(past)), expected, "x--b",
                 label=label2)
        plt.plot(range(len(past), len(pred)+len(past)), pred, "o--y",
                 label=label3)

    plt.legend(loc='best')
    plt.title("Predictions v.s. true values")
    plt.show()
