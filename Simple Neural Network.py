import numpy as np
import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# load the dataset
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# print(X_train.shape)

# preprocess the data
#plt.figure()
#plt.imshow(mnist.train, cmap='gray')
#plt.show()

# setting up some learning parameters
learning_rate = 0.1
epochs = 20
batch_size = 128

# with tf.variable_scope()??
# placehold the  and y
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# weight and bias into hidden layer
w1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='w1')
b1 = tf.Variable(tf.random_normal([300], stddev=0.03), name='b1')
#  w1 = tf.Variable(np.random.normal(0, 0.03, [784, 300]), name='w1')
# b1 = tf.Variable(np.random.randn(300), name='b1')
# to the output layer

w2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='w2')
b2 = tf.Variable(tf.random_normal([10]), name='b2')
# w2 = tf.Variable(np.random.normal(0, 0.03, [300, 10]), name='w2')
# b2 = tf.Variable(np.random.randn(10), name='b2')

# calculate the hidden output
hidden_out = tf.add(tf.matmul(x, w1), b1)
hidden_out = tf.nn.relu(hidden_out)
# calculate the final output
y_hid = tf.nn.softmax(tf.add(tf.matmul(hidden_out, w2), b2))

# cost function
y_good = tf.clip_by_value(y_hid, 1e-20, 0.99999) # limited y_hid between 1e-20 to 0.999999. avoid log(0)
# sum over the columns (second) m*10 for y_good and y, index start from 0,
cost = -tf.reduce_mean(tf.reduce_sum(y*tf.log(y_good) + (1-y)*tf.log(1-y_good), axis=1))

# find the optimiser
opm = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# set up initialisation operator
init = tf.global_variables_initializer()
correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_good, 1))
accurate = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# start the session
accs = []
with tf.Session () as sess:
    sess.run(init)
    total_batch = int(len(mnist.train.images) / batch_size)
    for epoch  in range (epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size = batch_size)
            _, c, accu = sess.run ([opm, cost, accurate], feed_dict={x:batch_x, y: batch_y})
            avg_cost +=c/total_batch
            # this = model.predict()
            accs.append(accu)
        print ("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
    print(sess.run(accurate, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
    print (w1.eval(), b1.eval(), w2.eval(), b2.eval())
plt.plot(accs)
plt.show()