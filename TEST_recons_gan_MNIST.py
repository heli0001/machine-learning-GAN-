import os
import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import mnist
from tqdm import tqdm
import recon_gan_model as rgm

(X_all, Y_all), (x_test, y_test) = mnist.load_data(path='mnist.npz')
train_filter = np.where(Y_all != 0)
x_train, y_train = X_all[train_filter], np.array(Y_all[train_filter])
print(x_train.shape)

# Should this be tf.intofsomekind rather than tr.float32
X = tf.placeholder(tf.float32, [None,28,28,1])
inputs = tf.placeholder(tf.float32, [None,28,28,1])


R_sample, noised_sample = rgm.reconstructor(inputs)
print("reconstructor shape", R_sample.shape)
r_rep, r_logits = rgm.discriminator(X)
f_rep, f_logits = rgm.discriminator(R_sample, reuse=True)


# Defining the loss functions of the network
smooth_factor =0.1
disc_loss_real = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits, labels=tf.ones_like(r_logits)*(1-smooth_factor)))
disc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.zeros_like(f_logits)))
disc_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits, labels=tf.ones_like(r_logits)*(1-smooth_factor)))\
            + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.zeros_like(f_logits)))
recon_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.ones_like(f_logits)))

# Defining the optimizer
recon_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "GAN/Reconstructor")
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "GAN/Discriminator")
recon_step = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(recon_loss, var_list= recon_vars)
disc_step = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(disc_loss, var_list= disc_vars)

disc_rrate = tf.reduce_mean(r_logits)
disc_frate = tf.reduce_mean(f_logits)

# sess = tf.Session(config=config)

# sess = tf.Session()
# tf.global_variables_initializer().run(session=sess)

epoch = 2
batch_size = 256
dlosses = []
relosses = []
rrates = []
frates = []
samples = []


saver = tf.train.Saver() # save both variables
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        num_batches = x_train.shape[0]//batch_size//35
        print("Epoch: %d" % i)
        for j in tqdm(range(num_batches)):
            ind = np.random.randint(low=0, high=x_train.shape[0], size=batch_size)
            image_batch = x_train[ind]
            X_batch = image_batch.reshape([batch_size, 28, 28,1])
            _, dloss, rrate, frate = sess.run([disc_step, disc_loss, disc_rrate, disc_frate], feed_dict={X: X_batch, inputs: X_batch})
            _, reloss = sess.run([recon_step, recon_loss], feed_dict={inputs: X_batch})
            print("Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f" % (j, dloss, reloss))
            dlosses.append(dloss)
            relosses.append(reloss)
            rrates.append(rrate)
            frates.append(frate)
        saver.save(sess, "training Recon GAN/reGAN.ckpt")
    saver.save(sess, "training Recon GAN/final_model.ckpt")
