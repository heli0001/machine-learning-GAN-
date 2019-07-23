import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

def true_data( samplesize = 100):
    x = np.random.randn(samplesize)
    y = np.random.randn(samplesize)
    z = 2*x-y
    return np.vstack([x,y,z]).T

def noise_maker( samplesize = 100):
    x = np.random.randn(samplesize)
    y = np.random.randn(samplesize)
    z = np.random.randn(samplesize)
    return np.vstack([x,y,z]).T


def generator(Z):
    with tf.variable_scope("GAN/Generator", reuse=False):
        #h0 = tf.keras.layers.Input(Z)
        h1 = tf.keras.layers.Dense(3, activation='relu')(Z)
        h2 = tf.keras.layers.Dense(3, activation='relu')(h1)
    return h2

def discriminator(X, reuse):
    with tf.variable_scope("GAN/Discriminator", reuse = reuse):
        h1 = tf.keras.layers.Dense(3, activation='relu')(X)
        # h2 = tf.keras.layers.Dense(1, activation='sigmoid')(h1)
        h2 = tf.keras.layers.Dense(2, activation='relu')(h1)
        out = tf.nn.softmax(h2)
    #return h2
    return out

X = tf.placeholder(tf.float32, [None,3])
r_logits = discriminator(X, False)

Z = tf.placeholder(tf.float32, [None,3])
G_sample = generator(Z)
f_logits = discriminator(G_sample, True)

disc_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits, labels=tf.ones_like(r_logits))
                            + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.zeros_like(f_logits)))

gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.ones_like(f_logits)))

#Defining the optimizer
gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "GAN/Generator")
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "GAN/Discriminator")
gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss, var_list= gen_vars)
disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss, var_list= disc_vars)

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

dlosses = []
glosses = []
for foo in range(1000):
    Z_batch = noise_maker()
    _, dloss = sess.run([disc_step, disc_loss], feed_dict={X: true_data(), Z: Z_batch})
    _, gloss, gsamp = sess.run([gen_step, gen_loss, G_sample], feed_dict={Z: Z_batch})
    print('Disc: ',dloss, 'Gen: ', gloss)
    dlosses.append(dloss)
    glosses.append(gloss)

plt.plot(dlosses)
plt.plot(glosses)
plt.show()

# print(gsamp)