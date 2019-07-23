import tensorflow as tf
import numpy as np

'''
Tensorflow generative adversarial neural network.

'''


scale = 2
def secret_function(x,y):
    return 1+np.exp(-4*(x-.2)**2-2*(y-.3)**2)

def train_data_gen(chunk_size = 10000, scale = scale):
    dsx = (np.random.rand(chunk_size)-.5)*scale
    dsy = (np.random.rand(chunk_size)-.5)*scale
    dsz = secret_function(dsx,dsy)
    train_examples = np.column_stack([dsx, dsy, dsz])
    return train_examples


def generator(Z, layer_sizes= [16,16], reuse=False):
    with tf.variable_scope("GAN/Generator", reuse = reuse):
        h1 = tf.layers.dense(Z, layer_sizes[0], activation = tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1, layer_sizes[1], activation = tf.nn.leaky_relu)
        out = tf.layers.dense(h2,3)
    return out

def discriminator(X, layer_sizes = [16,16], reuse = False):
    with tf.variable_scope("GAN/Discriminator", reuse = reuse):
        h1 = tf.layers.dense(X, layer_sizes[0], activation = tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1, layer_sizes[1], activation = tf.nn.leaky_relu)
        h3 = tf.layers.dense(h2,6)
        h4 = tf.layers.dense(h3,2)
        out = tf.layers.dense(h4,1)
    return out, h4

# Should this be tf.intofsomekind rather than tr.float32
X = tf.placeholder(tf.float32, [None,3])
Z = tf.placeholder(tf.float32, [None,3])

G_sample = generator(Z)
r_logits, r_rep = discriminator(X)
f_logits, g_rep = discriminator(G_sample, reuse=True)

#Defining the loss functions of the network
disc_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits, labels=tf.ones_like(r_logits))
                            + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.zeros_like(f_logits)))
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.ones_like(f_logits)))

#Defining the optimizer
gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "GAN/Generator")
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "GAN/Discriminator")
gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss, var_list= gen_vars)
disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss, var_list= disc_vars)


# sess = tf.Session(config=config)
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

batch_size = 128

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
X1 = np.linspace(-scale/2, scale/2, 20)
Y1 = np.linspace(-scale/2, scale/2, 20)
X1, Y1 = np.meshgrid(X1, Y1)
R1 = secret_function(X1,Y1)
x_plot = train_data_gen(chunk_size = batch_size)


dlosses = []
glosses = []
for i in range(1000):
    X_batch = train_data_gen(chunk_size = batch_size)
    Z_batch = 2*np.random.rand(batch_size,3)-1
    _, dloss = sess.run([disc_step, disc_loss], feed_dict={X: X_batch, Z: Z_batch})
    _, gloss = sess.run([gen_step, gen_loss], feed_dict={Z: Z_batch})
    print("Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f" % (i, dloss, gloss))
    dlosses.append(dloss)
    glosses.append(gloss)

plt.plot(dlosses)
plt.plot(glosses)
plt.show()


    # if i%500 == 0:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     g_plot = sess.run(G_sample, feed_dict={Z: Z_batch})
    #     xax = ax.scatter(x_plot[:,0], x_plot[:,1], x_plot[:,2])
    #     gax = ax.scatter(g_plot[:,0], g_plot[:,1], g_plot[:,2])
    #     ax.plot_surface(X1, Y1, R1, alpha=.3)
    #     plt.legend((xax,gax), ("Real Data","Generated Data"))
    #     plt.title('Samples at Iteration %d'%i)
    #     plt.tight_layout()
    #     plt.savefig('pics/iteration_%d.png'%i)
    #     plt.close()



