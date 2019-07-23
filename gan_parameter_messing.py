import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
Tensorflow generative adversarial neural network.

'''


scale = 2
def secret_function(x,y):
    return np.exp( -2*(x-.7)**2 -5*(y)**2)

def train_data_gen(chunk_size = 10000, scale = scale):
    dsx = (np.random.rand(chunk_size)-.5)*scale
    dsy = (np.random.rand(chunk_size)-.5)*scale
    dsz = secret_function(dsx,dsy)
    train_examples = np.column_stack([dsx, dsy, dsz])
    return train_examples


def generator(Z, layer_sizes= [16,16], reuse=False):
    with tf.variable_scope("GAN/Generator", reuse = reuse):
        h1 = tf.layers.dense(Z, layer_sizes[0], activation = tf.nn.leaky_relu)
        h2 = tf.keras.layers.Reshape([4,4,1]).apply(h1)
        h3 = tf.layers.conv2d(h2,10, [2,2])
        h4 = tf.keras.layers.Reshape([90]).apply(h3)
        # h2 = tf.layers.dense(h1, layer_sizes[1], activation = tf.nn.leaky_relu)
        # h3 = tf.layers.dense(h2, 6, activation = tf.nn.leaky_relu)
        out = tf.layers.dense(h4,3)
    return out

def discriminator(X, layer_sizes = [16,16], reuse = False):
    with tf.variable_scope("GAN/Discriminator", reuse = reuse):
        h1 = tf.layers.dense(X, layer_sizes[0], activation = tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1, layer_sizes[1], activation = tf.nn.leaky_relu)
        # h3 = tf.layers.batch_normalization(h2)
        out = tf.layers.dense(h2,1, activation = tf.nn.sigmoid )
    return out, h2

# Should this be tf.intofsomekind rather than tr.float32
X = tf.placeholder(tf.float32, [None,3])
Z = tf.placeholder(tf.float32, [None,3])
Y = tf.placeholder(tf.float32, [None,3])

G_sample = generator(Z)
r_logits, r_rep = discriminator(X)
f_logits, g_rep = discriminator(G_sample, reuse=True)
y_logits, y_rep = discriminator(Y, reuse=True)

#Defining the loss functions of the network
disc_y_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits, labels=tf.ones_like(r_logits))
                            + tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logits, labels=tf.zeros_like(y_logits)))
disc_loss_real = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits, labels=tf.ones_like(r_logits)))
disc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.zeros_like(f_logits)))
disc_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits, labels=tf.ones_like(r_logits))
                            + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.zeros_like(f_logits)))
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.ones_like(f_logits)))

#Defining the optimizer
gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "GAN/Generator")
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "GAN/Discriminator")
gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss, var_list= gen_vars)
disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss, var_list= disc_vars)



disc_rrate = tf.reduce_mean( r_logits)
disc_frate = tf.reduce_mean( f_logits)

# sess = tf.Session(config=config)
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

batch_size = 128


from mpl_toolkits.mplot3d import Axes3D
X1 = np.linspace(-scale/2, scale/2, 20)
Y1 = np.linspace(-scale/2, scale/2, 20)
X1, Y1 = np.meshgrid(X1, Y1)
R1 = secret_function(X1,Y1)
x_plot = train_data_gen(chunk_size = batch_size)


dlosses = []
glosses = []
rrates = []
frates = []
for i in range(1001):
    X_batch = train_data_gen(chunk_size = batch_size)
    Z_batch = 2*np.random.rand(batch_size,3)-1
    dloss_y = sess.run([disc_y_loss], feed_dict={X: X_batch, Y: Z_batch})
    print (dloss_y)

for i in range(5001):
    X_batch = train_data_gen(chunk_size = batch_size)
    Z_batch = 2*np.random.rand(batch_size,3)-1
    _, dloss, rrate, frate = sess.run([disc_step, disc_loss, disc_rrate, disc_frate], feed_dict={X: X_batch, Z: Z_batch})
    _, gloss = sess.run([gen_step, gen_loss], feed_dict={Z: Z_batch})
    print("Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f" % (i, dloss, gloss))
    dlosses.append(dloss)
    glosses.append(gloss)
    rrates.append(rrate)
    frates.append(frate)

    if i%500 == 0:
        g_plot, flog, rlog = sess.run([G_sample, f_logits, r_logits], feed_dict={X:X_batch, Z: Z_batch})
        fig = plt.figure(figsize = (16,9))
        ax = fig.add_subplot(111, projection='3d')
        plt.set_cmap('winter')
        xax = ax.scatter(X_batch[:, 0], X_batch[:, 1], X_batch[:, 2], c=rlog[:,0].T)
        plt.colorbar(xax)
        plt.set_cmap('plasma')
        gax = ax.scatter(g_plot[:, 0], g_plot[:, 1], g_plot[:, 2], c=flog[:,0].T)
        plt.colorbar(gax)
        ax.plot_surface(X1, Y1, R1, alpha=.3)
        plt.legend((xax,gax), ("Real Data","Generated Data"))
        plt.title('Iteration %d'%i)
        plt.tight_layout()
        plt.savefig('pics/i%d.png'%i)
        plt.close()

plt.plot(dlosses[100:], label='D loss')
plt.plot(glosses[100:], label='G loss')
plt.legend()
plt.show()

plt.plot(rrates[100:], label='Mean real D score')
plt.plot(frates[100:], label='Mean fake D score')
plt.legend()
plt.show()