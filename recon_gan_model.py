import tensorflow as tf
std = 0.1


def reconstructor(inputs, reuse=False):
    with tf.variable_scope('GAN/Reconstructor', reuse= reuse):
        with tf.variable_scope('GAN/encoder', reuse= reuse):
            # xc = tf.random_normal(shape=tf.shape(inputs), mean=0, stddev=std, dtype=tf.float32, seed=seed) + inputs
            xc = tf.random_normal(shape = tf.shape(inputs), mean= 0, stddev= std, dtype= tf.float32) + inputs
            xc = tf.reshape(xc, (-1, 28, 28, 1))
            # layer 1: 28,28,1 ---> 14,14,64
            conv1 = tf.layers.conv2d(xc, 64, 5, 2, 'same', use_bias=False)
            bn1 = tf.layers.batch_normalization(conv1)
            conv1_up = tf.nn.leaky_relu(bn1, alpha=0.1)
           # print("encoder layer1 shape", conv1.shape)
            # layer 2: 14,14,64 --->7,7,128
            conv2 = tf.layers.conv2d(conv1_up, 128, 5, 2, 'same', use_bias=False)
            bn2 = tf.layers.batch_normalization(conv2)
            conv2_up = tf.nn.leaky_relu(bn2, alpha=0.1)

            # layer 3: 7,7,128 --->4,4,256
            conv3 = tf.layers.conv2d(conv2_up, 256, 5, 2, 'same', use_bias=False)
            bn3 = tf.layers.batch_normalization(conv3)
            conv3_up = tf.nn.leaky_relu(bn3, alpha=0.1)

        with tf.variable_scope('GAN/decoder', reuse= reuse):
            # layer 1: 4,4,256 --->7,7,128
            dconv1 = tf.layers.conv2d_transpose(conv3_up, filters=128, kernel_size=4, strides=1, padding='valid')
            dbn1 = tf.layers.batch_normalization(dconv1, renorm_momentum=0.99)
            dconv1_up = tf.nn.relu(dbn1)

            # layer 2: 7,7,128 --->14,14,64
            dconv2 = tf.layers.conv2d_transpose(dconv1_up, filters=64, kernel_size=4, strides=2, padding='same')
            dbn2 = tf.layers.batch_normalization(dconv2, renorm_momentum=0.99)
            dconv2_up = tf.nn.relu(dbn2)

            # layer 3: 14,14,64 ---> 28,28,1
            out1 = tf.layers.conv2d_transpose(dconv2_up, filters=1, kernel_size=5, strides=2, padding='same')
            out = tf.tanh(out1)
    return out, xc


def discriminator(X, reuse=False):
    with tf.variable_scope("GAN/Discriminator", reuse=reuse):
    # Input layer is 28x28xn
    # Convolutional layer, 14x14x64
        X = tf.reshape(X,(-1,28,28,1))
        conv1 = tf.layers.conv2d(X, filters=64, kernel_size=5, strides=2, padding='same')
        a1 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv1)
        conv1_up = tf.layers.dropout(a1, rate=0.1)
    # Strided convolutional layer, 7x7x128
        conv2 = tf.layers.conv2d(conv1_up, 128, 5, 2, 'same', use_bias=False)
        bn2 = tf.layers.batch_normalization(conv2)
        a2 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn2)
        conv2_up = tf.layers.dropout(a2, rate=0.1)
    # Strided convolutional layer, 4x4x256
        conv3 = tf.layers.conv2d(conv2_up, 256, 5, 2, 'same', use_bias=False)
        bn3 = tf.layers.batch_normalization(conv3)
        a3 = tf.keras.layers.LeakyReLU(alpha=0.2)(bn3)
        conv3_up = tf.layers.dropout(a3, rate=0.1)

    # fully connected
        flat = tf.reshape(conv3_up, (-1, 4 * 4 * 256))
        logits = tf.layers.dense(flat, 1)
        out = tf.sigmoid(logits)
    return out, logits