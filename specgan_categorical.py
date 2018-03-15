"""An official implementation of

Donahue, C., McAuley, J., & Puckette, M. (2018). Synthesizing Audio with Generative Adversarial Networks.  http://arxiv.org/abs/1802.04208

using the improved WGAN described in https://arxiv.org/abs/1704.00028

WGAN code is taken from:
https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
"""

import argparse
import os
import datetime
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import Activation, UpSampling2D, Conv2D
from keras.layers.merge import _Merge
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.callbacks import TensorBoard
from keras.utils.np_utils import to_categorical
from keras import backend as K
from functools import partial
import tensorflow as tf
from utils import audio_tools as audio

try:
    from PIL import Image
except ImportError:
    print('This script depends on pillow! Please install it (e.g. with pip install pillow)')
    exit()

BATCH_SIZE = 64
TRAINING_RATIO = 5  # The training ratio is the number of discriminator updates per generator update. The paper uses 5.
GRADIENT_PENALTY_WEIGHT = 10  # As per the paper

D = 64 # model size coef

def wasserstein_loss(y_true, y_pred):
    """ for more detail: https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py"""
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    """ for more detail: https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py"""
    gradients = K.gradients(K.sum(y_pred), averaged_samples)
    gradient_l2_norm = K.sqrt(K.sum(K.square(gradients)))
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return gradient_penalty


def make_generator():
    """Creates a generator model that takes a 100-dimensional noise vector as a "seed", and outputs images
    of size 128x128x1."""

    model = Sequential()
    model.add(Dense(256 * D, input_dim=100))
    model.add(Reshape((4, 4, 16 * D)))
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(8 * D, (5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(4 * D, (5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(2 * D, (5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(D, (5, 5), padding='same'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))

    return model


def make_discriminator(nb_categories):
    """ Discriminator to determine if it's real or fake and category of the sound.
        Note that unlike normal GANs, the real/fake output is not sigmoid and does not represent a probability
     """

    input_data = Input(shape=(128, 128, 1))
    x = Conv2D(D, (5, 5), strides=(2,2), padding='same')(input_data)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(D * 2, (5, 5), strides=(2,2), kernel_initializer='he_normal',padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(D * 4, (5, 5), strides=(2,2), kernel_initializer='he_normal',padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(D * 8, (5, 5), strides=(2,2), kernel_initializer='he_normal',padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(D * 16, (5, 5), strides=(2,2), kernel_initializer='he_normal', padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Flatten()(x)
    real_fake = Dense(1, kernel_initializer='he_normal', name='real_fake')(x) # no activation for wasserstein_loss
    categories = Dense(nb_categories, kernel_initializer='he_normal', name='categories', activation='softmax')(x)

    model = Model(input_data, [real_fake, categories])

    return model


def tile_images(image_stack):
    assert len(image_stack.shape) == 3
    image_list = [image_stack[i, :, :] for i in range(image_stack.shape[0])]
    tiled_images = np.concatenate(image_list, axis=1)
    return tiled_images


class RandomWeightedAverage(_Merge):
    """ for more detail: https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py"""

    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


def generate_images(generator_model, output_dir, epoch):
    """Feeds random seeds into the generator and tiles and saves the output to a PNG file."""
    test_image_stack = generator_model.predict(np.random.rand(10, 100))

    # generate and save sample audio file for each epoch
    for i in range(4):
        w = test_image_stack[i]
        outfile = os.path.join(output_dir, "train_epoch_%02d(%02d).wav" % (epoch, i))
        save_audio(w,outfile)

    test_image_stack = (test_image_stack * 127.5) + 127.5
    test_image_stack = np.squeeze(np.round(test_image_stack).astype(np.uint8))
    tiled_output = tile_images(test_image_stack)
    tiled_output = Image.fromarray(tiled_output, mode='L')  # L specifies greyscale
    outfile = os.path.join(output_dir, 'epoch_{}.png'.format(epoch))
    tiled_output.save(outfile)

def save_audio(y, path):
    """ generate a wav file from a given spectrogram and save it """
    s = np.squeeze(y)
    s = denormalize(s)
    w = audio.inv_melspectrogram(s)
    audio.save_wav(w, path)

def denormalize(norm_s):
    """ normalized spectrogram to original spectrogram using the calculated mean/standard deviation """
    assert norm_s.shape[0] == mel_means.shape[0]
    Y = (norm_s * (3.0 * mel_stds)) + mel_means
    return Y

def swap_input(model, input_layer):
    """ swap model input/output.  TODO: Are there any smarter way? """
    x = input_layer
    for layer in model.layers[1:-2]:
        x = layer(x)
    real_fake = model.layers[-2](x)
    categories = model.layers[-1](x)
    return [real_fake, categories]

def write_tensorboard_log(callback, names, logs, batch_no):
    """ log training metrics """
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

#----------------------------------------
#----------------------------------------
# Arguments
parser = argparse.ArgumentParser(description="Unofficial implementation of SpecGAN - Generate audio through spectrogram image with adversarial training")
parser.add_argument("--output_dir", "-o", required=True, help="Directory to output generated files to")
parser.add_argument("--input_data", "-i", required=True, help="Training data in .npz created with preprocess.py")
parser.add_argument("--epochs", "-e", default=1000, type=int, help="Total number of Epochs")
parser.add_argument("--checkpoints", "-cp", default=50, type=int, help="Save model at each (checkpoints) epochs")
args = parser.parse_args()

if os.path.exists(args.output_dir) is False:
    os.mkdir(args.output_dir)
if os.path.exists("./logs/") is False:
    os.mkdir("./logs/")
#---------------------------------
# load data

if os.path.exists(args.input_data) is False:
    print "training data not found at:", args.input_data
    exit()

# First we load nomalized spectrogram data
datapath = args.input_data
X_train_ = np.load(datapath)["specs"]
y_train_ = np.load(datapath)["categories"]

nb_categories = int(np.max(y_train_)) + 1 # number of categories
y_train_ = to_categorical(y_train_)
X_train_ = X_train_[:, :, :, None]

# for denomalizing mel_spectrogram
mel_means = np.load(datapath)["mean"]
mel_stds = np.load(datapath)["std"]

# Now we initialize the generator and discriminator.
generator = make_generator()
discriminator = make_discriminator(nb_categories)

####################
# GENERATOR MODEL
for layer in discriminator.layers:
    layer.trainable = False
discriminator.trainable = False
generator.trainable = True

generator_input = Input(shape=(100,)) # 100 = dimention of random input vector
generator_layers = generator(generator_input)

# replace input layer of discriminator with generatoer output
# TODO: Are there any smarter way?
d_layers_for_generator,_ = swap_input(discriminator, generator_layers)
generator_model = Model(inputs=[generator_input], outputs= [d_layers_for_generator])
generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9), loss=[wasserstein_loss])
generator_model.summary()


####################
# DISCRIMINATOR MODEL
for layer in discriminator.layers:
    layer.trainable = True
for layer in generator.layers:
    layer.trainable = False
discriminator.trainable = True
generator.trainable = False

real_samples = Input(shape=X_train_.shape[1:])
generator_input_for_discriminator = Input(shape=(100,))  # random seed input
generated_samples_for_discriminator = generator(generator_input_for_discriminator)  # random seed -> generator
d_output_from_generator, d_output_from_generator_categories = swap_input(discriminator, generated_samples_for_discriminator) # # random seed -> generator -> discriminator
d_output_from_real_samples, d_output_from_real_samples_categories = swap_input(discriminator, real_samples) # real spectrogram images -> discriminator_loss_real

# We also need to generate weighted-averages of real and generated samples, to use for the gradient norm penalty.
averaged_samples = RandomWeightedAverage()([real_samples, generated_samples_for_discriminator])
averaged_samples_out, averaged_samples_out_categories = swap_input(discriminator,averaged_samples) # weighted-averages of real and generated samples -> discriminator

# The gradient penalty loss function: https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py"""
partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=averaged_samples,
                          gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error

# input: real samples / random seed for generator
# output: real samples -> discriminator / random seed -> generator -> discriminator_loss /
#         weighted-averages of real and generated samples -> discriminator / real samples -> discriminator -> categorical output
discriminator_model = Model(inputs=[real_samples, generator_input_for_discriminator],
                            outputs=[d_output_from_real_samples, d_output_from_generator,
                                     averaged_samples_out, d_output_from_real_samples_categories])
# We use the Adam paramaters from Gulrajani et al. We use the Wasserstein loss for both the real and generated
# samples, the gradient penalty loss for the averaged samples and categorical crossentropy for category
discriminator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                            loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss, categorical_crossentropy])
discriminator_model.summary()

##################
# Training

# labels
positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32) # for real samples
negative_y = -positive_y                                # for generated fake samples
dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32) # passed to the gradient_penalty loss function and is not used.

# TensorBoard
now=datetime.datetime.now()
datestr = now.strftime("%y%m%d-%H%M%s")

# for generator
log_path = './logs/generator/' + datestr
g_callback = TensorBoard(log_path)
g_callback.set_model(generator_model)
g_names = ["generator_loss", "generator_loss_ws"]
print "for generator"
print "tensorboard --port 6006 --logdir " + log_path + " &"

log_path = './logs/discriminator/' + datestr
d_callback = TensorBoard(log_path)
d_callback.set_model(discriminator_model)
d_names = ["discriminator_loss", "discriminator_loss_real", "discriminator_loss_fake",
                            "discriminator_loss_averaged", "discriminator_loss_categorical"]
print "for discriminator"
print "tensorboard --port 6007 --logdir " + log_path + " &"

for epoch in range(args.epochs):
    # shuffleing samples
    indices = np.arange(X_train_.shape[0])
    np.random.shuffle(indices)
    X_train = X_train_[indices]
    y_train = y_train_[indices]

    print("Epoch: ", epoch)
    print("Number of batches: ", int(X_train.shape[0] // BATCH_SIZE))

    minibatches_size = BATCH_SIZE * TRAINING_RATIO
    batch_per_epoch = int(X_train.shape[0] // (BATCH_SIZE * TRAINING_RATIO))
    for i in range(batch_per_epoch):
        discriminator_minibatches = X_train[i * minibatches_size:(i + 1) * minibatches_size]
        categories_minibatches = y_train[i * minibatches_size:(i + 1) * minibatches_size]

        # training D. D will be trained (TRAINING_RATIO) times more than G
        for j in range(TRAINING_RATIO):
            image_batch = discriminator_minibatches[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
            categories_batch = categories_minibatches[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]

            noise = np.random.rand(BATCH_SIZE, 100).astype(np.float32)
            d_logs = discriminator_model.train_on_batch([image_batch, noise], [positive_y, negative_y, dummy_y, categories_batch])
            nb_batch = (j + i * TRAINING_RATIO)  + epoch * (batch_per_epoch * TRAINING_RATIO)
            write_tensorboard_log(d_callback, d_names, d_logs, nb_batch)

        # training G
        g_logs = generator_model.train_on_batch(np.random.rand(BATCH_SIZE, 100), [positive_y])
        nb_batch =  epoch * (batch_per_epoch * TRAINING_RATIO) + i * TRAINING_RATIO
        write_tensorboard_log(g_callback, g_names, [g_logs], nb_batch)

    # export generated images and save sample audio per each epoch
    generate_images(generator, args.output_dir, epoch)

    # save models at checkpoints
    if epoch % args.checkpoints == 0:
        outfile = os.path.join(args.output_dir, 'generator_epoch_{}_{:.3}.h5'.format(epoch, g_logs))
        generator.save_weights(outfile)
        outfile = os.path.join(args.output_dir, 'discriminator_epoch_{}_{:.3}.h5'.format(epoch, d_logs[0]))
        discriminator.save_weights(outfile)
