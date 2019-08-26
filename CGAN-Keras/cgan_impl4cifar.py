from __future__ import print_function, division

from keras.datasets import mnist,cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Conv2DTranspose
from keras.layers.convolutional import MaxPooling2D
import matplotlib.pyplot as plt

import numpy as np
import argparse

class CGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 10
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])
        
        #self.discriminator.load_weights('./gen_images_cgan/discriminator_%d.h5' % 75000)

        # Build the generator
        self.generator = self.build_generator()
        #self.generator.load_weights('./gen_images_cgan/generator_%d.h5' % 75000)

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
            optimizer=optimizer)

    def build_generator(self):
        n_colors=self.channels
        model = Sequential()

        model.add(Dense(1024, input_dim=self.latent_dim))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('tanh'))

        model.add(Reshape((8,8,16)))
        model.add(Conv2D(16, (3, 3), activation='tanh', strides=1, padding='same'))
        #model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2DTranspose(128, (3, 3), activation='tanh', strides=2, padding='same'))
        #model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2DTranspose(64, (3, 3), activation='tanh', strides=2, padding='same'))
        #model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2DTranspose(n_colors,(3, 3), activation='tanh', strides=1, padding='same'))
        #model.add(BatchNormalization(momentum=0.8))
        
        print('generator')
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):
        n_colors=self.channels
        model = Sequential()
        model.add(Dense(3072, input_dim=np.prod(self.img_shape)))
        model.add(Reshape(self.img_shape))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    
        model.add(Conv2D(32, (5, 5), padding='same'))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    
        model.add(Conv2D(64, (5, 5), padding='same'))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    
        model.add(Conv2D(128, (5, 5), padding='same'))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('tanh'))

        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        print('discriminator')
        model.summary()

        img = Input(shape=self.img_shape)
        print(img.shape)
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        flat_img = Flatten()(img)

        model_input = multiply([flat_img, label_embedding])

        validity = model(model_input)

        return Model([img, label], validity)
    

    def train(self, epochs, batch_size=128, sample_interval=1000):

        # Load the dataset
        (X_train, y_train), (_, _) = cifar10.load_data()

        # Configure input
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        y_train = y_train.reshape(-1, 1)
        print('X_train.shape',X_train.shape[0])

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(0,epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx].reshape(32,32,32,3), y_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, labels])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)  #draw images
                # Plot the progress
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
                if epoch%5000 == 0:
                    self.generator.save_weights('./gen_images_cgan/weights/generator_%d.h5' % epoch, True)
                    self.discriminator.save_weights('./gen_images_cgan/weights/discriminator_%d.h5' % epoch, True)

    def sample_images(self, epoch):
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.arange(0, 10).reshape(-1, 1)

        gen_imgs = self.generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(np.clip(gen_imgs[cnt,:,:,:],0,1.)) 
                axs[i,j].set_title("Digit: %d" % sampled_labels[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("./gen_images_cgan/%d.png" % epoch)
        plt.close()

    def generate(self, epoch):
        r, c = 10, 10
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.arange(0, 100).reshape(-1, 1)
        for k in range(0,100000,5000):
            print('k',k)
            gen_weights=k
            self.generator.load_weights('./gen_images_cgan/weights/generator_%d.h5' % gen_weights)
        
            for i in range(100):
                sampled_labels[i]=i%10
        
            gen_imgs = self.generator.predict([noise, sampled_labels])
        
            # Rescale images 0 - 1
            gen_imgs = 0.5 * gen_imgs + 0.5

            fig, axs = plt.subplots(r, c,figsize=(16,16))
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i,j].imshow(np.clip(gen_imgs[cnt,:,:,:],0,1.))
                    if i==0:
                        axs[i,j].set_title("Digit: %d" % sampled_labels[cnt])
                    axs[i,j].axis('off')
                    cnt += 1
            fig.savefig("./gen_images_cgan/%d_%d.png" % (gen_weights,epoch))
            plt.close()        

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    args = parser.parse_args()
    return args
        
if __name__ == '__main__':
    args = get_args()
    cgan = CGAN()
    if args.mode == "train":
        cgan.train(epochs=100000, batch_size=32, sample_interval=1000)
    elif args.mode == "generate":
        for epoch in range(1):
            cgan.generate(epoch)

"""
discriminator
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 3072)              9440256
_________________________________________________________________
reshape_1 (Reshape)          (None, 32, 32, 3)         0
_________________________________________________________________
activation_1 (Activation)    (None, 32, 32, 3)         0
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 16, 16, 3)         0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 16, 16, 32)        2432
_________________________________________________________________
activation_2 (Activation)    (None, 16, 16, 32)        0
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 8, 8, 32)          0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 8, 8, 64)          51264
_________________________________________________________________
activation_3 (Activation)    (None, 8, 8, 64)          0
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 4, 4, 64)          0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 4, 4, 128)         204928
_________________________________________________________________
activation_4 (Activation)    (None, 4, 4, 128)         0
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 2, 2, 128)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 512)               0
_________________________________________________________________
dense_2 (Dense)              (None, 128)               65664
_________________________________________________________________
activation_5 (Activation)    (None, 128)               0
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 129
_________________________________________________________________
activation_6 (Activation)    (None, 1)                 0
=================================================================
Total params: 9,764,673
Trainable params: 9,764,673
Non-trainable params: 0
_________________________________________________________________
(?, 32, 32, 3)

generator
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_4 (Dense)              (None, 1024)              103424
_________________________________________________________________
batch_normalization_1 (Batch (None, 1024)              4096
_________________________________________________________________
activation_7 (Activation)    (None, 1024)              0
_________________________________________________________________
reshape_2 (Reshape)          (None, 8, 8, 16)          0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 8, 8, 16)          2320
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr (None, 16, 16, 16)        2320
_________________________________________________________________
conv2d_transpose_2 (Conv2DTr (None, 32, 32, 64)        9280
_________________________________________________________________
conv2d_transpose_3 (Conv2DTr (None, 32, 32, 3)         1731
=================================================================
Total params: 123,171
Trainable params: 121,123
Non-trainable params: 2,048
_________________________________________________________________
"""