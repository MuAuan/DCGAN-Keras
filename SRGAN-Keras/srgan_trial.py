"""
Super-resolution of CelebA using Generative Adversarial Networks.

The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0

Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to 'datasets/'
4. Run the sript using command 'python srgan.py'
"""

from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.applications import VGG19
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os
import argparse
import keras.backend as K
from PIL import Image
import cv2
from keras.layers import Dense, Conv2DTranspose
from keras.layers.convolutional import Conv2D, MaxPooling2D

class SRGAN():
    def __init__(self):
        # Input shape
        self.channels = 3
        self.lr_height = 64                 # Low resolution height
        self.lr_width = 64                  # Low resolution width
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        self.hr_height = self.lr_height*4   # High resolution height
        self.hr_width = self.lr_width*4     # High resolution width
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)

        # Number of residual blocks in the generator
        self.n_residual_blocks = 16

        optimizer = Adam(0.0002, 0.5)

        # We use a pre-trained VGG19 model to extract image features from the high resolution
        # and the generated high resolution images and minimize the mse between them
        self.vgg = self.build_vgg()
        self.vgg.trainable = False
        self.vgg.compile(loss='binary_crossentropy', #mse
            optimizer=optimizer,
            metrics=['accuracy'])

        # Configure data loader
        self.dataset_name = 'img_align_anime'  #'img_align_celeba'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.hr_height, self.hr_width))

        # Calculate output shape of D (PatchGAN)
        patch = int(self.hr_height / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        print('discriminator.summary()')
        self.discriminator.summary()
        self.discriminator.load_weights('./weights2/discriminator_1800.h5')
        
        # Build the generator
        self.generator = self.build_generator()
        print('generator.summary()')
        self.generator.summary()
        self.generator.load_weights('./weights2/generator_1800.h5')
 
        # High res. and low res. images
        img_hr = Input(shape=self.hr_shape)
        img_lr = Input(shape=self.lr_shape)

        # Generate high res. version from low res.
        fake_hr = self.generator(img_lr)

        # Extract image features of the generated img
        fake_features = self.vgg(fake_hr)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminator determines validity of generated high res. images
        validity = self.discriminator(fake_hr)

        self.combined = Model([img_lr, img_hr], [validity, fake_features])
        self.combined.compile(loss=['binary_crossentropy', 'mse'],
                              loss_weights=[1e-3, 1],
                              optimizer=optimizer)
        print('combined.summary()')
        self.combined.summary()


    def build_vgg(self):
        """
        Builds a pre-trained VGG19 model that outputs image features extracted at the
        third block of the model
        """
        vgg = VGG19(weights="imagenet")
        # Set outputs to outputs of last conv. layer in block 3
        # See architecture at: https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
        vgg.outputs = [vgg.layers[9].output]

        img = Input(shape=self.hr_shape)

        # Extract image features
        img_features = vgg(img)

        return Model(img, img_features)
    
    def build_generator(self):
        img_lr = Input(shape=self.lr_shape)

        x=Conv2D(64, (3, 3), activation='relu', strides=1, padding='same')(img_lr)
        x=Conv2DTranspose(128, (3, 3), activation='relu', strides=2, padding='same')(x)
        x=Conv2DTranspose(256, (3, 3), activation='relu', strides=2, padding='same')(x)
        gen_hr=Conv2D(self.channels,(3, 3), activation='tanh', strides=1, padding='same')(x)
        return Model(img_lr, gen_hr)
    
    def build_discriminator(self):
        img_hr = Input(shape=self.hr_shape)
    
        x=Conv2D(64, (5, 5), activation='tanh', padding='same')(img_hr)
        x=MaxPooling2D(pool_size=(2, 2))(x)
        x=Conv2D(128, (5, 5), activation='tanh', padding='same')(x)
        x=MaxPooling2D(pool_size=(2, 2))(x)
        x=Conv2D(256, (5, 5), activation='tanh', padding='same')(x)
        x=MaxPooling2D(pool_size=(2, 2))(x)
        x=Conv2D(512, (5, 5), activation='tanh', padding='same')(x)
        x=MaxPooling2D(pool_size=(2, 2))(x)
        x=Conv2D(1024, (5, 5), activation='tanh', padding='same')(x)
        #x=MaxPooling2D(pool_size=(2, 2))(x)
       
        #x=Flatten()(x)
        x=Dense(1024)(x)
        x=Activation('tanh')(x)
        
        validity=Dense(1,activation='sigmoid')(x)
        #validity=Activation('sigmoid')(x)
        
        return Model(img_hr, validity)
    

    def train(self, epochs, batch_size=1, sample_interval=500):

        start_time = datetime.datetime.now()
        BATCH_SIZE=batch_size
        for epoch in range(1800,epochs):

            # ----------------------
            #  Train Discriminator
            # ----------------------

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)
            #noise = np.random.uniform(size=[BATCH_SIZE, 64*64*3], low=-1.0, high=1.0)
            #imgs_lr=noise.reshape(BATCH_SIZE,64,64,3)

            # From low res. image generate high res. version
            fake_hr = self.generator.predict(imgs_lr)

            valid = np.ones((batch_size,) + self.disc_patch)
            fake = np.zeros((batch_size,) + self.disc_patch)

            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ------------------
            #  Train Generator
            # ------------------

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)
            #noise = np.random.uniform(size=[BATCH_SIZE, 64*64*3], low=-1.0, high=1.0)
            #imgs_lr=noise.reshape(BATCH_SIZE,64,64,3)

            # The generators want the discriminators to label the generated images as real
            valid = np.ones((batch_size,) + self.disc_patch)

            # Extract ground truth image features using pre-trained VGG19 model
            image_features = self.vgg.predict(imgs_hr)

            # Train the generators
            g_loss = self.combined.train_on_batch([imgs_lr, imgs_hr], [valid, image_features])

            elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress
            print ("%d time: %s" % (epoch, elapsed_time))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                if epoch % 1000 == 0:
                    self.generator.save_weights('./weights2/generator_%d.h5' % epoch, True)
                    self.discriminator.save_weights('./weights2/discriminator_%d.h5' % epoch, True)                

    def sample_images(self, epoch):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 4, 3

        imgs_hr, imgs_lr = self.data_loader.load_data(batch_size=4, is_testing=True)
        BATCH_SIZE=4
        #noise = np.random.uniform(size=[BATCH_SIZE, 64*64*3], low=-1.0, high=1.0)
        #imgs_lr=noise.reshape(BATCH_SIZE,64,64,3)

        fake_hr = self.generator.predict(imgs_lr)

        # Rescale images 0 - 1
        imgs_lr = 0.5 * imgs_lr + 0.5
        fake_hr = 0.5 * fake_hr + 0.5
        imgs_hr = 0.5 * imgs_hr + 0.5

        # Save generated images and the high resolution originals
        titles = ['Generated', 'Original', 'Input_low']
        fig, axs = plt.subplots(r, c,figsize=(12, 16))
        cnt = 0
        for row in range(r):
            for col, image in enumerate([fake_hr, imgs_hr, imgs_lr]):
                axs[row, col].imshow(image[row])
                axs[row, col].set_title(titles[col],size=20)
                axs[row, col].axis('off')
            cnt += 1
        fig.savefig("images/%s/%d.png" % (self.dataset_name, epoch))
        plt.close()

        """
        # Save low resolution images for comparison
        for i in range(r):
            fig = plt.figure()
            plt.imshow(imgs_lr[i])
            fig.savefig('images/%s/%d_lowres%d.png' % (self.dataset_name, epoch, i))
            plt.close()
        """
    def generate(self, batch_size=25, sample_interval=50):
        BATCH_SIZE=batch_size
        ite=10000
        self.generator = self.build_generator()
        g = self.generator
        g.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
        g.load_weights('./weights/generator_10000.h5')
        for i in range(10):
            noise = np.random.uniform(size=[BATCH_SIZE, 64*64*3], low=-1.0, high=1.0) ##32*32
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size=BATCH_SIZE, is_testing=True)
            print('noise[0]',imgs_lr[0])
            plt.imshow(imgs_lr[0].reshape(64,64,3)) #32,32
            plt.pause(1)
            noise=imgs_lr.reshape(BATCH_SIZE,64,64,3)
            generated_images = g.predict(noise)
            plt.imshow(generated_images[0])
            plt.pause(1)
            image_noise = combine_images2(noise)
            image_noise.save("./images/noise_%s%d.png" % (ite,i))
            image = combine_images(generated_images)
            image.save("./images/%s%d.png" % (ite,i))
            print(i)
        os.makedirs(os.path.join(".", "gen_images"), exist_ok=True)
        image.save("./images/%s%d.png" % (ite,i))            

def combine_images(generated_images, cols=5, rows=5):
    shape = generated_images.shape
    h = shape[1]
    w = shape[2]
    image = np.zeros((rows * h,  cols * w, 3))
    for index, img in enumerate(generated_images):
        if index >= cols * rows:
            break
        i = index // cols
        j = index % cols
        image[i*h:(i+1)*h, j*w:(j+1)*w, :] = img[:, :, :]
    image = image * 127.5 + 127.5
    image = Image.fromarray(image.astype(np.uint8))
    return image

def combine_images2(generated_images, cols=5, rows=5):
    BATCH_SIZE=25
    imgs=[]
    for i in range(BATCH_SIZE):
        img=cv2.resize(generated_images[i],(256,256))
        imgs.append(img)
    imgs=np.array(imgs)    
    shape = imgs.shape
    h = shape[1]
    w = shape[2]
    image = np.zeros((rows * h,  cols * w, 3))
    for index, img in enumerate(imgs):
        if index >= cols * rows:
            break
        i = index // cols
        j = index % cols
        image[i*h:(i+1)*h, j*w:(j+1)*w, :] = img[:, :, :]
    image = image * 127.5 + 127.5
    image = Image.fromarray(image.astype(np.uint8))
    return image

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    gan = SRGAN()
    args = get_args()
    if args.mode == "train":
        gan.train(epochs=30000, batch_size=1, sample_interval=100)
    elif args.mode == "generate":
        gan.generate(batch_size=25, sample_interval=100)
