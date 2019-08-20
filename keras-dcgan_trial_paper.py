from keras.models import Sequential
from keras.layers import Dense, Conv2DTranspose
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten
from keras.optimizers import Adam,SGD
import numpy as np
from PIL import Image
import os
import glob
import random
import argparse
import cv2
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.layers import Input
from keras.layers.core import Dropout
from keras.applications.vgg16 import VGG16
from keras.models import Model

n_colors = 3

def generator_model():
    model = Sequential()

    model.add(Dense(8*8*128, input_shape=(32*32,))) #1024,100
    #model.add(Activation('tanh'))

    #model.add(Dense(128 * 16 * 16)) #128
    model.add(BatchNormalization())
    model.add(Activation('tanh'))

    model.add(Reshape((8, 8, 128)))
    model.add(Conv2DTranspose(128, (5, 5), activation='tanh', strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(64, (5, 5), activation='tanh', strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(32, (5, 5), activation='tanh', strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(n_colors,(5, 5), activation='tanh', strides=2, padding='same'))
    #model.add(BatchNormalization())

    return model

def discriminator_model():
    model = Sequential()
    
    model.add(Conv2D(16, (5, 5), input_shape=(128, 128, n_colors), padding='same'))
    #model.add(BatchNormalization())
    #model.add(LeakyReLU(alpha=0.01))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (5, 5), padding='same'))
    #model.add(BatchNormalization())
    #model.add(LeakyReLU(alpha=0.01))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    #model.add(BatchNormalization())
    #model.add(LeakyReLU(alpha=0.01))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5), padding='same'))
    #model.add(BatchNormalization())
    #model.add(LeakyReLU(alpha=0.01))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (5, 5), padding='same'))
    #model.add(BatchNormalization())
    #model.add(LeakyReLU(alpha=0.01))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
        
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('tanh'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    #discriminator.trainable = False
    model.add(discriminator)
    return model

def image_batch(batch_size):
    files = glob.glob("./in_images/**/*.png", recursive=True)
    files = random.sample(files, batch_size)
    # print(files)
    res = []
    for path in files:
        img = Image.open(path)
        img = img.resize((128, 128))  #(64, 64)
        arr = np.array(img)
        arr = (arr - 127.5) / 127.5
        arr.resize((128, 128, n_colors)) #(64, 64)
        res.append(arr)
    return np.array(res)

def combine_images(generated_images, cols=5, rows=5):
    shape = generated_images.shape
    h = shape[1]
    w = shape[2]
    image = np.zeros((rows * h,  cols * w, n_colors))
    for index, img in enumerate(generated_images):
        if index >= cols * rows:
            break
        i = index // cols
        j = index % cols
        image[i*h:(i+1)*h, j*w:(j+1)*w, :] = img[:, :, :]
    image = image * 127.5 + 127.5
    image = Image.fromarray(image.astype(np.uint8))
    return image

def set_trainable(model, trainable):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

def main(BATCH_SIZE=55, ite=1000):
    batch_size = BATCH_SIZE
    discriminator = discriminator_model()
    generator = generator_model()

    discriminator_on_generator = generator_containing_discriminator(generator, discriminator)
    set_trainable(discriminator, False)
    opt = optimizer=Adam(lr=0.0001, beta_1=0.5) #"SGD" # (lr=0.01, decay=1e-4, momentum=0.2, nesterov=True) Adam(alpha=1e-4, beta1=0.5)
    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=opt)
    
    print('generator.summary()---')
    generator.summary()
    print('discriminator_on_generator.summary()---')
    discriminator_on_generator.summary()

    set_trainable(discriminator, True)
    discriminator.compile(loss='binary_crossentropy', optimizer=opt)
    #SGD(decay=1e-4, lr=0.01,momentum=0.9, nesterov=True)     
    #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) keras
    #set_trainable(discriminator, True)
    print('discriminator.summary()---')
    discriminator.summary()
    #generator.load_weights('./gen_images/generator_11000.h5', by_name=True)
    #discriminator.load_weights('./gen_images/discriminator_11000.h5', by_name=True)

    for i in range(0 * 1000,31 * 1000):
        batch_images = image_batch(batch_size)
        noise = np.random.uniform(size=[batch_size, 32*32], low=-1.0, high=1.0) #32*32
        generated_images = generator.predict(noise)
        X = np.concatenate((batch_images, generated_images))
        y = [1] * batch_size + [0] * batch_size
        d_loss = discriminator.train_on_batch(X, y)
        noise = np.random.uniform(size=[batch_size, 32*32], low=-1.0, high=1.0) ##32*32
        g_loss = discriminator_on_generator.train_on_batch(noise, [1] * batch_size)
        if i % 100 == 0:
            print("step %d d_loss, g_loss : %g %g" % (i, d_loss, g_loss))
            image = combine_images(generated_images)
            #os.system('mkdir -p ./gen_images')
            os.makedirs(os.path.join(".", "gen_images"), exist_ok=True)
            image.save("./gen_images/gen%05d.png" % i)
            if i%ite == 0:
                generator.save_weights('./gen_images/generator_%d.h5' % i, True)
                discriminator.save_weights('./gen_images/discriminator_%d.h5' % i, True)
            
def generate(BATCH_SIZE=55, ite=10000, nice=False):
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))  #optimizer=Adam(lr=0.0002, beta_1=0.5)) #optimizer="SGD"
    g.load_weights('./gen_images/generator_%d.h5'% ite)
    if nice:
        d = discriminator_model()
        d.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5)) #optimizer="SGD"
        d.load_weights('./gen_images/discriminator_%d.h5'% ite)
        noise = np.random.uniform(-1, 1, (BATCH_SIZE*20, 32*32)) ##32*32
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        for i in range(10):
            noise = np.random.uniform(size=[BATCH_SIZE, 32*32], low=-1.0, high=1.0) ##32*32
            #print(noise)
            plt.imshow(noise[0].reshape(10,10)) #32,32
            plt.pause(0.01)
            generated_images = g.predict(noise)
            plt.imshow(generated_images[0])
            plt.pause(0.01)
            image = combine_images(generated_images)
            image.save("./gen_images/generate4_%05d_%d.png" % (ite,i))
            print(i)
    #os.system('mkdir -p ./gen_images')
    os.makedirs(os.path.join(".", "gen_images"), exist_ok=True)
    image.save("./gen_images/generate%05d.png" % ite)
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=55)
    parser.add_argument("--iteration", type=int, default=10000)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        main(BATCH_SIZE=args.batch_size,ite=args.iteration)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size,ite=args.iteration, nice=args.nice)
    
