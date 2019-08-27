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
from keras.layers import Add

n_colors = 3

def generator_model():

    def residual_block(layer_input, filters):
        """Residual block described in paper"""
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
        d = Activation('tanh')(d)  #relu
        d = BatchNormalization(momentum=0.8)(d)
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
        d = BatchNormalization(momentum=0.8)(d)
        d = Add()([d, layer_input])
        return d

    def deconv2d(layer_input):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
        u = Activation('tanh')(u)  #relu
        return u
    
    gf=64
    n_residual_blocks = 16
    # Low resolution image input
    img_lr = Input(shape=(10,))

    x=Dense(8*8*128, )(img_lr) #1024,100
    
    x=BatchNormalization()(x)
    x=Activation('tanh')(x)
    x=Reshape((8, 8, 128))(x)
    x=Conv2DTranspose(128, (5, 5), activation='tanh', strides=2, padding='same')(x)
    x=BatchNormalization()(x)
    x=Conv2DTranspose(64, (5, 5), activation='tanh', strides=2, padding='same')(x)
    x=BatchNormalization()(x)
    
    # Pre-residual block
    c1 = Conv2D(64, kernel_size=9, strides=1, padding='same')(x)
    c1 = Activation('tanh')(c1) #relu

    # Propogate through residual blocks
    r = residual_block(c1, gf)
    for _ in range(n_residual_blocks - 1):
        r = residual_block(r, gf)

    # Post-residual block
    c2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
    c2 = BatchNormalization(momentum=0.8)(c2)
    c2 = Add()([c2, c1])

    # Upsampling
    u1 = deconv2d(c2)
    u2 = deconv2d(u1)

    # Generate high resolution output
    gen_hr = Conv2D(3, kernel_size=9, strides=1, padding='same', activation='tanh')(u2)
    model=Model(img_lr, gen_hr)
    return model

"""
def generator_model():
    model = Sequential()

    model.add(Dense(8*8*128, input_shape=(10,))) #1024,100
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
"""

def discriminator_model():
    df = 16
    hr_shape=(128, 128, 3)

    def d_block(layer_input, filters, strides=1, bn=True):
        #Discriminator layer
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
        d = Activation('tanh')(d)
        if strides==2:
            d = MaxPooling2D(pool_size=(2, 2))(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    # Input img
    d0 = Input(shape=hr_shape)

    d1 = d_block(d0, df , bn=False)
    d2 = d_block(d1, df, strides=2, bn=False)
    d3 = d_block(d2, df*2, bn=False)
    d4 = d_block(d3, df*2, strides=2, bn=False)
    d5 = d_block(d4, df*4, bn=False)
    d6 = d_block(d5, df*4, strides=2, bn=False)
    d7 = d_block(d6, df*8, bn=False)
    d8 = d_block(d7, df*8, strides=2, bn=False)
    
    x=Flatten()(d8)
    d9 = Dense(df*16)(x)
    d10 = Activation('tanh')(d9)
    validity = Dense(1, activation='sigmoid')(d10)
    model=Model(d0, validity)

    return model
"""    
def discriminator_model():
    model = Sequential()
    
    model.add(Conv2D(16, (5, 5), input_shape=(128, 128, n_colors), padding='same'))
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
    
    model.add(Conv2D(256, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
        
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('tanh'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model
"""
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
    opt = optimizer=Adam(lr=0.0001, beta_1=0.5) 
    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=opt)
    
    print('generator.summary()---')
    generator.summary()
    print('discriminator_on_generator.summary()---')
    discriminator_on_generator.summary()

    set_trainable(discriminator, True)
    discriminator.compile(loss='binary_crossentropy', optimizer=opt)
    print('discriminator.summary()---')
    discriminator.summary()
    #generator.load_weights('./gen_images/generator_11000.h5', by_name=True)
    #discriminator.load_weights('./gen_images/discriminator_11000.h5', by_name=True)

    for i in range(0 * 1000,31 * 1000):
        batch_images = image_batch(batch_size)
        noise = np.random.uniform(size=[batch_size, 10], low=-1.0, high=1.0) #32*32
        
        generated_images = generator.predict(noise)
        X = np.concatenate((batch_images, generated_images))
        y = [1] * batch_size + [0] * batch_size
        d_loss = discriminator.train_on_batch(X, y)
        noise = np.random.uniform(size=[batch_size, 10], low=-1.0, high=1.0) ##32*32
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
    g.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    g.load_weights('./gen_images/generator_%d.h5'%ite)
    if nice:
        d = discriminator_model()
        d.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5)) #optimizer="SGD"
        d.load_weights('./gen_images/discriminator_%d.h5'%ite)
        noise = np.random.uniform(-1, 1, (BATCH_SIZE*20, 10)) ##32*32
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
            noise = np.random.uniform(size=[BATCH_SIZE, 10], low=-1.0, high=1.0) ##32*32
            print('noise[0]',noise[0])
            plt.imshow(noise[0].reshape(32,32)) #32,32
            plt.pause(0.01)
            generated_images = g.predict(noise)
            plt.imshow(generated_images[0])
            plt.pause(0.01)
            #image_noise = combine_images(noise.reshape(BATCH_SIZE,32,32,))
            #image_noise.save("./gen_images/generate_noise_%05d_%d.png" % (ite,i))
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
    parser.add_argument("--iteration", type=int, default=1000)
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
    
