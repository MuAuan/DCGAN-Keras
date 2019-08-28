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
        #d = Activation('relu')(d)
        #d = LeakyReLU(alpha=0.2)(d)
        if strides==2:
            d = MaxPooling2D(pool_size=(2, 2))(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    # Input img
    d0 = Input(shape=hr_shape)

    #d1 = d_block(d0, df , bn=False)
    d2 = d_block(d0, df, strides=2, bn=False)
    #d3 = d_block(d2, df*2, bn=False)
    d4 = d_block(d2, df*2, strides=2, bn=False)
    #d5 = d_block(d4, df*4, bn=False)
    d6 = d_block(d4, df*4, strides=2, bn=False)
    #d7 = d_block(d6, df*8, bn=False)
    d8 = d_block(d6, df*8, strides=2, bn=False)
    #d9 = d_block(d8, df*8, bn=False)
    d10 = d_block(d8, df*8, strides=2, bn=False)
    
    x=Flatten()(d10)
    d11 = Dense(df*16)(x)
    d12 = Activation('tanh')(d11)
    #d10 = LeakyReLU(alpha=0.2)(d9)
    validity = Dense(1, activation='sigmoid')(d12)
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
    opt = optimizer=Adam(lr=0.00005, beta_1=0.5) 
    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=opt)
    
    print('generator.summary()---')
    generator.summary()
    print('discriminator_on_generator.summary()---')
    discriminator_on_generator.summary()

    set_trainable(discriminator, True)
    discriminator.compile(loss='binary_crossentropy', optimizer=opt)
    print('discriminator.summary()---')
    discriminator.summary()
    generator.load_weights('./gen_images/generator_5000.h5', by_name=True)
    discriminator.load_weights('./gen_images/discriminator_5000.h5', by_name=True)

    for i in range(5 * 1000,31 * 1000):
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
    
"""
discriminator.summary()---
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 128, 128, 3)       0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 128, 128, 16)      448
_________________________________________________________________
activation_1 (Activation)    (None, 128, 128, 16)      0
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 64, 64, 16)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 64, 64, 32)        4640
_________________________________________________________________
activation_2 (Activation)    (None, 64, 64, 32)        0
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 32, 32, 32)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 32, 32, 64)        18496
_________________________________________________________________
activation_3 (Activation)    (None, 32, 32, 64)        0
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 16, 16, 64)        0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 16, 16, 128)       73856
_________________________________________________________________
activation_4 (Activation)    (None, 16, 16, 128)       0
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 8, 8, 128)         0
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 8, 8, 128)         147584
_________________________________________________________________
activation_5 (Activation)    (None, 8, 8, 128)         0
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 4, 4, 128)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 2048)              0
_________________________________________________________________
dense_1 (Dense)              (None, 256)               524544
_________________________________________________________________
activation_6 (Activation)    (None, 256)               0
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 257
=================================================================
Total params: 769,825
Trainable params: 769,825
Non-trainable params: 0
_________________________________________________________________

generator.summary()---
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_2 (InputLayer)            (None, 10)           0
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 8192)         90112       input_2[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 8192)         32768       dense_3[0][0]
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 8192)         0           batch_normalization_1[0][0]
__________________________________________________________________________________________________
reshape_1 (Reshape)             (None, 8, 8, 128)    0           activation_7[0][0]
__________________________________________________________________________________________________
conv2d_transpose_1 (Conv2DTrans (None, 16, 16, 128)  409728      reshape_1[0][0]
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 16, 16, 128)  512         conv2d_transpose_1[0][0]
__________________________________________________________________________________________________
conv2d_transpose_2 (Conv2DTrans (None, 32, 32, 64)   204864      batch_normalization_2[0][0]
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 32, 32, 64)   256         conv2d_transpose_2[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 32, 32, 64)   331840      batch_normalization_3[0][0]
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 32, 32, 64)   0           conv2d_6[0][0]
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 32, 32, 64)   36928       activation_8[0][0]
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 32, 32, 64)   0           conv2d_7[0][0]
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 32, 32, 64)   256         activation_9[0][0]
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 32, 32, 64)   36928       batch_normalization_4[0][0]
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 32, 32, 64)   256         conv2d_8[0][0]
__________________________________________________________________________________________________
add_1 (Add)                     (None, 32, 32, 64)   0           batch_normalization_5[0][0]
                                                                 activation_8[0][0]
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 32, 32, 64)   36928       add_1[0][0]
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 32, 32, 64)   0           conv2d_9[0][0]
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 32, 32, 64)   256         activation_10[0][0]
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 32, 32, 64)   36928       batch_normalization_6[0][0]
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 32, 32, 64)   256         conv2d_10[0][0]
__________________________________________________________________________________________________
add_2 (Add)                     (None, 32, 32, 64)   0           batch_normalization_7[0][0]
                                                                 add_1[0][0]
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 32, 32, 64)   36928       add_2[0][0]
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 32, 32, 64)   0           conv2d_11[0][0]
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 32, 32, 64)   256         activation_11[0][0]
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 32, 32, 64)   36928       batch_normalization_8[0][0]
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 32, 32, 64)   256         conv2d_12[0][0]
__________________________________________________________________________________________________
add_3 (Add)                     (None, 32, 32, 64)   0           batch_normalization_9[0][0]
                                                                 add_2[0][0]
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 32, 32, 64)   36928       add_3[0][0]
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 32, 32, 64)   0           conv2d_13[0][0]
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 32, 32, 64)   256         activation_12[0][0]
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 32, 32, 64)   36928       batch_normalization_10[0][0]
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 32, 32, 64)   256         conv2d_14[0][0]
__________________________________________________________________________________________________
add_4 (Add)                     (None, 32, 32, 64)   0           batch_normalization_11[0][0]
                                                                 add_3[0][0]
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 32, 32, 64)   36928       add_4[0][0]
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 32, 32, 64)   0           conv2d_15[0][0]
__________________________________________________________________________________________________
batch_normalization_12 (BatchNo (None, 32, 32, 64)   256         activation_13[0][0]
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 32, 32, 64)   36928       batch_normalization_12[0][0]
__________________________________________________________________________________________________
batch_normalization_13 (BatchNo (None, 32, 32, 64)   256         conv2d_16[0][0]
__________________________________________________________________________________________________
add_5 (Add)                     (None, 32, 32, 64)   0           batch_normalization_13[0][0]
                                                                 add_4[0][0]
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 32, 32, 64)   36928       add_5[0][0]
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 32, 32, 64)   0           conv2d_17[0][0]
__________________________________________________________________________________________________
batch_normalization_14 (BatchNo (None, 32, 32, 64)   256         activation_14[0][0]
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 32, 32, 64)   36928       batch_normalization_14[0][0]
__________________________________________________________________________________________________
batch_normalization_15 (BatchNo (None, 32, 32, 64)   256         conv2d_18[0][0]
__________________________________________________________________________________________________
add_6 (Add)                     (None, 32, 32, 64)   0           batch_normalization_15[0][0]
                                                                 add_5[0][0]
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 32, 32, 64)   36928       add_6[0][0]
__________________________________________________________________________________________________
activation_15 (Activation)      (None, 32, 32, 64)   0           conv2d_19[0][0]
__________________________________________________________________________________________________
batch_normalization_16 (BatchNo (None, 32, 32, 64)   256         activation_15[0][0]
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, 32, 32, 64)   36928       batch_normalization_16[0][0]
__________________________________________________________________________________________________
batch_normalization_17 (BatchNo (None, 32, 32, 64)   256         conv2d_20[0][0]
__________________________________________________________________________________________________
add_7 (Add)                     (None, 32, 32, 64)   0           batch_normalization_17[0][0]
                                                                 add_6[0][0]
__________________________________________________________________________________________________
conv2d_21 (Conv2D)              (None, 32, 32, 64)   36928       add_7[0][0]
__________________________________________________________________________________________________
activation_16 (Activation)      (None, 32, 32, 64)   0           conv2d_21[0][0]
__________________________________________________________________________________________________
batch_normalization_18 (BatchNo (None, 32, 32, 64)   256         activation_16[0][0]
__________________________________________________________________________________________________
conv2d_22 (Conv2D)              (None, 32, 32, 64)   36928       batch_normalization_18[0][0]
__________________________________________________________________________________________________
batch_normalization_19 (BatchNo (None, 32, 32, 64)   256         conv2d_22[0][0]
__________________________________________________________________________________________________
add_8 (Add)                     (None, 32, 32, 64)   0           batch_normalization_19[0][0]
                                                                 add_7[0][0]
__________________________________________________________________________________________________
conv2d_23 (Conv2D)              (None, 32, 32, 64)   36928       add_8[0][0]
__________________________________________________________________________________________________
activation_17 (Activation)      (None, 32, 32, 64)   0           conv2d_23[0][0]
__________________________________________________________________________________________________
batch_normalization_20 (BatchNo (None, 32, 32, 64)   256         activation_17[0][0]
__________________________________________________________________________________________________
conv2d_24 (Conv2D)              (None, 32, 32, 64)   36928       batch_normalization_20[0][0]
__________________________________________________________________________________________________
batch_normalization_21 (BatchNo (None, 32, 32, 64)   256         conv2d_24[0][0]
__________________________________________________________________________________________________
add_9 (Add)                     (None, 32, 32, 64)   0           batch_normalization_21[0][0]
                                                                 add_8[0][0]
__________________________________________________________________________________________________
conv2d_25 (Conv2D)              (None, 32, 32, 64)   36928       add_9[0][0]
__________________________________________________________________________________________________
activation_18 (Activation)      (None, 32, 32, 64)   0           conv2d_25[0][0]
__________________________________________________________________________________________________
batch_normalization_22 (BatchNo (None, 32, 32, 64)   256         activation_18[0][0]
__________________________________________________________________________________________________
conv2d_26 (Conv2D)              (None, 32, 32, 64)   36928       batch_normalization_22[0][0]
__________________________________________________________________________________________________
batch_normalization_23 (BatchNo (None, 32, 32, 64)   256         conv2d_26[0][0]
__________________________________________________________________________________________________
add_10 (Add)                    (None, 32, 32, 64)   0           batch_normalization_23[0][0]
                                                                 add_9[0][0]
__________________________________________________________________________________________________
conv2d_27 (Conv2D)              (None, 32, 32, 64)   36928       add_10[0][0]
__________________________________________________________________________________________________
activation_19 (Activation)      (None, 32, 32, 64)   0           conv2d_27[0][0]
__________________________________________________________________________________________________
batch_normalization_24 (BatchNo (None, 32, 32, 64)   256         activation_19[0][0]
__________________________________________________________________________________________________
conv2d_28 (Conv2D)              (None, 32, 32, 64)   36928       batch_normalization_24[0][0]
__________________________________________________________________________________________________
batch_normalization_25 (BatchNo (None, 32, 32, 64)   256         conv2d_28[0][0]
__________________________________________________________________________________________________
add_11 (Add)                    (None, 32, 32, 64)   0           batch_normalization_25[0][0]
                                                                 add_10[0][0]
__________________________________________________________________________________________________
conv2d_29 (Conv2D)              (None, 32, 32, 64)   36928       add_11[0][0]
__________________________________________________________________________________________________
activation_20 (Activation)      (None, 32, 32, 64)   0           conv2d_29[0][0]
__________________________________________________________________________________________________
batch_normalization_26 (BatchNo (None, 32, 32, 64)   256         activation_20[0][0]
__________________________________________________________________________________________________
conv2d_30 (Conv2D)              (None, 32, 32, 64)   36928       batch_normalization_26[0][0]
__________________________________________________________________________________________________
batch_normalization_27 (BatchNo (None, 32, 32, 64)   256         conv2d_30[0][0]
__________________________________________________________________________________________________
add_12 (Add)                    (None, 32, 32, 64)   0           batch_normalization_27[0][0]
                                                                 add_11[0][0]
__________________________________________________________________________________________________
conv2d_31 (Conv2D)              (None, 32, 32, 64)   36928       add_12[0][0]
__________________________________________________________________________________________________
activation_21 (Activation)      (None, 32, 32, 64)   0           conv2d_31[0][0]
__________________________________________________________________________________________________
batch_normalization_28 (BatchNo (None, 32, 32, 64)   256         activation_21[0][0]
__________________________________________________________________________________________________
conv2d_32 (Conv2D)              (None, 32, 32, 64)   36928       batch_normalization_28[0][0]
__________________________________________________________________________________________________
batch_normalization_29 (BatchNo (None, 32, 32, 64)   256         conv2d_32[0][0]
__________________________________________________________________________________________________
add_13 (Add)                    (None, 32, 32, 64)   0           batch_normalization_29[0][0]
                                                                 add_12[0][0]
__________________________________________________________________________________________________
conv2d_33 (Conv2D)              (None, 32, 32, 64)   36928       add_13[0][0]
__________________________________________________________________________________________________
activation_22 (Activation)      (None, 32, 32, 64)   0           conv2d_33[0][0]
__________________________________________________________________________________________________
batch_normalization_30 (BatchNo (None, 32, 32, 64)   256         activation_22[0][0]
__________________________________________________________________________________________________
conv2d_34 (Conv2D)              (None, 32, 32, 64)   36928       batch_normalization_30[0][0]
__________________________________________________________________________________________________
batch_normalization_31 (BatchNo (None, 32, 32, 64)   256         conv2d_34[0][0]
__________________________________________________________________________________________________
add_14 (Add)                    (None, 32, 32, 64)   0           batch_normalization_31[0][0]
                                                                 add_13[0][0]
__________________________________________________________________________________________________
conv2d_35 (Conv2D)              (None, 32, 32, 64)   36928       add_14[0][0]
__________________________________________________________________________________________________
activation_23 (Activation)      (None, 32, 32, 64)   0           conv2d_35[0][0]
__________________________________________________________________________________________________
batch_normalization_32 (BatchNo (None, 32, 32, 64)   256         activation_23[0][0]
__________________________________________________________________________________________________
conv2d_36 (Conv2D)              (None, 32, 32, 64)   36928       batch_normalization_32[0][0]
__________________________________________________________________________________________________
batch_normalization_33 (BatchNo (None, 32, 32, 64)   256         conv2d_36[0][0]
__________________________________________________________________________________________________
add_15 (Add)                    (None, 32, 32, 64)   0           batch_normalization_33[0][0]
                                                                 add_14[0][0]
__________________________________________________________________________________________________
conv2d_37 (Conv2D)              (None, 32, 32, 64)   36928       add_15[0][0]
__________________________________________________________________________________________________
activation_24 (Activation)      (None, 32, 32, 64)   0           conv2d_37[0][0]
__________________________________________________________________________________________________
batch_normalization_34 (BatchNo (None, 32, 32, 64)   256         activation_24[0][0]
__________________________________________________________________________________________________
conv2d_38 (Conv2D)              (None, 32, 32, 64)   36928       batch_normalization_34[0][0]
__________________________________________________________________________________________________
batch_normalization_35 (BatchNo (None, 32, 32, 64)   256         conv2d_38[0][0]
__________________________________________________________________________________________________
add_16 (Add)                    (None, 32, 32, 64)   0           batch_normalization_35[0][0]
                                                                 add_15[0][0]
__________________________________________________________________________________________________
conv2d_39 (Conv2D)              (None, 32, 32, 64)   36928       add_16[0][0]
__________________________________________________________________________________________________
batch_normalization_36 (BatchNo (None, 32, 32, 64)   256         conv2d_39[0][0]
__________________________________________________________________________________________________
add_17 (Add)                    (None, 32, 32, 64)   0           batch_normalization_36[0][0]
                                                                 activation_8[0][0]
__________________________________________________________________________________________________
up_sampling2d_1 (UpSampling2D)  (None, 64, 64, 64)   0           add_17[0][0]
__________________________________________________________________________________________________
conv2d_40 (Conv2D)              (None, 64, 64, 256)  147712      up_sampling2d_1[0][0]
__________________________________________________________________________________________________
activation_25 (Activation)      (None, 64, 64, 256)  0           conv2d_40[0][0]
__________________________________________________________________________________________________
up_sampling2d_2 (UpSampling2D)  (None, 128, 128, 256 0           activation_25[0][0]
__________________________________________________________________________________________________
conv2d_41 (Conv2D)              (None, 128, 128, 256 590080      up_sampling2d_2[0][0]
__________________________________________________________________________________________________
activation_26 (Activation)      (None, 128, 128, 256 0           conv2d_41[0][0]
__________________________________________________________________________________________________
conv2d_42 (Conv2D)              (None, 128, 128, 3)  62211       activation_26[0][0]
==================================================================================================
Total params: 3,097,155
Trainable params: 3,076,163
Non-trainable params: 20,992
__________________________________________________________________________________________________
discriminator_on_generator.summary()---
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
model_2 (Model)              (None, 128, 128, 3)       3097155
_________________________________________________________________
model_1 (Model)              (None, 1)                 769825
=================================================================
Total params: 3,866,980
Trainable params: 3,076,163
Non-trainable params: 790,817
_________________________________________________________________
"""