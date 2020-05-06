from keras.models import Sequential, Model
from keras.layers import Dense, Input, Reshape, Flatten, BatchNormalization, Activation
import numpy as np
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# Hyper Parameters
BATCH_SIZE = 64
LR_G = 0.0002  # learning rate for generator
LR_D = 0.0002  # learning rate for discriminator
N_IDEAS = 5  # think of this as number of ideas for generating an art work (Generator)
ART_COMPONENTS = 15  # it could be total point G can draw in the canvas
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])

# show our beautiful painting range
plt.close('all')
plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
plt.legend(loc='upper right')
plt.show()


def artist_works():  # painting from the famous artist (real target)
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    paintings = a * np.power(PAINT_POINTS, 2) + (a - 1)
    return paintings


def G():
    model = Sequential()
    model.add(Dense(64, input_shape=(N_IDEAS,), activation= 'relu'))
    model.add(Dense(64, input_shape=(64,), activation= 'relu' ))
    model.add(Dense(ART_COMPONENTS, input_shape= (64,)))
    return model

def D():
    mdl = Sequential()
    mdl.add(Dense(64, input_shape=(ART_COMPONENTS,), activation= 'relu'))
    mdl.add(Dense(64, input_shape=(64,), activation='relu'))
    mdl.add(Dense(1, input_shape=(64,), activation='sigmoid'))
    return mdl

G=G()
D=D()
D.compile(loss='binary_crossentropy', optimizer=Adam(lr=LR_D),metrics=['accuracy'])
z = Input(shape=(N_IDEAS,))
img = G(z)
D.trainable = False

validity = D(img)
C = Model(z, validity)
C.compile(loss='binary_crossentropy', optimizer=Adam(lr=LR_D))



plt.ion()  # something about continuous plotting

for step in range(5000):
    imgs = artist_works()  # real painting from artist
    valid = np.ones((BATCH_SIZE, 1))
    fake = np.zeros((BATCH_SIZE, 1))

    noise = np.random.uniform(0,1,(BATCH_SIZE, N_IDEAS))
    gen_imgs = G.predict(noise)

    #train Discriminator
    D.train_on_batch(imgs, valid)
    D.train_on_batch(gen_imgs, fake)

    #train Generator
    C.train_on_batch(noise, valid)

    if step % 500 == 0:  # plotting
        plt.cla()
        plt.plot(PAINT_POINTS[0], gen_imgs[0], c='#4AD631', lw=3, label='Generated painting', )
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')

        plt.ylim((0, 3))
        plt.legend(loc='upper right', fontsize=10)
        plt.draw()
        plt.pause(0.01)

plt.ioff()
plt.show()
