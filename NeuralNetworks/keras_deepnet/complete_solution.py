import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense
from keras.optimizers import SGD, Adam, RMSprop

# the data, shuffled and split between train and test sets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

"""
from matplotlib import pyplot as plt
plt.imshow(X_train[0])
print Y_train[0]
plt.show()
"""

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

ipt = Input(shape=(28, 28))

x = Flatten()(ipt)
x = Dense(512, activation="relu")(x)
x = Dense(512, activation="relu")(x)
x = Dense(10, activation="softmax")(x)

model = Model(input=ipt, output=x)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=256, nb_epoch=20,
                    verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])