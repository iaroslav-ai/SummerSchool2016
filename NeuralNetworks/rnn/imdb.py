import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, GRU, SimpleRNN
from keras.optimizers import Adam
from keras.datasets import imdb

max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

(X, Y), (Xv, Yv) = imdb.load_data(nb_words=max_features)

X = sequence.pad_sequences(X, maxlen=maxlen)
Xv = sequence.pad_sequences(Xv, maxlen=maxlen)

model = Sequential()
model.add(Embedding(max_features, 128, dropout=0.2))
model.add(GRU(128))  # try using a GRU instead, for fun
model.add(Dense(1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model.fit(X, Y, batch_size=batch_size, nb_epoch=15,
          validation_data=(Xv, Yv))
score, acc = model.evaluate(Xv, Yv,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)