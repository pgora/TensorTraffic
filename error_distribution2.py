import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import pickle

all_data = tf.contrib.learn.datasets.base.load_csv_without_header(
      filename='micro_data.csv',
      target_dtype=np.float32,
      features_dtype=np.float32)

def get_sample(data, sample_size = 20000):
    h = np.random.permutation(data.target.shape[0])
    X_sample = data.data[h[:sample_size], :]
    target_sample = data.target[h[:sample_size]]
    return X_sample, target_sample

X, y = get_sample(all_data, sample_size=all_data.target.shape[0])

X = np.delete(X, [15, 16], 1)

X = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, keepdims=True)

y_std = np.std(y)
y_min = np.min(y)
y = (y - y_min) / y_std + 1


def build_model(no_layers=2, no_units=100, dropout=0.6):
    initial_dropout = dropout / no_layers
    model = Sequential()
    model.add(Dense(no_units, input_dim=X.shape[1], activation='relu'))
    model.add(Dropout(initial_dropout))
    model.add(BatchNormalization())

    for i in range(no_layers - 1):
        model.add(Dense(no_units, activation='relu'))
        model.add(Dropout(initial_dropout * (i + 2)))
        model.add(BatchNormalization())

    model.add(Dense(1))
    model.compile(loss='mape', metrics=[], optimizer=Adam(lr=0.001))
    return model

model = build_model(no_layers=3)

def train_test_split(X, y, train_ratio):
    h = np.random.permutation(X.shape[0])
    n_train = int(train_ratio * X.shape[0])
    X_train = X[h[:n_train], :]
    X_test = X[h[n_train:], :]
    y_train = y[h[:n_train]]
    y_test = y[h[n_train:]]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(X, y, 0.8)


history = model.fit(X_train,
                    y_train,
                    validation_data=(X_test, y_test),
                    epochs=100000,
                    batch_size=10240,
                    verbose=0,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=10000)])


# %matplotlib inline
# import matplotlib.pyplot as plt

# plt.plot(history.history['loss'], c='red')
# plt.plot(history.history['val_loss'], c='blue')


# plt.plot(history.history['loss'][200:], c='red')
# plt.plot(history.history['val_loss'][200:], c='blue')

with open('history.pickle', 'wb') as handle:
    pickle.dump(history, handle, protocol=2)


def get_errors(actual, predicted):
    actual = actual.flatten()
    predicted = predicted.flatten()
    actual = (actual - 1) * y_std + y_min
    predicted = (predicted - 1) * y_std + y_min
    error = np.abs(actual - predicted)
    rel_error = np.abs(actual - predicted) / actual
    return np.max(error), np.mean(error), np.max(rel_error), np.mean(rel_error)


predicted = model.predict(X_test)
get_errors(y_test, predicted)


predicted_train = model.predict(X_train)
get_errors(y_train, predicted_train)


