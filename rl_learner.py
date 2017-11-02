import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.merge import concatenate
from keras import backend as K
from keras.optimizers import SGD, Adam, Adamax
from keras.utils import to_categorical
import tensorflow as tf

# from sklearn.model_selection import train_test_split
import h5py
import numpy as np
from random import sample
import math

batch_size = 128*1
num_classes = 9
epochs = 20

# input image dimensions
img_rows, img_cols = 32, 32

# Change tf params
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

def create_model(weights_path=None):
    img_input_shape = (img_rows, img_cols, 1)
    goal_input_shape = (1, )
    # Model
    img_input = Input(shape=img_input_shape, dtype='float32', name='img_input')
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(img_input)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(16, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(8, kernel_size=(3, 3), activation='relu')(x)
    # x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    # x = Dropout(0.25)(x)
    # x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(4, kernel_size=(3, 3), activation='relu')(x)
    # x = Dropout(0.25)(x)
    x = Flatten()(x)

    goal_input = Input(shape=goal_input_shape, dtype='float32', name='goal_input')
    x = concatenate([x, goal_input])

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(32, activation='relu')(x)
    out = Dense(num_classes, activation='softmax')(x) #softmax

    model = Model(inputs = [img_input, goal_input], outputs = out)
    if weights_path:
        model.load_weights(weights_path)
        print('Loaded weights from ' + weights_path)

    optimizer = Adamax(lr=0.02, clipnorm=1e-6, clipvalue=1e6)
    model.compile(loss=keras.losses.categorical_crossentropy, #mse mae
                  optimizer=optimizer, # keras.optimizers.Adadelta()
                  metrics=[ 'accuracy'
                  # 'mae', abs_diff
                  ] )
    return model

def create_simp_model(weights_path=None):
    img_input_shape = (img_rows, img_cols, 1)
    goal_input_shape = (1, )
    # Model
    img_input = Input(shape=img_input_shape, dtype='float32', name='img_input')
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(img_input)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(16, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(8, kernel_size=(3, 3), activation='relu')(x)
    # x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    # x = Dropout(0.25)(x)
    # x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(4, kernel_size=(3, 3), activation='relu')(x)
    # x = Dropout(0.25)(x)
    x = Flatten()(x)

    goal_input = Input(shape=goal_input_shape, dtype='float32', name='goal_input')
    x = concatenate([x, goal_input])

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(32, activation='relu')(x)
    out = Dense(img_rows*img_cols*1 + 1, activation='sigmoid')(x) #softmax

    model = Model(inputs = [img_input, goal_input], outputs = out)
    if weights_path:
        model.load_weights(weights_path)
        print('Loaded weights from ' + weights_path)

    optimizer = Adamax(lr=0.02, clipnorm=1e-6, clipvalue=1e6)
    model.compile(loss=keras.losses.categorical_crossentropy, #mse mae
                  optimizer=optimizer, # keras.optimizers.Adadelta()
                  metrics=[ 'accuracy'
                  # 'mae', abs_diff
                  ] )
    return model

def train_test_split(data1, data2, out, val_ratio=0.2):
    print 'out ', out, len(out)
    l = len(out)
    f = int((1 - val_ratio) * l)  # training values s
    train_indices = sample(range(l),f)
    test_indices = list(set(range(l)) - set(train_indices))
    return data1[train_indices], data2[train_indices], out[train_indices], \
           data1[test_indices], data2[test_indices], out[test_indices], 
           
if __name__ == '__main__': 

    f = h5py.File('data.h5', "r")

    imgs = np.asarray(f['img'], dtype='float32')
    angles = np.asarray(f['angle'], dtype='float32')
    goal_angles = np.asarray(f['goal_angle'], dtype='float32')
    # print goal_angles

    print imgs.shape, angles.shape, goal_angles.shape



    x_train, a_train, y_train, x_test, a_test, y_test = train_test_split(
                                                    imgs, goal_angles, angles, val_ratio = 0.2)

    # [x_train, a_train], [x_test, a_test], y_train, y_test = train_test_split(
    #                 imgs, goal_angles], angles, test_size=0.2, random_state=11)

    # the data, shuffled and split between train and test sets
    # (x_train, y_train), (x_test, y_test) = 

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        img_input_shape = (1, img_rows, img_cols)
        goal_input_shape = (1, )
    else:
        # We are here - channels last
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        img_input_shape = (img_rows, img_cols, 1)
        goal_input_shape = (1, )

    # Scale input
    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    a_train /= math.pi
    a_test  /= math.pi
    print('x_train shape:', x_train.shape, a_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    y_train /= math.pi
    y_test  /= math.pi
    y_train += 1
    y_test  += 1
    y_train *= 4
    y_test  *= 4

    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)


    # Create model
    model = create_model()

    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, verbose=1,
                                  patience=5, cooldown=10, min_lr=0.000001)
    # def abs_diff(y_true, y_pred):
    #     return K.abs(y_true - y_pred)

    model.fit([x_train, a_train], y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=([x_test, a_test], y_test),
              callbacks=[reduce_lr])
    score = model.evaluate([x_test, a_test], y_test, verbose=0)
    print('Test loss:', score)
    # print('Test accuracy:', score[1])
    print [model.predict([x_test, a_test], verbose=0)[:10], y_test[:10]]

    # Saving model
    print 'Saving model'
    model.save('model.h5')


