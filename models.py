import tensorflow as tf
import tensorflow.keras.layers as ly
from tensorflow.keras import backend as K
from parameters import *


def _conv_block(filters, kernel_size, x):
    x = ly.Conv2D(filters, kernel_size,
                  kernel_constraint=tf.keras.constraints.max_norm(max_value=3.36))(x)
    x = ly.SpatialDropout2D(0.25)(x)
    x = ly.BatchNormalization(axis=-1)(x)
    x = ly.LeakyReLU(alpha=0.3)(x)
    x = ly.MaxPooling2D()(x)
    return x

def _custom_softmax(x) :
    return [tf.nn.softmax(x[k]) for k in range(10) ]


def make_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = _conv_block(32, 5, inputs)
    x = _conv_block(64, 3, x)
    x = _conv_block(128, 3, x)
    x = _conv_block(256, 3, x)
    x = ly.Flatten()(x)
    x = ly.Dense(units=4096)(x)
    x = ly.Dropout(0.25)(x)
    x = ly.LeakyReLU(alpha=0.3)(x)
    x = ly.Dense(units=4096)(x)
    x = ly.LeakyReLU(alpha=0.3)(x)
    x = ly.Dense(units= nb_classes*nb_digits)(x)
    x = ly.Lambda(lambda x: tf.split(x, num_or_size_splits=nb_digits, axis=1))(x)
    x = ly.Lambda(_custom_softmax)(x)
    outputs = ly.Concatenate()(x)
    #x = ly.Softmax(axis=-1)(x)
    #outputs = ly.Reshape((10, 10))(x)
    model = tf.keras.Model(inputs=[inputs], outputs=outputs)

    #loss = K.mean([categorical_crossentropy(labels[k*nb_classes:k*nb_classes + 10], x[k*nb_classes:k*nb_classes + 10]) for k in range(nb_digits)])
    #model.add_loss(loss)
    return model


def make_model2(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = _conv_block(32, 5, inputs)
    x = _conv_block(64, 3, x)
    x = _conv_block(128, 3, x)
    x = _conv_block(256, 3, x)
    x = ly.Flatten()(x)
    x = ly.Dense(units=4096)(x)
    x = ly.Dropout(0.25)(x)
    x = ly.Activation("relu")(x)
    x = ly.Dense(units=4096)(x)
    x = ly.Activation("relu")(x)
    d0 = ly.Dense(units= 10, activation="softmax")(x)
    d1 = ly.Dense(units= 10, activation="softmax")(x)
    d2 = ly.Dense(units= 10, activation="softmax")(x)
    d3 = ly.Dense(units= 10, activation="softmax")(x)
    d4 = ly.Dense(units= 10, activation="softmax")(x)
    d5 = ly.Dense(units= 10, activation="softmax")(x)
    d6 = ly.Dense(units= 10, activation="softmax")(x)
    d7 = ly.Dense(units= 10, activation="softmax")(x)
    d8 = ly.Dense(units= 10, activation="softmax")(x)
    d9 = ly.Dense(units= 10, activation="softmax")(x)

    model = tf.keras.Model(inputs=[inputs], outputs=[d0, d1, d2, d3, d4, d5, d6, d7, d8, d9])
    return model