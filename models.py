import tensorflow as tf
import tensorflow.keras.layers as ly


def _conv_block(filters, kernel_size, x):
    x = ly.Conv2D(filters, kernel_size,
                  kernel_constraint=tf.keras.constraints.max_norm(max_value=3.36))(x)
    x = ly.SpatialDropout2D(0.25)(x)
    x = ly.BatchNormalization(axis=-1)(x)
    x = ly.Activation("relu")(x)
    x = ly.MaxPooling2D()(x)
    return x

def make_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = _conv_block(32, 5, inputs)
    x = _conv_block(64, 3, x)
    x = _conv_block(128, 3, x)
    x = _conv_block(256, 3, x)
    x = ly.Flatten()(x)
    x = ly.Dense(units=256)(x)
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