import tensorflow as tf
import tensorflow.keras.layers as ly


def _conv_block(filters, kernel_size, x) :
    x = ly.Conv2D(filters, kernel_size,
                  kernel_constraint=tf.keras.constraints.max_norm(max_value=3.36))(x)
    x = ly.SpatialDropout2D(0.25)(x)
    x = ly.BatchNormalization(axis=1)(x)
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
    x = ly.Dense(units=4096)(x)
    x = ly.Dropout(0.25)(x)
    x = ly.Activation("relu")(x)
    x = ly.Dense(units=4096)(x)
    x = ly.Activation("relu")(x)
    x = ly.Dense(units= 100)(x)
    #x = ly.Reshape((10, 10))(x)
    outputs = ly.Activation("softmax")(x)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model