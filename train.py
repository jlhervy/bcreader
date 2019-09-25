import tensorflow as tf
from tensorflow.keras import backend as K
import models
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
from import_dataset import *
from parameters import *
from callbacks import *



# mirrored_strategy = tf.distribute.MirroredStrategy()
# with mirrored_strategy.scope():

with tf.device('/gpu:1'):
	model = models.make_model(input_shape)
	model.summary()


	adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00094,beta_1=0.864,beta_2=0.9996,epsilon=1e-07, amsgrad=False,)
	model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer,metrics=['accuracy'])

	train_dataset = train_dataset.batch(batch_size)
	history = model.fit(train_dataset,
			epochs=epochs,
			verbose=0,
			callbacks=callbacks)


# inference_model = load_model(checkpoint_filepath)
# score = inference_model.evaluate(x_test, y_test, verbose=1)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])