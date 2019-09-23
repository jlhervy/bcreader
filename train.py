import os
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.python.keras import backend as K
import models
from tensorflow.keras.losses import categorical_crossentropy
from import_dataset import *
from parameters import *




model = models.make_model(input_shape)
model.summary()

def custom_loss(y_true, y_pred):
    K.mean(K.concatenate([categorical_crossentropy(y_true[k*nb_classes:k*nb_classes + 10], y_pred[k*nb_classes:k*nb_classes + 10]) for k in range(nb_digits)]))

adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00094,beta_1=0.864,beta_2=0.9996,epsilon=1e-07, amsgrad=False,)
model.compile(loss=custom_loss, optimizer=adam_optimizer,metrics=["accuracy"])


#Callbacks
def generate_unique_logpath(logdir, raw_run_name):
    i = 0
    while(True):
        run_name = raw_run_name + "-" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1

logpath = generate_unique_logpath("logs", "train")
if not os.path.exists(logpath):
    os.mkdir(logpath)
checkpoint_filepath = os.path.join(logpath,  "best_model.h5")

checkpoint_cb = ModelCheckpoint(checkpoint_filepath, save_best_only=True)

tensorboard_callback = TensorBoard(log_dir=logpath)

callbacks = [checkpoint_cb, tensorboard_callback]

history = model.fit(X, y,
			batch_size=batch_size,
			epochs=epochs,
			verbose=1,
            validation_split=0.1,
			callbacks=callbacks)


# inference_model = load_model(checkpoint_filepath)
# score = inference_model.evaluate(x_test, y_test, verbose=1)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])