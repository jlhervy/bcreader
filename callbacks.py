from parameters import *
import os
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


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


file_writer = tf.summary.create_file_writer(logpath + "/metrics")
file_writer.set_as_default()

class CustomCallback(tf.keras.callbacks.Callback) :

    # def on_train_batch_end(self, batch, logs=None):
    #     accuracy = 0
    #     for k in range(nb_digits):
    #         key = "lambda_1_accuracy" if k==0 else "lambda_1_" + str(k) + "_accuracy"
    #         accuracy = accuracy + logs[key]
    #     accuracy = accuracy / nb_digits
    #
    #
    #     print('For batch {}, accuracy is '.format(batch) + str(accuracy))

    def on_epoch_end(self, epoch, logs=None):
        train_acc = 0
        for k in range(nb_digits):
            key = "lambda_1_accuracy" if k == 0 else "lambda_1_" + str(k) + "_accuracy"
            train_acc = train_acc + logs[key]
        train_acc = train_acc / nb_digits

        val_acc = 0
        for k in range(nb_digits):
            key = "val_lambda_1_accuracy" if k == 0 else "val_lambda_1_" + str(k) + "_accuracy"
            val_acc = val_acc + logs[key]
        val_acc = val_acc / nb_digits
        tf.summary.scalar('Average training digit accuracy (random guess is ' + str(1/nb_digits) + ')', data=train_acc, step=epoch)
        tf.summary.scalar('Average validation digit accuracy (random guess is ' + str(1 / nb_digits) + ')',
                          data=val_acc, step=epoch)
        print('For epoch {}, the average training accuracy is'.format(epoch) + str(train_acc) + " and the average validation accuracy is "+ str(val_acc))


checkpoint_filepath = os.path.join(logpath,  "best_model.h5")

checkpoint_cb = ModelCheckpoint(checkpoint_filepath, save_best_only=True)

custom_cb = CustomCallback()
tensorboard_callback = TensorBoard(log_dir=logpath)

earlystop_cb = EarlyStopping(monitor='val_loss',patience=10)
callbacks = [checkpoint_cb, earlystop_cb, custom_cb]