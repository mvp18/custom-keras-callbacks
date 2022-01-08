import os
import yaml
import random
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard

from model_cifar import build_model
from callbacks.checkpoint_callback import EarlyStop_ModelChkpt

with open('../configs/cifar10.yaml') as f:
	config = yaml.safe_load(f)

# set gpu number
os.environ["CUDA_VISIBLE_DEVICES"]=str(config["gpu"])

# set random seeds for reproducible runs
os.environ['PYTHONHASHSEED']=str(config["seed"])
np.random.seed(config["seed"])
random.seed(config["seed"])


def lr_scheduler(epoch):
	"""
	Returns a custom learning rate that decreases as epochs progress.
	"""
	learning_rate = config["learning_rate"]
	if epoch > 10:
		learning_rate = 0.005
	if epoch > 20:
		learning_rate = 0.001
	if epoch > 50:
		learning_rate = 0.0001

	tf.summary.scalar('learning rate', data=learning_rate, step=epoch)

	return learning_rate


# load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, config["num_classes"])
y_test = keras.utils.to_categorical(y_test, config["num_classes"])
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

model = build_model(config)

logdir = '../logs/cifar10/tensorboard/' + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir)
file_writer.set_as_default()
tboard = TensorBoard(log_dir=logdir, histogram_freq=0)

change_lr = LearningRateScheduler(lr_scheduler, verbose=1)

checkpoint = EarlyStop_ModelChkpt(monitor=config["monitor"], min_delta=config["min_delta"], patience=config["patience"], 
								 verbose=config["verbose"], mode=config["mode"])

cbks = [tboard, change_lr, checkpoint]

model.fit(x_train, y_train, batch_size=config["batch_size"], epochs=config["epochs"], validation_split=config["val_split"], 
		  callbacks=cbks, shuffle=True)

print("\nRestoring model weights from the end of the best epoch.")
model.set_weights(checkpoint.best_weights)
model_save_name = '{}-{:.4f}_epoch-{}'.format(checkpoint.monitor, checkpoint.best, checkpoint.best_epoch)
model.save('../weights/cifar10/' + model_save_name + '.h5')

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])