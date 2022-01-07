import os
import yaml
import random
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


def scheduler(epoch):
    if epoch < 100:
        return 0.01
    if epoch < 150:
        return 0.005
    return 0.001


# load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, config["num_classes"])
y_test = keras.utils.to_categorical(y_test, config["num_classes"])
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

model = build_model(config)

tb_cb = TensorBoard(log_dir='../logs/cifar10/tensorboard', histogram_freq=0)
change_lr = LearningRateScheduler(scheduler, verbose=1)
checkpoint = EarlyStop_ModelChkpt(monitor=config["monitor"], min_delta=config["min_delta"], patience=config["patience"], 
								 verbose=config["verbose"], mode=config["mode"])
cbks = [change_lr, tb_cb, checkpoint]

model.fit(x_train, y_train, batch_size=config["batch_size"], epochs=config["epochs"], validation_split=config["val_split"], 
		  callbacks=cbks, shuffle=True)

print("\nRestoring model weights from the end of the best epoch.")
model.set_weights(checkpoint.best_weights)
model_save_name = '{}-{}_epoch-{}'.format(checkpoint.monitor, checkpoint.best, checkpoint.best_epoch)
model.save('../weights/cifar10/' + model_save_name + '.h5')

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])