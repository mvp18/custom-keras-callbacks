import os
import yaml
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.callbacks import TensorBoard

from model_cifar import build_model
from callbacks.checkpoint_callback import EarlyStop_ModelChkpt
from callbacks.lr_callbacks import SGDRScheduler
from callbacks.metric_callback import clf_metrics


with open('../configs/cifar100.yaml') as f:
	config = yaml.safe_load(f)

# set gpu number
os.environ["CUDA_VISIBLE_DEVICES"]=str(config["gpu"])

# set random seeds for reproducible runs
os.environ['PYTHONHASHSEED']=str(config["seed"])
np.random.seed(config["seed"])
random.seed(config["seed"])

# load data
(x_trval, y_trval), (x_test, y_test) = cifar100.load_data()

va_ind = np.random.choice(x_trval.shape[0], int(config["val_split"]*x_trval.shape[0]), replace=False)
tr_ind = np.setdiff1d(np.arange(x_trval.shape[0]), va_ind)

x_train = x_trval[tr_ind]
x_val = x_trval[va_ind]
y_train = y_trval[tr_ind]
y_val = y_trval[va_ind]

y_train = keras.utils.to_categorical(y_train, config["num_classes"])
y_val = keras.utils.to_categorical(y_val, config["num_classes"])
y_test = keras.utils.to_categorical(y_test, config["num_classes"])

x_train = x_train.astype('float32')/255.0
x_val = x_val.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

print('Train:{}; Val:{}; Test:{}.'.format(x_train.shape[0], x_val.shape[0], x_test.shape[0]))

model = build_model(config)

tb_cb = TensorBoard(log_dir='../logs/cifar100/tensorboard', histogram_freq=0)
num_batches = x_train.shape[0]//config["batch_size"]
change_lr = SGDRScheduler(min_lr=config["min_lr"], max_lr=config["max_lr"], steps_per_epoch=num_batches, lr_decay=0.9, cycle_length=5, mult_factor=1.2)
metrics = clf_metrics(x_val, y_val)
checkpoint = EarlyStop_ModelChkpt(monitor=config["monitor"], min_delta=config["min_delta"], patience=config["patience"], 
								  verbose=config["verbose"], mode=config["mode"])
cbks = [tb_cb, change_lr, metrics, checkpoint]

model.fit(x_train, y_train, batch_size=config["batch_size"], epochs=config["epochs"], validation_data=(x_val, y_val), 
		  callbacks=cbks, shuffle=True)

print("\nRestoring model weights from the end of the best epoch.")
model.set_weights(checkpoint.best_weights)
model_save_name = '{}-{:.4f}_epoch-{}'.format(checkpoint.monitor, checkpoint.best, checkpoint.best_epoch)
model.save('../weights/cifar100/' + model_save_name + '.h5')

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])