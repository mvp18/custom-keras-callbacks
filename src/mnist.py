import os
import yaml
import random
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from model_mnist import build_model
from callbacks.lr_callbacks import CyclicLR

with open('../configs/mnist.yaml') as f:
	config = yaml.safe_load(f)

# set gpu number
os.environ["CUDA_VISIBLE_DEVICES"]=str(config["gpu"])

# set random seeds for reproducible runs
os.environ['PYTHONHASHSEED']=str(config["seed"])
np.random.seed(config["seed"])
random.seed(config["seed"])


# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, config["num_classes"])
y_test = keras.utils.to_categorical(y_test, config["num_classes"])

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

model = build_model(config)

logdir = '../logs/mnist/tensorboard/' + datetime.now().strftime("%Y%m%d-%H%M%S")
tboard = TensorBoard(log_dir=logdir, histogram_freq=0)

num_batches = x_train.shape[0]//config["batch_size"]
change_lr = CyclicLR(base_lr=config["min_lr"], max_lr=config["max_lr"], step_size=2*num_batches, mode='triangular')

checkpoint = ModelCheckpoint('../weights/mnist/best_model.h5', monitor=config["monitor"], save_best_only=True, verbose=config["verbose"], 
							 mode=config["mode"])

early_stop = EarlyStopping(monitor=config["monitor"], min_delta=config["min_delta"], patience=config["patience"], verbose=config["verbose"], 
						   mode=config["mode"])

cbks = [tboard, change_lr, checkpoint, early_stop]

model.fit(x_train, y_train, batch_size=config["batch_size"], epochs=config["epochs"], validation_split=config["val_split"], 
		  callbacks=cbks, shuffle=True)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])