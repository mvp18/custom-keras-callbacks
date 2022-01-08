### Getting Started

```
conda create --name keras243 --file requirements.txt
conda activate keras243
cd custom-keras-callbacks/src
python3 [dataset].py; [dataset] = {mnist, cifar10, cifar100}
```

### Experiment logs

For visualizing Tensorboard plots

```
cd custom-keras-callbacks
tensorboard --logdir ./logs/[dataset]/tensorboard; [dataset] = {mnist, cifar10, cifar100}
```

For past run logs

```
cat ./logs/[dataset]/terminal/run.txt; [dataset] = {mnist, cifar10, cifar100}
```

### List of Callbacks per training script

- `src/mnist.py`: 
	- `CyclicLR` - taken from [this repo](https://github.com/bckenstler/CLR).
	- `Tensorboard`, `ModelCheckpoint`, `EarlyStopping` - basic usage of built-in Keras Callbacks
- `src/cifar10.py`: 
	- `LearningRateScheduler` (built-in) for epoch-based learning rate scheduling
	- `EarlyStop_ModelChkpt` - a custom callback combining the functionalities of `ModelCheckpoint` & `EarlyStopping`. Can also monitor custom metrics defined in other callbacks.
- `src/cifar100.py`: 
	- `SGDRScheduler` - modified from the implementation [here](https://www.jeremyjordan.me/nn-learning-rate/).
	- `clf_metrics` - a custom callback for calculating and monitoring global classification metrics like F1 score, precision etc. on validation set. As mentioned in this [Keras issue](https://github.com/keras-team/keras/issues/5794), these were previously part of built-in Keras metrics, but were removed in later versions due to misleading batch-wise calculations. 

### References

[1] https://www.tensorflow.org/guide/keras/custom_callback

[2] https://github.com/BIGBALLON/cifar-10-cnn

[3] https://keras.io/examples/vision/mnist_convnet

[4] https://www.jeremyjordan.me/nn-learning-rate/

[5] https://github.com/bckenstler/CLR

### More Resources

[1] [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)

[2] https://github.com/titu1994/keras-one-cycle

[3] https://github.com/davidtvs/pytorch-lr-finder

[4] https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html