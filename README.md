### Callback examples per training script

- `mnist.py`: Cyclic LR, Tensorboard, ModelCheckpoint, EarlyStopping (all pre-built)
- `cifar10.py`: Simple LR scheduler, custom early stopping and model checkpoint callback (similar API and usage as official Keras callback)
- `cifar100.py`: SGDRScheduler, metric callback for calculating and monitoring F1, precision etc. on validation set.

### References

[1] https://www.tensorflow.org/guide/keras/custom_callback
[2] https://www.jeremyjordan.me/nn-learning-rate/
[3] https://github.com/bckenstler/CLR
[4] https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
[5] https://www.tensorflow.org/guide/keras/custom_callback
[6] https://github.com/BIGBALLON/cifar-10-cnn
[7] https://keras.io/examples/vision/mnist_convnet
[8] https://github.com/rohanchopra/cifar10

### More Resources

[1] [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)
[2] https://github.com/titu1994/keras-one-cycle
[3] https://github.com/davidtvs/pytorch-lr-finder