import math
import numpy as np
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt


"""Borrowed and modified from https://github.com/surmenok/keras_lr_finder/blob/master/keras_lr_finder/lr_finder.py"""


class LRFinder:
    """
    Plots the change of the loss function of a Keras model when the learning rate is exponentially increasing.
    See for details:
    https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0
    """

    def __init__(self, model, stopFactor=4, beta=0.98):
        
        self.model = model
        self.stopFactor = stopFactor
        self.beta = beta
    
    
    def reset(self):
        
        self.losses = []
        self.lrs = []
        self.best_loss = 1e9
        self.batchNum = 0
        self.avgLoss = 0
        
    
    def on_batch_end(self, batch, logs):
        
        self.batchNum += 1
        # grab the current learning rate and add log it to the list of
        # learning rates that we've tried
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)

        # grab the loss at the end of this batch, increment the total number of batches processed, compute the average average loss, smooth it, 
        # and update the losses list with the smoothed value
        loss = logs["loss"]
        self.avgLoss = self.beta*self.avgLoss + (1 - self.beta)*loss
        smooth_loss = self.avgLoss/(1 - (self.beta**self.batchNum))
        self.losses.append(smooth_loss)

        stopLoss = self.stopFactor*self.best_loss
        
        # Check whether the loss got too large or NaN
        if self.batchNum > 1 and (smooth_loss > stopLoss or math.isnan(smooth_loss)):
            # stop returning and return from the method
            self.model.stop_training = True
            return

        # check to see if the best loss should be updated
        if self.batchNum == 1 or smooth_loss < self.best_loss:
            self.best_loss = smooth_loss

        # Increase the learning rate for the next batch
        lr *= self.lr_mult
        K.set_value(self.model.optimizer.lr, lr)
        logs["lr"] = lr
    
    
    def find(self, x_train, y_train, start_lr, end_lr, batch_size=64, epochs=1):
        # If x_train contains data for multiple inputs, use length of the first input.
        # Assumption: the first element in the list is single input; NOT a list of inputs.
        N = x_train[0].shape[0] if isinstance(x_train, list) else x_train.shape[0]

        # Compute number of batches and LR multiplier
        num_batches = epochs * N / batch_size
        self.lr_mult = (float(end_lr) / float(start_lr)) ** (float(1) / float(num_batches))

        # Remember the original learning rate
        original_lr = K.get_value(self.model.optimizer.lr)

        # Set the initial learning rate
        K.set_value(self.model.optimizer.lr, start_lr)

        callback = LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))

        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[callback])

        # Restore the original learning rate
        K.set_value(self.model.optimizer.lr, original_lr)

    
    def find_generator(self, generator, start_lr, end_lr, epochs=1, steps_per_epoch=None, **kw_fit):
        
        self.reset()
        
        if steps_per_epoch is None:
            try:
                steps_per_epoch = len(generator)
            except (ValueError, NotImplementedError) as e:
                raise e('`steps_per_epoch=None` is only valid for a generator based on the `keras.utils.Sequence` class')
                
        self.start_lr = start_lr
        self.end_lr = end_lr

        # compute the total number of batch updates for finding a good starting learning rate
        self.total_iterations = epochs*steps_per_epoch
        self.lr_mult = (end_lr / start_lr)**(1.0 / self.total_iterations)

        # Save weights into a file
#         self.model.save_weights('tmp.h5')

        # Remember the original learning rate
        original_lr = K.get_value(self.model.optimizer.lr)

        # Set the initial learning rate
        K.set_value(self.model.optimizer.lr, start_lr)

        callback = LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))

        self.model.fit(x=generator, validation_data=None, epochs=epochs, callbacks=[callback], workers=24, **kw_fit)

        # Restore the weights to the state before model fitting
#         self.model.load_weights('tmp.h5')

        # Restore the original learning rate
        K.set_value(self.model.optimizer.lr, original_lr)

    
    def plot_loss(self, save_dir, suffix, n_skip_beginning=10, n_skip_end=5, x_scale='log'):
        """
        Plots the loss.
        Parameters:
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
        """
        plt.ylabel("loss")
        plt.xlabel("Learning rate (log scale)")
        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], self.losses[n_skip_beginning:-n_skip_end])
        plt.xscale(x_scale)
        xticks = np.logspace(np.log10(self.start_lr), np.log10(self.end_lr), num=int(np.log10(self.end_lr)-np.log10(self.start_lr)+1), base=10)
        plt.xticks(xticks)
        for xc in xticks:
            plt.axvline(x=xc, color='r', linestyle='--')
#         plt.show()
        plt.savefig(save_dir+'loss_vs_lr'+suffix+'.jpg')
        
    
    def plot_lr(self, yscale='log'):
        '''Helper function to quickly inspect the learning rate schedule.'''
        plt.plot(np.arange(self.total_iterations), self.lrs)
        plt.yscale(yscale)
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate (log scale)')
        plt.show()

    
    def plot_loss_change(self, save_dir, suffix, sma=1, n_skip_beginning=10, n_skip_end=5, y_lim=(-0.01, 0.01)):
        """
        Plots rate of change of the loss function.
        Parameters:
            sma - number of batches for simple moving average to smooth out the curve.
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
            y_lim - limits for the y axis.
        """
        derivatives = self.get_derivatives(sma)[n_skip_beginning:-n_skip_end]
        lrs = self.lrs[n_skip_beginning:-n_skip_end]
        plt.ylabel("rate of loss change")
        plt.xlabel("learning rate (log scale)")
        plt.plot(lrs, derivatives)
        plt.xscale('log')
        xticks = np.logspace(np.log10(self.start_lr), np.log10(self.end_lr), num=int(np.log10(self.end_lr)-np.log10(self.start_lr)+1), base=10)
        plt.xticks(xticks)
        for xc in xticks:
            plt.axvline(x=xc, color='r', linestyle='--')
        plt.ylim(y_lim)
#         plt.show()
        plt.savefig(save_dir+'losschange_vs_lr'+suffix+'.jpg')

    
    def get_derivatives(self, sma):
        
        assert sma >= 1
        derivatives = [0] * sma
        
        for i in range(sma, len(self.lrs)):
            derivatives.append((self.losses[i] - self.losses[i - sma]) / sma)
        
        return derivatives

    
    def get_best_lr(self, sma, n_skip_beginning=10, n_skip_end=5):
        
        derivatives = self.get_derivatives(sma)
        best_der_idx = np.argmin(derivatives[n_skip_beginning:-n_skip_end])
        
        return self.lrs[n_skip_beginning:-n_skip_end][best_der_idx]