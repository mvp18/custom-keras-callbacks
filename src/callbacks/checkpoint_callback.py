from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import numpy as np


class EarlyStop_ModelChkpt(Callback):

    def __init__(self, monitor='acc', min_delta=0, patience=5, verbose=0, mode='max'):
        super(EarlyStop_ModelChkpt, self).__init__()
        
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode

        self.monitor_cond = np.less if self.mode=="min" else np.greater
        if self.mode=="min":
            self.min_delta *= -1
            
    
    def on_train_begin(self, logs=None):        
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # The epoch with the best performance
        self.best_epoch = 0
        # Weights at best_epoch
        self.best_weights = None
        # Initialize the best as infinity or 0.
        self.best = np.inf if self.mode=="min" else -np.inf
    
    
    def on_epoch_end(self, epoch, logs=None):
        
        current = logs.get(self.monitor)
        
        if self.monitor_cond(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch + 1
        else:
            self.wait += 1
            if self.verbose > 0:
                print("\n{} did not improve from {:.4f}.".format(self.monitor, self.best))
            if self.wait >= self.patience:
                self.stopped_epoch = epoch + 1
                if self.verbose > 0:
                    print("Epoch {:03d}: early stopping.".format(self.stopped_epoch))
                self.model.stop_training = True
    
    
    def on_train_end(self, logs=None):
        
        print("\nBest epoch: {} with {}={:.4f}.".format(self.best_epoch, self.monitor, self.best))