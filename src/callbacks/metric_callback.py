import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import numpy as np

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


class clf_metrics(Callback):

	def __init__(self, xVal, yVal):
		super(clf_metrics, self).__init__()

		self.X = xVal
		self.y = yVal

	
	def on_epoch_end(self, epoch, logs=None):

		# X, y = self.validation_data[0], self.validation_data[1] #doesn't work in latest version of Keras. self.validation_data is None inside clbk.

		y_true = np.argmax(self.y, axis=1)
		y_prob = self.model.predict(self.X)
		y_pred = np.argmax(y_prob, axis=1)

		prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
		roc_auc = roc_auc_score(y_true, y_prob, average='macro', multi_class='ovr')

		# Adding them to the logs dict ensures they can be monitored in ModelCheckpoint or EarlyStopping callbacks
		logs["val_f1"] = f1
		logs["val_precision"] = prec
		logs["val_recall"] = rec
		logs["val_roc_auc"] = roc_auc

		# Registering as Tensorboard scalars for visualization with loss and accuracy 
		tf.summary.scalar('val_f1', data=f1, step=epoch)
		tf.summary.scalar('val_precision', data=prec, step=epoch)
		tf.summary.scalar('val_recall', data=rec, step=epoch)
		tf.summary.scalar('val_roc_auc', data=roc_auc, step=epoch)