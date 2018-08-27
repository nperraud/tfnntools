import tensorflow as tf
from tensorflow.contrib.rnn import ConvLSTMCell
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import MultiRNNCell
from numpy import prod
from tfblocks import *



def rprint(msg, reuse=False):
    """Print message only if reuse is False.
    If a block is being resued, its description will not be re-printed.
    """
    if not reuse:
        print(msg)

class BaseNet(object):
    """Base Net abstract class."""
    def __init__(self, params, name="BaseNet"):
        self._params = params
        self._name = name
        self._outputs = None
        self._inputs = None
        self._loss = None
        self._build_net()
        self._add_summary()

    def _build_net(self, inputs):
        raise ValueError('This is an abstract class')
        
    def batch2dict(self, inputs):
        raise ValueError('This is an abstract class')

    def _add_summary(self):
        tf.summary.scalar('train/loss', self._loss,  collections=["train"])
        
    @property
    def name(self):
        return self._name

    @property
    def loss(self):
        return self._loss

    @property
    def outputs(self):
        return self._outputs
    
    @property
    def inputs(self):
        return self._inputs