import time

from collections import OrderedDict

import numpy as np
# np.random.seed(1234)
 
import ipdb

import theano
import theano.tensor as T

import lasagne
from theano.scalar.basic import UnaryScalarOp, same_out_nocomplex
from theano.tensor.elemwise import Elemwise
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams



# This class extends the Lasagne DenseLayer to support LAB
class DenseLayer(lasagne.layers.DenseLayer):
	def __init__(self, mask, weight, incoming, num_units,  **kwargs):
		self.mask = mask
		self.fixed = weight	

		super(DenseLayer, self).__init__(incoming, num_units, **kwargs)


	def get_output_for(self, input, deterministic=False, **kwargs):
		# import ipdb; ipdb.set_trace()
		self.Ws = self.W *self.mask
		Wr = self.W
		self.W = self.Ws + self.fixed
			
		rvalue = super(DenseLayer, self).get_output_for(input, **kwargs)
		
		self.W = Wr
		
		return rvalue

# This class extends the Lasagne Conv2DLayer to support LAB
class Conv2DLayer(lasagne.layers.Conv2DLayer):
	
	def __init__(self, mask, weight, incoming, num_filters, filter_size, **kwargs):
		
		self.mask = mask
		self.fixed = weight
		
		super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size, **kwargs)    


	def get_output_for(self, input, deterministic=False, **kwargs):
		
		self.Ws = self.W *self.mask
		Wr = self.W
		self.W = self.Ws + self.fixed
		rvalue = super(Conv2DLayer, self).get_output_for(input, **kwargs)		
		self.W = Wr
		
		return rvalue


def compute_grads1(loss,network, mask):
		
	layers = lasagne.layers.get_all_layers(network)
	grads = []
	i=0
	for layer in layers:	
		# import ipdb; ipdb.set_trace()
		params = layer.get_params()
		if params:
			grads.append((mask[i+i]*theano.grad(loss, wrt=layer.Ws)).astype(theano.config.floatX))
			grads.append((mask[i+i+1]*theano.grad(loss, wrt=layer.b)).astype(theano.config.floatX))
			i = i+1	
	return grads	