from __future__ import print_function

import sys
import os
import time

import numpy as np
# np.random.seed(1234) 
import theano
import theano.tensor as T
import lasagne

import cPickle as pickle
import gzip

import ipdb
import lab

from pylearn2.datasets.zca_dataset import ZCA_Dataset    
from pylearn2.utils import serial

from collections import OrderedDict
from argparse import ArgumentParser


######################################### load dataset ###########################################
def load_dataset():
	train_set_size = 45000
	print("train_set_size = "+str(train_set_size))
	
	print('Loading CIFAR-10 dataset...')
	
	preprocessor = serial.load("/home/eric/Desktop/powerLaw/power-law/expt3/cnn/cifar10/pylearn2_gcn_whitened/preprocessor.pkl")
	train_set = ZCA_Dataset(
		preprocessed_dataset=serial.load("/home/eric/Desktop/powerLaw/power-law/expt3/cnn/cifar10/pylearn2_gcn_whitened/train.pkl"),
		preprocessor = preprocessor,
		start=0, stop = train_set_size)
	valid_set = ZCA_Dataset(
		preprocessed_dataset= serial.load("/home/eric/Desktop/powerLaw/power-law/expt3/cnn/cifar10/pylearn2_gcn_whitened/train.pkl"),
		preprocessor = preprocessor,
		start=45000, stop = 50000)  
	test_set = ZCA_Dataset(
		preprocessed_dataset= serial.load("/home/eric/Desktop/powerLaw/power-law/expt3/cnn/cifar10/pylearn2_gcn_whitened/test.pkl"),
		preprocessor = preprocessor)
	# bc01 format
	X_train = train_set.X.reshape(-1,3,32,32)
	X_val   = valid_set.X.reshape(-1,3,32,32)
	X_test  = test_set.X.reshape(-1,3,32,32)
	
	# flatten targets
	y_train = train_set.y.flatten()
	y_val   = valid_set.y.flatten()
	y_test  = test_set.y.flatten()
	
	return X_train, y_train, X_val, y_val, X_test, y_test


def build_sparse_cnn(mask, input_var=None):
	# Input layer, as usual:
	### one may also add dropout layer according to the icml 2017 paper
	network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32), input_var=input_var)
	network = lab.Conv2DLayer(mask[0], network, num_filters=32, filter_size=(3, 3), pad=1)
	network = lab.Conv2DLayer(mask[2], network, num_filters=32, filter_size=(3, 3),pad=1)
	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
	# network = lasagne.layers.DropoutLayer(network, p=0.25)
	network = lab.Conv2DLayer(mask[4], network, num_filters=64, filter_size=(3, 3), pad=1)			
	network = lab.Conv2DLayer(mask[6], network, num_filters=64, filter_size=(3, 3), pad=1)
	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2)) 
	# network = lasagne.layers.DropoutLayer(network, p=0.25)	                      
	network = lab.DenseLayer(mask[8], network, num_units=512, nonlinearity=lasagne.nonlinearities.rectify)  
	# network = lasagne.layers.DropoutLayer(network, p=0.5)    
	network = lab.DenseLayer(mask[10], network, num_units=10, nonlinearity=lasagne.nonlinearities.softmax)      

	return network

############################### Batch iterator ###############################
def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
	assert len(inputs) == len(targets)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batchsize]
		else:
			excerpt = slice(start_idx, start_idx + batchsize)
		yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################

def main(prune_fraction, num_epochs=500):
	# Load the dataset
	print("Loading data...")
	X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
	# Prepare Theano variables for inputs and targets
	input_var = T.tensor4('inputs')
	# input_var = T.fmatrix('inputs')
	target_var = T.ivector('targets')


	with np.load('cifar/model/dense_cnn.npz') as f:
	    param_values = [f['arr_%d' % i] for i in range(len(f.files))]

############################### thresholding the weights ##########################

	prune_fraction = prune_fraction
	thres = []
	mask = []
	for i in range(len(param_values)):
		data_current = np.abs(param_values[i])
		if len(param_values[i].shape)>1:
			vec_data = data_current.flatten()
			a = int(prune_fraction*data_current.size)
			thres_current = np.sort(vec_data)[a]
		else:
			thres_current = np.float32(0.0)

		mask_current = (data_current>thres_current).astype('int16')  
		### int32 multiplied with flot32 gives float32, float32 multiplied with int32 gives float64
		param_values[i] *= mask_current 

		thres.append(thres_current)
		mask.append(mask_current)

	print(thres)
################################### rebuild sparse model ############################
	network_s = build_sparse_cnn(mask, input_var)	

	# Create a loss expression for training
	prediction = lasagne.layers.get_output(network_s)
	loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
	loss = loss.mean()

	# Create update expressions for training
	params = lasagne.layers.get_all_params(network_s, trainable=True)
	# import ipdb; ipdb.set_trace()
	W_grads = lab.compute_grads1(loss, network_s, mask)
	# updates = lasagne.updates.nesterov_momentum(
			# loss_or_grads=W_grads, params=params, learning_rate=0.01, momentum=0.9)
	updates = lasagne.updates.rmsprop(W_grads, params, learning_rate=0.0001)

	test_prediction = lasagne.layers.get_output(network_s, deterministic=True)
	test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
															target_var)
	test_loss = test_loss.mean()
	test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
					  dtype=theano.config.floatX)


	train_fn = theano.function([input_var, target_var], [loss, W_grads[0], W_grads[2], W_grads[4]], updates=updates)
	val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

	lasagne.layers.set_all_param_values(network_s, param_values)
############################### retraining ########################################


	print("Starting retraining...")
	best_val_acc = 0
	best_epoch = 0
	# We iterate over epochs:
	for epoch in range(num_epochs):
		# In each epoch, we do a full pass over the training data:
		train_loss = 0
		train_batches = 0
		start_time = time.time()
		for batch in iterate_minibatches(X_train, y_train, 50, shuffle=True):
			inputs, targets = batch
			tmp, g1, g2, g3 = train_fn(inputs, targets)
			# import ipdb; ipdb.set_trace()
			train_loss += tmp
			train_batches += 1

		train_loss = train_loss / train_batches

		# And a full pass over the validation data:
		val_loss = 0
		val_acc = 0
		val_batches = 0
		for batch in iterate_minibatches(X_val, y_val, 50, shuffle=False):
			inputs, targets = batch
			loss, acc = val_fn(inputs, targets)
			val_loss += loss
			val_acc += acc
			val_batches += 1

		val_loss = val_loss / val_batches
		val_acc = val_acc / val_batches * 100
		# Then we print the results for this epoch:

		if val_acc > best_val_acc:
			best_val_acc = val_acc
			best_epoch = epoch			
			# After training, we compute and print the test error:
			test_loss = 0
			test_acc = 0
			test_batches = 0
			for batch in iterate_minibatches(X_test, y_test, 50, shuffle=False):
				inputs, targets = batch
				loss, acc = val_fn(inputs, targets)
				test_loss += loss
				test_acc += acc
				test_batches += 1
			test_loss = test_loss / test_batches
			test_acc = test_acc / test_batches * 100

			np.savez('cifar/model/sparse_cnn_{0}.npz'.format(prune_fraction), *lasagne.layers.get_all_param_values(network_s))

		print("Epoch {} of {} took {:.3f}s".format(
			epoch + 1, num_epochs, time.time() - start_time))
		print("  training loss:\t\t{:.6f}".format(train_loss))
		print("  validation loss:\t\t{:.6f}".format(val_loss))
		print("  validation accuracy:\t\t{:.2f} %".format(val_acc))
		print("  test loss:\t\t\t{:.6f}".format(test_loss))
		print("  test accuracy:\t\t{:.2f} %".format(test_acc))
		
		with open("cifar/model/sparse_cnn_{0}.txt".format(prune_fraction), "a") as myfile:
			myfile.write("{0}  {1:.3f} {2:.3f} {3:.3f} {4:.3f} {5:.3f}\n".format(epoch, train_loss, val_loss, val_acc, test_loss, test_acc))



if __name__ == "__main__":
	parser = ArgumentParser()
	# parser.add_argument("--model", type=str, dest="model",
	# 			default='cnn')	
	parser.add_argument("--num_epochs",  type=int, dest="num_epochs",
				default=100, help="number of epochs")  
	parser.add_argument("--prune_fraction",  type=float, dest="prune_fraction",
				default=0.7, help="fraction of weights pruned away") 
	args = parser.parse_args()

	main(**vars(args))
