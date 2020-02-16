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


from pylearn2.datasets.zca_dataset import ZCA_Dataset    
from pylearn2.utils import serial

from collections import OrderedDict
from argparse import ArgumentParser

def load_dataset():
	train_set_size = 45000
	print("train_set_size = "+str(train_set_size))
	
	print('Loading CIFAR-10 dataset...')
	
	preprocessor = serial.load("${PYLEARN2_DATA_PATH}/cifar10/pylearn2_gcn_whitened/preprocessor.pkl")
	train_set = ZCA_Dataset(
		preprocessed_dataset=serial.load("${PYLEARN2_DATA_PATH}/cifar10/pylearn2_gcn_whitened/train.pkl"), 
		preprocessor = preprocessor,
		start=0, stop = train_set_size)
	valid_set = ZCA_Dataset(
		preprocessed_dataset= serial.load("${PYLEARN2_DATA_PATH}/cifar10/pylearn2_gcn_whitened/train.pkl"), 
		preprocessor = preprocessor,
		start=45000, stop = 50000)  
	test_set = ZCA_Dataset(
		preprocessed_dataset= serial.load("${PYLEARN2_DATA_PATH}/cifar10/pylearn2_gcn_whitened/test.pkl"), 
		preprocessor = preprocessor)
	# bc01 format
	X_train = train_set.X.reshape(-1,3,32,32)
	X_val   = valid_set.X.reshape(-1,3,32,32)
	X_test  = test_set.X.reshape(-1,3,32,32)
	
	# flatten targets
	y_train = train_set.y.flatten()
	y_val   = valid_set.y.flatten()
	y_test  = test_set.y.flatten()
	# y_train = np.hstack(train_set.y)
	# y_val = np.hstack(valid_set.y)
	# y_test = np.hstack(test_set.y)

   
	# # Onehot the targets
	# y_train = np.float32(np.eye(10)[y_train])    
	# y_val  = np.float32(np.eye(10)[y_val])
	# y_test = np.float32(np.eye(10)[y_test])
	
	# # for hinge loss
	# y_train = 2* y_train - 1.
	# y_val = 2* y_val - 1.
	# y_test = 2* y_test - 1.
	
	return X_train, y_train, X_val, y_val, X_test, y_test




############################### Build Network ###############################
def build_cnn(input_var=None):
	# Input layer, as usual:
	### one may also add dropout layer according to the icml 2017 paper
	l_in   = lasagne.layers.InputLayer(shape=(None, 3, 32, 32), input_var=input_var)
	l_cnn1 = lasagne.layers.Conv2DLayer(l_in, num_filters=32, filter_size=(3, 3), pad=1)
	l_cnn2 = lasagne.layers.Conv2DLayer(l_cnn1, num_filters=32, filter_size=(3, 3),pad=1)
	l_mp1  = lasagne.layers.MaxPool2DLayer(l_cnn2, pool_size=(2, 2))
	l_dr1  = lasagne.layers.DropoutLayer(l_mp1, p=0.25)
	l_cnn3 = lasagne.layers.Conv2DLayer(l_dr1, num_filters=64, filter_size=(3, 3), pad=1)			
	l_cnn4 = lasagne.layers.Conv2DLayer(l_cnn3, num_filters=64, filter_size=(3, 3), pad=1)
	l_mp2  = lasagne.layers.MaxPool2DLayer(l_cnn4, pool_size=(2, 2)) 
	l_dr2  = lasagne.layers.DropoutLayer(l_mp2, p=0.25)	                      
	l_dn1  = lasagne.layers.DenseLayer(l_dr2, num_units=512, nonlinearity=lasagne.nonlinearities.rectify)  
	l_dr3  = lasagne.layers.DropoutLayer(l_dn1, p=0.5)    
	l_dn2 =  lasagne.layers.DenseLayer(l_dr3, num_units=10, nonlinearity=lasagne.nonlinearities.softmax)      

	return l_dn2

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


def main(num_epochs=200):
	# Load the dataset
	print("Loading data...")
	X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

	# Prepare Theano variables for inputs and targets
	input_var = T.tensor4('inputs')
	target_var = T.ivector('targets')

############################################ train dense model #############################################
	network = build_cnn(input_var)	
	# Create a loss expression for training
	prediction = lasagne.layers.get_output(network)
	loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
	loss = loss.mean()

	# Create update expressions for training
	params = lasagne.layers.get_all_params(network, trainable=True)
	# updates = lasagne.updates.adam(loss, params, learning_rate=1e-3)
	# updates = lasagne.updates.nesterov_momentum(
			# loss, params, learning_rate=0.01, momentum=0.9)
	updates = lasagne.updates.rmsprop(loss, params, learning_rate=0.0001)
	# Create a loss expression for validation/testing.
	test_prediction = lasagne.layers.get_output(network, deterministic=True)
	test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
															target_var)
	test_loss = test_loss.mean()
	test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
					  dtype=theano.config.floatX)


	train_fn = theano.function([input_var, target_var], loss, updates=updates)
	val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

	# Finally, launch the training loop.
	print("Starting training...")
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
			train_loss += train_fn(inputs, targets)
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
		
		if val_acc>best_val_acc:
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
			np.savez('cifar/model/dense_cnn.npz', *lasagne.layers.get_all_param_values(network))

		# Then we print the results for this epoch:
		print("Epoch {} of {} took {:.3f}s".format(
			epoch + 1, num_epochs, time.time() - start_time))
		print("  training loss:\t\t{:.6f}".format(train_loss))
		print("  validation loss:\t\t{:.6f}".format(val_loss))
		print("  validation accuracy:\t\t{:.2f} %".format(val_acc))		
		print("  test loss:\t\t\t{:.6f}".format(test_loss))
		print("  test accuracy:\t\t{:.2f} %".format(test_acc))
		
		with open("cifar/model/dense_cnn.txt", "a") as myfile:
			myfile.write("{0}  {1:.3f} {2:.3f} {3:.3f} {4:.3f} {5:.3f}\n".format(epoch, train_loss, val_loss, val_acc, test_loss, test_acc))




if __name__ == "__main__":
	parser = ArgumentParser()
	# parser.add_argument("--model", type=str, dest="model",
	# 			default='cnn')	
	parser.add_argument("--num_epochs",  type=int, dest="num_epochs",
				default=100, help="number of epochs")  
	# parser.add_argument("--prune_fraction",  type=float, dest="prune_fraction",
	# 			default=0.8, help="fraction of weights pruned away") 
	args = parser.parse_args()

	main(**vars(args))
