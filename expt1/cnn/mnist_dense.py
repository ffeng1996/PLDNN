from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
import lab
import lasagne
from collections import OrderedDict
from argparse import ArgumentParser


# ################## Download and prepare the MNIST dataset ##################
def load_dataset():
	# We first define a download function, supporting both Python 2 and 3.
	if sys.version_info[0] == 2:
		from urllib import urlretrieve
	else:
		from urllib.request import urlretrieve

	def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
		print("Downloading %s" % filename)
		urlretrieve(source + filename, filename)

	# We then define functions for loading MNIST images and labels.
	# For convenience, they also download the requested files if needed.
	import gzip

	def load_mnist_images(filename):
		if not os.path.exists(filename):
			download(filename)
		with gzip.open(filename, 'rb') as f:
			data = np.frombuffer(f.read(), np.uint8, offset=16)
		# data = data.reshape(-1,784)
		data = data.reshape(-1, 1, 28, 28)
		return data / np.float32(256)

	def load_mnist_labels(filename):
		if not os.path.exists(filename):
			download(filename)
		# Read the labels in Yann LeCun's binary format.
		with gzip.open(filename, 'rb') as f:
			data = np.frombuffer(f.read(), np.uint8, offset=8)
		# The labels are vectors of integers now, that's exactly what we want.
		return data

	# We can now download and read the training and test set images and labels.
	X_train = load_mnist_images('train-images-idx3-ubyte.gz')
	y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
	X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
	y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

	# We reserve the last 10000 training examples for validation.
	X_train, X_val = X_train[:-10000], X_train[-10000:]
	y_train, y_val = y_train[:-10000], y_train[-10000:]
    
	# We just return all the arrays in order, as expected in main().
	# (It doesn't matter how we do this as long as we can read them again.)
	return X_train, y_train, X_val, y_val, X_test, y_test

############################### Build Network ###############################
def build_cnn(input_var=None):
	# Input layer, as usual:
	network = lasagne.layers.InputLayer(shape=(None, 1,28,28),
										input_var=input_var)
	network = lasagne.layers.Conv2DLayer(
			network, num_filters=32, filter_size=(5, 5),
			nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.GlorotUniform())

	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

	# Another convolution with 32 5x5 kernels, and another 2x2 pooling:
	network = lasagne.layers.Conv2DLayer(
			network, num_filters=32, filter_size=(5, 5),
			nonlinearity=lasagne.nonlinearities.rectify)
	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

	# A fully-connected layer of 256 units with 50% dropout on its inputs:
	network = lasagne.layers.DenseLayer(
			network,
			# lasagne.layers.dropout(network, p=.5),
			num_units=256,
			nonlinearity=lasagne.nonlinearities.rectify)

	# And, finally, the 10-unit output layer with 50% dropout on its inputs:
	network = lasagne.layers.DenseLayer(
			network,
			# lasagne.layers.dropout(network, p=.5),
			num_units=10,
			nonlinearity=lasagne.nonlinearities.softmax)

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

def main( model='fc', num_epochs=500):
	# Load the dataset
	print("Loading data...")
	X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
	# import ipdb; ipdb.set_trace()
	# X_train_new, y_train_new, X_val_new, y_val_new, X_test_new, y_test_new = permute_mnist(26, X_train, y_train, X_val, y_val, X_test, y_test)

	# Prepare Theano variables for inputs and targets
	input_var = T.tensor4('inputs')
	# input_var = T.fmatrix('inputs')
	target_var = T.ivector('targets')

############################################ train dense model #############################################
	network = build_cnn(input_var)	
	# Create a loss expression for training
	prediction = lasagne.layers.get_output(network)
	loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
	loss = loss.mean()

	# Create update expressions for training
	params = lasagne.layers.get_all_params(network, trainable=True)
	updates = lasagne.updates.nesterov_momentum(
			loss, params, learning_rate=0.01, momentum=0.9)

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
		for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
			inputs, targets = batch
			train_loss += train_fn(inputs, targets)
			train_batches += 1

		train_loss = train_loss / train_batches

		# And a full pass over the validation data:
		val_loss = 0
		val_acc = 0
		val_batches = 0
		for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
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
			for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
				inputs, targets = batch
				loss, acc = val_fn(inputs, targets)
				test_loss += loss
				test_acc += acc
				test_batches += 1
			test_loss = test_loss / test_batches
			test_acc = test_acc / test_batches * 100
			np.savez('model/dense_cnn.npz', *lasagne.layers.get_all_param_values(network))

		# Then we print the results for this epoch:
		print("Epoch {} of {} took {:.3f}s".format(
			epoch + 1, num_epochs, time.time() - start_time))
		print("  training loss:\t\t{:.6f}".format(train_loss))
		print("  validation loss:\t\t{:.6f}".format(val_loss))
		print("  validation accuracy:\t\t{:.2f} %".format(val_acc))		
		print("  test loss:\t\t\t{:.6f}".format(test_loss))
		print("  test accuracy:\t\t{:.2f} %".format(test_acc))
		
		with open("model/dense_cnn.txt", "a") as myfile:
			myfile.write("{0}  {1:.3f} {2:.3f} {3:.3f} {4:.3f} {5:.3f}\n".format(epoch, train_loss, val_loss, val_acc, test_loss, test_acc))




if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("--model", type=str, dest="model",
				default='cnn')	
	parser.add_argument("--num_epochs",  type=int, dest="num_epochs",
				default=200, help="number of epochs")  
	# parser.add_argument("--prune_fraction",  type=float, dest="prune_fraction",
	# 			default=0.8, help="fraction of weights pruned away") 
	args = parser.parse_args()

	main(**vars(args))
