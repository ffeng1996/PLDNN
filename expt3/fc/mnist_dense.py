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

# ############################ Permute images ############################

def permute_mnist(window_size, X_train, y_train, X_val, y_val, X_test, y_test):
	num_permute = window_size*window_size
	shift = (X_train.shape[1] - num_permute)/2
	perm_inds = range(num_permute)
	np.random.shuffle(perm_inds)
	perm_inds = np.array(perm_inds) +  shift
	# import ipdb; ipdb.set_trace()
	
	def permute_one(inds, X):
		X_new = np.array([X[:,c] for c in inds])
		return X_new

	perm_inds = np.concatenate([np.arange(shift), perm_inds, np.arange(shift)+num_permute+shift])
	perm_inds = perm_inds.tolist()


	X_train_new = permute_one(perm_inds, X_train)
	X_val_new = permute_one(perm_inds, X_val)
	X_test_new = permute_one(perm_inds, X_test)

	return X_train_new, y_train, X_val_new, y_val, X_test_new, y_test



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
		data = data.reshape(-1,784)
		# import ipdb; ipdb.set_trace()
		# data = data.reshape(-1, 1, 28, 28)
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
def build_custom_mlp(input_var=None, depth=2, width=800, drop_input=.2,
					 drop_hidden=.5):

	network = lasagne.layers.InputLayer(shape=(None, 784),
										input_var=input_var)
	if drop_input:
		network = lasagne.layers.dropout(network, p=drop_input)
	# Hidden layers and dropout:
	nonlin = lasagne.nonlinearities.rectify
	for _ in range(depth):
		network = lasagne.layers.DenseLayer(
				network, width, nonlinearity=nonlin)
		if drop_hidden:
			network = lasagne.layers.dropout(network, p=drop_hidden)
	# Output layer:
	softmax = lasagne.nonlinearities.softmax
	network = lasagne.layers.DenseLayer(network, 10, nonlinearity=softmax)
	return network


# ############################### Build Sparse Network ###############################
# def build_sparse_mlp(mask, input_var=None, depth=2, width=800, drop_input=.2,
# 					 drop_hidden=.5):

# 	network = lasagne.layers.InputLayer(shape=(None, 784),
# 										input_var=input_var)
# 	if drop_input:
# 		network = lasagne.layers.dropout(network, p=drop_input)
# 	# Hidden layers and dropout:
# 	nonlin = lasagne.nonlinearities.rectify
# 	i=0
# 	for _ in range(depth):
# 		network = lab.DenseLayer(mask[i+i], network, width, nonlinearity=nonlin)
# 		if drop_hidden:
# 			network = lasagne.layers.dropout(network, p=drop_hidden)
# 		i = i +1
# 	# Output layer:
# 	softmax = lasagne.nonlinearities.softmax
# 	network = lab.DenseLayer(mask[i+i], network, 10, nonlinearity=softmax)
# 	return network

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

def main(model='fc', num_epochs=500):
	# Load the dataset
	print("Loading data...")
	X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
	# X_train_new, y_train_new, X_val_new, y_val_new, X_test_new, y_test_new = permute_mnist(26, X_train, y_train, X_val, y_val, X_test, y_test)

	# Prepare Theano variables for inputs and targets
	# input_var = T.tensor4('inputs')
	input_var = T.fmatrix('inputs')
	target_var = T.ivector('targets')

	# Create neural network model (depending on first command line parameter)

	depth, width, drop_in, drop_hid = model.split(':', 1)[1].split(',')

############################################ train dense model #############################################
	network = build_custom_mlp(input_var, int(depth), int(width),
								   float(drop_in), float(drop_hid))

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
			np.savez('model/model_{0}_{1}.npz'.format(depth, width), *lasagne.layers.get_all_param_values(network))

		# Then we print the results for this epoch:
		print("Epoch {} of {} took {:.3f}s".format(
			epoch + 1, num_epochs, time.time() - start_time))
		print("  training loss:\t\t{:.6f}".format(train_loss))
		print("  validation loss:\t\t{:.6f}".format(val_loss))
		print("  validation accuracy:\t\t{:.2f} %".format(val_acc))		
		print("  test loss:\t\t\t{:.6f}".format(test_loss))
		print("  test accuracy:\t\t{:.2f} %".format(test_acc))
		
		with open("model/dense_{0}_{1}_{2}_{3}.txt".format(depth, width, drop_in, drop_hid), "a") as myfile:
			myfile.write("{0}  {1:.3f} {2:.3f} {3:.3f} {4:.3f} {5:.3f}\n".format(epoch, train_loss, val_loss, val_acc, test_loss, test_acc))




if __name__ == '__main__':
	if ('--help' in sys.argv) or ('-h' in sys.argv):
		print("Trains a neural network on MNIST using Lasagne.")
		print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
		print("EPOCHS: number of training epochs to perform (default: 500)")
	else:
		kwargs = {}
		if len(sys.argv) > 1:
			kwargs['model'] = sys.argv[1]
		if len(sys.argv) > 2:
			kwargs['num_epochs'] = int(sys.argv[2])
		if len(sys.argv) >3:
			kwargs['prune_fraction'] = float(sys.argv[3])
		main(**kwargs)
