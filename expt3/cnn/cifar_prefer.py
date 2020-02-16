from __future__ import print_function
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import lab_c
from pylearn2.datasets.zca_dataset import ZCA_Dataset    
from pylearn2.utils import serial
from argparse import ArgumentParser
import os

# define new task
def new_task():
	
	print('Loading CIFAR-100 dataset...')
	train_set_size = 45000

	#preprocessor = serial.load("${PYLEARN2_DATA_PATH}/cifar100/pylearn2_gcn_whitened/preprocessor.pkl")\
	preprocessor = serial.load("/home/eric/Desktop/powerLaw/power-law/expt3/cnn/cifar100/pylearn2_gcn_whitened/preprocessor.pkl")

	train_set = ZCA_Dataset(
		preprocessed_dataset=serial.load("/home/eric/Desktop/powerLaw/power-law/expt3/cnn/cifar100/pylearn2_gcn_whitened/train.pkl"),
		preprocessor = preprocessor,
		start=0, stop = train_set_size)
	valid_set = ZCA_Dataset(
		preprocessed_dataset= serial.load("/home/eric/Desktop/powerLaw/power-law/expt3/cnn/cifar100/pylearn2_gcn_whitened/train.pkl"),
		preprocessor = preprocessor,
		start=45000, stop = 50000)  
	test_set = ZCA_Dataset(
		preprocessed_dataset= serial.load("/home/eric/Desktop/powerLaw/power-law/expt3/cnn/cifar100/pylearn2_gcn_whitened/test.pkl"),
		preprocessor = preprocessor)

	ind = np.where(train_set.y.flatten()<10)[0]
	X_train_new = train_set.X[ind, :]
	y_train_new = train_set.y[ind, :]

	ind = np.where(valid_set.y.flatten()<10)[0]
	X_val_new = valid_set.X[ind, :]
	y_val_new = valid_set.y[ind, :]

	ind = np.where(test_set.y.flatten()<10)[0]
	X_test_new = test_set.X[ind, :]
	y_test_new = test_set.y[ind, :]

	X_train_new = X_train_new.reshape(-1,3,32,32)
	X_val_new  	= X_val_new.reshape(-1,3,32,32)
	X_test_new  = X_test_new.reshape(-1,3,32,32)
	
	# flatten targets
	y_train_new = y_train_new.flatten().astype('int32')
	y_val_new   = y_val_new.flatten().astype('int32')
	y_test_new  = y_test_new.flatten().astype('int32')

	return X_train_new, y_train_new, X_val_new, y_val_new, X_test_new, y_test_new



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


# Build Network
def build_cnn(num_filters, input_var=None):
	# Input layer, as usual:
	# one may also add dropout layer according to the icml 2017 paper
	network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32), input_var=input_var)
	network = lasagne.layers.Conv2DLayer(network, num_filters=num_filters, filter_size=(3, 3), pad=1)
	network = lasagne.layers.Conv2DLayer(network, num_filters=num_filters, filter_size=(3, 3),pad=1)
	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
	network = lasagne.layers.Conv2DLayer(network, num_filters=2*num_filters, filter_size=(3, 3), pad=1)
	network = lasagne.layers.Conv2DLayer(network, num_filters=2*num_filters, filter_size=(3, 3), pad=1)
	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2)) 
	network = lasagne.layers.DenseLayer(network, num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
	network =  lasagne.layers.DenseLayer(network, num_units=10, nonlinearity=lasagne.nonlinearities.softmax)

	return network


def build_add_cnn(mask, weight, num_filters, input_var=None):
	# Input layer, as usual:
	# one may also add dropout layer according to the icml 2017 paper
	network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32), input_var=input_var)
	network = lab_c.Conv2DLayer(mask[0], weight[0], network, num_filters=num_filters, filter_size=(3, 3), pad=1)
	network = lab_c.Conv2DLayer(mask[2], weight[2], network, num_filters=num_filters, filter_size=(3, 3),pad=1)
	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
	network = lab_c.Conv2DLayer(mask[4], weight[4], network, num_filters=2*num_filters, filter_size=(3, 3), pad=1)
	network = lab_c.Conv2DLayer(mask[6], weight[6], network, num_filters=2*num_filters, filter_size=(3, 3), pad=1)
	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2)) 
	network = lab_c.DenseLayer(mask[8], weight[8], network, num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
	network =  lab_c.DenseLayer(mask[10], weight[10], network, num_units=10, nonlinearity=lasagne.nonlinearities.softmax)

	return network


# Batch iterator
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

# generate random mask
def random_connection(mask, add_nodes, connect_fraction, layer_size, layer_size_new):
	mask_new = []
	for i in range(len(layer_size)):
		W_mask = np.zeros(layer_size_new[i]).astype('int16')
		if len(layer_size[i])>2:
			# all the bs are relearned
			b_mask = np.ones(layer_size_new[i][0]).astype('int16')
		else:
			# all the bs are relearned
			b_mask = np.ones(layer_size_new[i][1]).astype('int16')
		
		mask_new.append(W_mask)
		mask_new.append(b_mask)			

	for i in range(1, len(layer_size)-1): # i=1,2 only consider the conv layers
		if i>len(layer_size)-3:
			add_connection = int(connect_fraction*layer_size[i][0])
			prob_prev = np.ones(layer_size[i][0])/layer_size[i][0] # choose the filter, and connect to the whole filter
			for j in range(layer_size[i][1]): # the first fc
				ind = np.random.choice(layer_size[i][0], size = (add_connection, 1), p=prob_prev, replace=False)
				mask_new[i+i][ind,j]=1				
			
			mask_new[i+i][layer_size[i][0]::, :]=1   # 512*256 -> 1024*256

		else:
			add_connection = int(connect_fraction*layer_size[i][1])
			prob_prev = np.ones(layer_size[i][1])/layer_size[i][1] # choose the filter, and connect to the whole filter
			for j in range(add_nodes):
				ind = np.random.choice(layer_size[i][1], size = (add_connection, 1), p=prob_prev, replace=False)
				mask_new[i+i][layer_size[i][0]+j, ind,:,:]=1

			mask_new[i+i][layer_size[i][0]::, layer_size[i][1]::, :,:] = 1 

		mask_new[0][layer_size[0][0]:layer_size_new[0][0],:,:,: ]=1
		mask_new[len(layer_size)+len(layer_size)-2][:,:]=1 # the last layer is also relearned

	return mask_new


# generate new mask according to preferential attachment
def prefer_connection(mask, add_nodes, connect_fraction, layer_size, layer_size_new):
	mask_new = []
	for i in range(len(layer_size)):
		W_mask = np.zeros(layer_size_new[i]).astype('int16')
		if len(layer_size[i])>2:
			# all the bs are relearned
			b_mask = np.ones(layer_size_new[i][0]).astype('int16')
		else:
			# all the bs are relearned
			b_mask = np.ones(layer_size_new[i][1]).astype('int16')
		
		mask_new.append(W_mask)
		mask_new.append(b_mask)			

	for i in range(1, len(layer_size)-1):

		if i>len(layer_size)-3:
			add_connection = int(connect_fraction*layer_size[i][0])
			degree_next = np.sum(mask[i+i], axis=1).astype('float32')
			degree_next[degree_next==0]=1e-3
			prob_next = degree_next/np.sum(degree_next)
			for j in range(layer_size[i][1]): # the first fc
				ind = np.random.choice(layer_size[i][0], size = (add_connection, 1), p=prob_next, replace=False)
				mask_new[i+i][ind,j]=1				
			
			mask_new[i+i][layer_size[i][0]::, :]=1   

		else:
			add_connection = int(connect_fraction*layer_size[i][1])
			degree_next = np.sum(mask[i+i], axis=(0,2,3)).astype('float32')
			prob_next = degree_next/np.sum(degree_next)	 ## e.g. 784 \times 400			
			for j in range(add_nodes):
				ind = np.random.choice(layer_size[i][1], size = (add_connection, 1), p=prob_next, replace=False)
				mask_new[i+i][layer_size[i][0]+j, ind,:,:]=1

			mask_new[i+i][layer_size[i][0]::, layer_size[i][1]::, :,:] = 1 

		mask_new[0][layer_size[0][0]:layer_size_new[0][0],:,:,: ]=1
		mask_new[len(layer_size)+len(layer_size)-2][:,:]=1 # the last layer is also relearned

	return mask_new


#  Main program
def main(permute_size, sparsity, num_epochs=500, add_nodes=8, method='AoB', connect_fraction=0.2):
	# Load the dataset
	print("Loading data...")
	X_train_new, y_train_new, X_val_new, y_val_new, X_test_new, y_test_new = new_task()


	input_var = T.tensor4('inputs')
	target_var = T.ivector('targets')

	with np.load('cifar/model/sparse_cnn_{0}.npz'.format(sparsity)) as f:
		param_values = [f['arr_%d' % i] for i in range(len(f.files))]

	mask = []
	for i in range(len(param_values)):
		mask_current= (param_values[i]!=0.).astype('int16') # dense part: 1, sparse part: 0
		mask.append(mask_current)

	layer_size = [[32,3,3,3], [32,32,3,3], [64,32,3,3], [64,64,3,3], [4096, 512], [512,10]]
	layer_size_new = [[32+add_nodes,3,3,3], [32+add_nodes,32+add_nodes,3,3], \
					  [64+add_nodes*2,32+add_nodes,3,3], [64+add_nodes*2,64+add_nodes*2,3,3],
	[4096+2*add_nodes*8*8,512],[512,10]]


	if method=='AoB':
		mask_new = random_connection(mask, add_nodes, connect_fraction, layer_size, layer_size_new)
	else: 
		mask_new = prefer_connection(mask, add_nodes, connect_fraction, layer_size, layer_size_new)

	# the fc layers are all retrained
	param_values[-1]= np.zeros(param_values[-1].shape).astype('float32')
	param_values[-2]= np.zeros(param_values[-2].shape).astype('float32')
	param_values[-3]= np.zeros(param_values[-3].shape).astype('float32')
	param_values[-4]= np.zeros(param_values[-4].shape).astype('float32')

	param_values_aug=[]   # used to construct the network
	for i in range(len(layer_size)):
		if len(layer_size[i])>2:
			W_current = np.zeros(layer_size_new[i]).astype('float32')
			W_current[0:layer_size[i][0],0:layer_size[i][1], 0:layer_size[i][2],0:layer_size[i][3]] = param_values[i+i]
			b_current = np.zeros((layer_size_new[i][0])).astype('float32')
			b_current[0:layer_size[i][0]] = param_values[i+i+1]
		else:
			W_current = np.zeros(layer_size_new[i]).astype('float32')
			W_current[0:layer_size[i][0],0:layer_size[i][1]] = param_values[i+i]
			b_current = np.zeros((layer_size_new[i][1])).astype('float32')
			b_current[0:layer_size[i][1]] = param_values[i+i+1]
		
		param_values_aug.append(W_current)
		param_values_aug.append(b_current)			

	#  rebuild sparse model
	network_c_s = build_add_cnn(mask_new, param_values_aug, param_values_aug[0].shape[0], input_var)

	# Create a loss expression for training
	prediction = lasagne.layers.get_output(network_c_s)
	loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
	loss = loss.mean()
	acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),
					  dtype=theano.config.floatX)	

	# Create update expressions for training
	params = lasagne.layers.get_all_params(network_c_s, trainable=True)
	W_grads = lab_c.compute_grads1(loss, network_c_s, mask_new)
	updates = lasagne.updates.rmsprop(W_grads, params, learning_rate=0.0001)

	# Create a loss expression for validation/testing.
	test_prediction = lasagne.layers.get_output(network_c_s, deterministic=True)
	test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
															target_var)
	test_loss = test_loss.mean()
	test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
					  dtype=theano.config.floatX)


	train_fn = theano.function([input_var, target_var], [loss, acc, W_grads[0], W_grads[2], W_grads[4]], updates=updates)
	val_fn = theano.function([input_var, target_var], [test_loss, test_acc])


	# initialization
	# the first column can not be randomly initialized, and should be sparse
	init_params = lasagne.layers.get_all_param_values(network_c_s)
	for i in range(len(layer_size)-1):   # i=0,1,2
		if len(layer_size[i])>2:
			init_params[i+i][0:layer_size[i][0],0:layer_size[i][1],:,:] = param_values[i+i]  #### optional, as mask will make it 0
			init_params[i+i][0:layer_size[i][0],layer_size[i][1]:] = \
				init_params[i+i][0:layer_size[i][0],layer_size[i][1]:]*mask_new[i+i][0:layer_size[i][0],layer_size[i][1]:]
			init_params[i+i][layer_size[i][0]:,0:layer_size[i][1]] = \
				init_params[i+i][layer_size[i][0]:,0:layer_size[i][1]]*mask_new[i+i][layer_size[i][0]:,0:layer_size[i][1]]
			init_params[i+i+1][0:layer_size[i][0]] = param_values[i+i+1]
		else:
			init_params[i+i][0:layer_size[i][0],0:layer_size[i][1]] = param_values[i+i] #### optional, as mask will make it 0
			init_params[i+i][0:layer_size[i][0],layer_size[i][1]:] = \
				init_params[i+i][0:layer_size[i][0],layer_size[i][1]:]*mask_new[i+i][0:layer_size[i][0],layer_size[i][1]:]
			init_params[i+i][layer_size[i][0]:,0:layer_size[i][1]] = \
				init_params[i+i][layer_size[i][0]:,0:layer_size[i][1]]*mask_new[i+i][layer_size[i][0]:,0:layer_size[i][1]]
			init_params[i+i+1][0:layer_size[i][0]] = param_values[i+i+1]

	lasagne.layers.set_all_param_values(network_c_s, init_params)


	print("Starting retraining...")
	best_val_acc = 0
	# We iterate over epochs:
	for epoch in range(num_epochs):
		# In each epoch, we do a full pass over the training data:
		train_loss = 0
		train_acc = 0
		train_batches = 0
		start_time = time.time()
		for batch in iterate_minibatches(X_train_new, y_train_new, 50, shuffle=True):
			inputs, targets = batch
			loss, acc, g1, g2, g3 = train_fn(inputs, targets)
			train_loss += loss
			train_acc += acc
			train_batches += 1

		train_loss = train_loss / train_batches
		train_acc = train_acc /train_batches * 100

		# And a full pass over the validation data:
		val_loss = 0
		val_acc = 0
		val_batches = 0
		for batch in iterate_minibatches(X_val_new, y_val_new, 50, shuffle=False):
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
			# After training, we compute and print the test error:
			test_loss = 0
			test_acc = 0
			test_batches = 0
			for batch in iterate_minibatches(X_test_new, y_test_new, 50, shuffle=False):
				inputs, targets = batch
				loss, acc = val_fn(inputs, targets)
				test_loss += loss
				test_acc += acc
				test_batches += 1
			test_loss = test_loss / test_batches
			test_acc = test_acc / test_batches * 100

		print("Epoch {} of {} took {:.3f}s".format(
			epoch + 1, num_epochs, time.time() - start_time))
		print("  training loss:\t\t{:.6f}".format(train_loss))
		print("  train accuracy:\t\t{:.2f} %".format(train_acc))
		print("  validation loss:\t\t{:.6f}".format(val_loss))
		print("  validation accuracy:\t\t{:.2f} %".format(val_acc))
		print("  test loss:\t\t\t{:.6f}".format(test_loss))
		print("  test accuracy:\t\t{:.2f} %".format(test_acc))

		result_folder = "cifar/prefer_model/{0}".format(int(connect_fraction*100))
		if not os.path.exists(result_folder):
			os.mkdir(result_folder)

		with open("cifar/prefer_model/{0}/cont_sparse_cnn_{1}_{2}_{3}_{4}.txt".format(
				int(connect_fraction*100), method, add_nodes,sparsity, permute_size), "a") as myfile:
			myfile.write("{0}  {1:.3f} {2:.3f} {3:.3f} {4:.3f} {5:.3f} {6:.3f}\n".format(epoch, 
				train_loss, train_acc, val_loss, val_acc, test_loss, test_acc))


if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("--permute_size",  type=int, dest="permute_size",
				default=1, help="the second task folder")  
	parser.add_argument("--sparsity",  type=float, dest="sparsity",
				default=0.7, help="sparsity of sparse model")  
	parser.add_argument("--num_epochs",  type=int, dest="num_epochs",
				default=100, help="number of epochs")
	parser.add_argument("--method",  type=str, dest="method",
				default='AoB', help="methods: AoB, AoB+, ApB, ApB+")  
	parser.add_argument("--add_nodes",  type=int, dest="add_nodes",
				default=8, help="number of nodes added to each hiddn layer") 
	parser.add_argument("--connect_fraction",  type=float, dest="connect_fraction",
				default=0.2, help="fraction of connection to previous solutions") 
	args = parser.parse_args()

	main(**vars(args))
