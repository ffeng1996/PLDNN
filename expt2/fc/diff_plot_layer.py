import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from argparse import ArgumentParser

def update_label(old_label, exponent_text):
    if exponent_text == "":
        return old_label
    
    try:
        units = old_label[old_label.index("[") + 1:old_label.rindex("]")]
    except ValueError:
        units = ""
    label = old_label.replace("[{}]".format(units), "")
    
    exponent_text = exponent_text.replace("\\times", "")
    # import ipdb; ipdb.set_trace()
    return "{} [{} {}]".format(label, exponent_text, units)
    
def format_label_string_with_exponent(ax, axis='both'):  
    """ Format the label string with the exponent from the ScalarFormatter """
    ax.ticklabel_format(axis=axis, style='sci')
    # import ipdb; ipdb.set_trace()

    axes_instances = []
    if axis in ['x', 'both']:
        axes_instances.append(ax.xaxis)
    if axis in ['y', 'both']:
        axes_instances.append(ax.yaxis)
    
    for ax in axes_instances:
        ax.major.formatter._useMathText = True
        plt.draw() # Update the text
        exponent_text = ax.get_offset_text().get_text()
        label = ax.get_label().get_text()    ##### it does not work, returns nothing
        ax.offsetText.set_visible(False)
        ax.set_label_text(update_label(label, exponent_text))


def diff_plot(sparse_degree, cont_sparse_degree, num_layers, hidden_units, permute_size, layer):
	
	difference = [i-j for i,j in zip(cont_sparse_degree, sparse_degree)]
	min_degree = np.min(sparse_degree)
	max_degree = np.max(sparse_degree)
	num_bins = max_degree - min_degree + 1
	degree_counts = np.zeros(num_bins)
	diff = np.zeros(num_bins)
	for i in xrange(len(sparse_degree)):
		degree_counts[sparse_degree[i]-min_degree] += 1
		diff[sparse_degree[i]-min_degree] += difference[i]
		
	# mean_diff = diff/degree_counts

	# ind = np.nonzero(degree_counts)
	ind = np.nonzero(diff)
	mean_diff_new = diff[ind]/degree_counts[ind]
	# mean_diff_new = mean_diff[np.nonzero(degree_counts)]

	# import ipdb; ipdb.set_trace()

	p = plt.figure(figsize=(6,4), dpi=80)
	p3 = p.add_subplot(111)
	p3.set_xlabel(' $d$ [$\\times 10^2$]', fontsize=16)
	p3.set_ylabel(' $ \\hat{\Omega}_0^l(d)$ [$\\times 10^2$]', fontsize=16)
	p3.tick_params(axis="both",  which='major', labelsize=16)
	p3.xaxis.get_offset_text().set_fontsize(16)
	p3.yaxis.get_offset_text().set_fontsize(16)
	# format_label_string_with_exponent(p3, axis='both')  ### it can not ge the off_set_text

	# p3.set_title("log-log plot")
	# import ipdb; ipdb.set_trace()
	p3.plot((ind[0]+min_degree)/100., (mean_diff_new)/100., color="b", linewidth=2)
	plt.tight_layout()
	p.savefig('figures/diff/layer/mnist_{0}_{1}_{2}_{3}.png'.format(str(num_layers), str(hidden_units), str(permute_size), layer))




def plot(data, data2, num_layers, hidden_units, permute_size):
	degree = []
	degree2 = []
	for i in range(len(data)):
		data_current = np.abs(data[i])
		data_current2 = np.abs(data2[i])
		if i < len(data)-1:
			data_next = np.abs(data[i+1])
			data_next2 = np.abs(data2[i+1])
		data_current[data_current > 0.0] = 1
		data_current2[data_current2>0.0] = 1
		# counts both-sides degree
		# input nodes
		if i < 1:
			first_degree = np.sum(data_current,axis = 1).astype(int)
			first_degree2 = np.sum(data_current2,axis = 1).astype(int)
			degree = degree + first_degree.tolist()
			degree2 = degree2 + first_degree2.tolist()
			diff_plot(first_degree, first_degree2,num_layers, hidden_units, permute_size, i)

		# hidden nodes 
		if i < len(data)-1 and len(data_current.shape)==len(data_next.shape):
			current_degree = np.sum(data_current,axis=0).astype(int)  
			current_degree2 = np.sum(data_current2,axis=0).astype(int)  

			data_next[data_next>0.0] = 1
			data_next2[data_next2>0.0] = 1
			next_degree = np.sum(data_next,axis=1).astype(int)
			next_degree2 = np.sum(data_next2,axis=1).astype(int)

			current_degree = current_degree+next_degree
			current_degree2 = current_degree2+next_degree2

			diff_plot(current_degree, current_degree2,num_layers, hidden_units, permute_size, (i+1))
			degree = degree + current_degree.tolist()
			degree2 = degree2 + current_degree2.tolist()
	diff_plot(degree, degree2,  num_layers, hidden_units, permute_size, 'all')


def main(filename, filename1, option, num_layers, hidden_units, permute_size):

	with np.load(filename) as f:
		param_values_sparse = [f['arr_%d' % i] for i in range(len(f.files))]

	with np.load(filename1) as f:
		param_values_cont_sparse = [f['arr_%d' % i] for i in range(len(f.files))]

	plot([param_values_sparse[0], param_values_sparse[2], param_values_sparse[4]],  
		[param_values_cont_sparse[0], param_values_cont_sparse[2], param_values_cont_sparse[4]],
	 num_layers, hidden_units, permute_size)



if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("--filename", type=str, dest="filename",
				default='model/sparse_model_0.9_2_1024.npz', help = "the sparse model")
	parser.add_argument("--filename1", type=str, dest="filename1",
				default='cont_model/cont_sparse_fc_0.1_0.9_8_2_1024.npz', help = "sparse model trained on the second task")
	parser.add_argument("--option", type=str, dest="option",
				default="orig", help="orig or ccdf")
	parser.add_argument("--num_layers",  type=float, dest="num_layers",
				default=2, help="number of layers") 
	parser.add_argument("--hidden_units",  type=int, dest="hidden_units",
				default=1024, help="number of hidden units in each layer")
	parser.add_argument("--permute_size",  type=int, dest="permute_size",
				default=8, help="permutation size, can be 8 or 26") 
	args = parser.parse_args()

	main(**vars(args))