# set xticks by hand for better visualization

import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
import pylab

pylab.rcParams['xtick.major.pad'] = '16'
pylab.rcParams['ytick.major.pad'] = '16'
from matplotlib import rc

rc('font', family='sans-serif')
rc('font', size=16.0)
rc('text', usetex=False)
from scipy.optimize import minimize
import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import BatchNormLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import DenseLayer
from lasagne.nonlinearities import rectify, softmax

def S(xmin, xmax, x, a):
    z = np.sum(np.power(np.arange(xmin, xmax + 1), -a))
    yPred = (np.power(xmax, 1 - a) - np.power(x, 1 - a)) / ((1 - a) * z)
    return yPred


def precise(xmin, xmax, x):
    def nll(params):
        a = params[0]
        z = np.sum(np.power(np.arange(xmin, xmax + 1), -a))
        yPred = 1 / z * np.power(x, -a)

        # Calculate negative log likelihood
        NLL = -np.sum(np.log10(yPred))
        return NLL

    return nll


def fitting(xmin, xmax, x):
    init_params = 10
    results = minimize(precise(xmin, xmax, x), init_params, method='L-BFGS-B', bounds=((1.001, 10),))
    a = results.x[0]
    return a


# this follows the method in section G of paper corral et. al
def monte_carlo_simulation(xmin, xmax, alpha_e, number_of_samples):
    xmin = np.float32(xmin)
    xmax = np.float32(xmax)
    r = xmin / xmax
    mu = np.random.uniform(0, 1, number_of_samples)
    sim = xmin / np.power(1 - (1 - np.power(r, alpha_e - 1)) * mu, 1 / (alpha_e - 1))

    sim = np.floor(sim)
    return sim


def p_test(xmin, xmax, alpha_e, KS, number_of_smaples):
    num_larger = 0.
    N = 500
    for i in range(N):
        x_s = monte_carlo_simulation(xmin, xmax, alpha_e, number_of_smaples)

        #         print x_s
        alpha_s = fitting(xmin, xmax, x_s)
        Theoretical_CCDF = S(xmin, xmax, np.arange(xmin, xmax + 1), alpha_s)

        x_s = sorted(x_s, reverse=True)
        bins = np.arange(xmin - 0.5, xmax + 1, 1)
        h, bins = np.histogram(x_s, density=True, bins=bins)
        counts = h * len(x_s)
        counts = np.cumsum(counts[::-1])[::-1]
        Actual_CCDF = counts / counts[0]

        # get the fitting results
        CCDF_diff = Theoretical_CCDF - Actual_CCDF
        D = np.max(np.abs(CCDF_diff))
        if D > KS:
            num_larger = num_larger + 1

        p_value = num_larger / N

    return p_value


def plot(degree, str1, str2, str3, option, xaxis_labels):
    degree = sorted(degree, reverse=True)

    KS_max = 1
    cut_fraction1 = 0.30
    cut_fraction2 = 0.30
    # choose xmin in the smallest xmin_fraction degrees, and xmax in the largest xmax_fraction degrees
    cutting_number1 = int(len(degree) * cut_fraction1)
    cutting_number2 = int(len(degree) * cut_fraction2)
    for i in range(2, cutting_number1):  # iterate for xmin
        for j in range(cutting_number2):  # iterate for xmax
            x = degree[j:-i + 1]  # from large to small
            xmin = min(x)
            xmax = max(x)
            # get the actual results
            bins = np.arange(min(x) - 0.5, max(x) + 1, 1)
            h, bins = np.histogram(x, density=True, bins=bins)
            counts = h * len(x)
            counts = np.cumsum(counts[::-1])[::-1]
            Actual_CCDF = counts / counts[0]

            # get the fitting results
            init_params = 10
            bounds_emp = ((1.001, 10),)  # set the lower-bound to 0.0001 or 1.0001 and see the diff
            results = minimize(precise(xmin, xmax, x), init_params, method='L-BFGS-B', bounds=bounds_emp)
            a = results.x[0]
            Theoretical_CCDF = S(xmin, xmax, np.arange(xmin, xmax + 1), a)
            # get KS
            CCDF_diff = Theoretical_CCDF - Actual_CCDF
            D = np.max(np.abs(CCDF_diff))  # this is the same as the cdf diff

            if D < KS_max:
                KS_max = D
                best_xmin = xmin
                best_xmax = xmax
                best_a = a
                best_counts = counts
                best_Actual_CCDF = Actual_CCDF
                best_num_x = len(x)

    p_value = p_test(best_xmin, best_xmax, best_a, KS_max, best_num_x)
    print best_xmin, best_xmax, best_a, best_num_x, p_value


    p = plt.figure (figsize=(6,4), dpi=80)
    p3 = p.add_subplot(111)
    p3.set_xscale("log")
    p3.set_yscale("log")
    p3.set_title(r'$\alpha$=')
    p3.set_xlim(best_xmin, best_xmax)
    p3.set_ylim(1e-4, 1)
    p3.set_xlabel('degree', fontsize=20)
    p3.set_ylabel('p(X>=x)', fontsize=20)

    p3.plot(np.arange(len(best_counts)) + best_xmin, best_Actual_CCDF, "o", color='b', linewidth=5)
    p3.tick_params(axis="x", which='major', bottom='off', top='off', labelbottom='off')
    p3.tick_params(axis="both", which='minor', labelsize=14)

    p3.set_xticks([best_xmin, best_xmax], minor=True)  # xaxis by hand
    p3.plot(np.arange(best_xmin, best_xmax), S(best_xmin, best_xmax, np.arange(best_xmin, best_xmax), best_a),
            color='r', linewidth=5)

    plt.tight_layout()
    p.savefig(
        'figures/resnet60_dsd_layer/powerlaw_clauset/{3}_imagenet_{0}_{1}_{2}_{3}.png'.format(option, str1, str2, str3))


def degree_plot(data, name, prune_fraction, shape_info, option, xaxis_labels_all):
    thres = []
    for i in range(len(data)):
        data_current = np.abs(data[i])
        vec_data = data_current.flatten()
        a = int(prune_fraction * data_current.size)
        thres.append(np.sort(vec_data)[a])


    degree = []
    for i in range(len(data)):
        data_current = np.abs(data[i])
        if i < len(data) - 1:
            data_next = np.abs(data[i + 1])
        # this will not work when the threshold is larger than 1
        data_current[data_current <= thres[i]] = 0
        data_current[data_current > thres[i]] = 1

        if len(data_current.shape) > 2:
            current_degree = np.sum(data_current, axis=(1, 2, 3)).astype(int)
            current_degree = current_degree * shape_info[i][2] * shape_info[i][3]
        else:
            current_degree = np.sum(data_current, axis=1).astype(int)  # axis is different from vgg_s

        if i < len(data) - 1 and len(data_current.shape) == len(data_next.shape):  # neglect the conv_fc connection layer nodes
            data_next[data_next <= thres[i + 1]] = 0
            data_next[data_next > thres[i + 1]] = 1
            if len(data_next.shape) > 2:
                next_degree = np.sum(data_next, axis=(0, 2, 3)).astype(int)
                next_degree = next_degree * shape_info[i + 1][2] * shape_info[i + 1][3]
            else:
                next_degree = np.sum(data_next, axis=0).astype(int)

            if next_degree.size < current_degree.size:
                next_degree = np.concatenate((next_degree, next_degree))

            #print (current_degree.shape, next_degree.shape)
            current_degree = current_degree + next_degree
            current_degree = current_degree[np.nonzero(current_degree)]
        plot(current_degree.tolist(), name, str(i + 1), str(prune_fraction).split('.')[1], option, xaxis_labels_all[i])
        degree = degree + current_degree.tolist()


def build_simple_block(incoming_layer, names,
                       num_filters, filter_size, stride, pad,
                       use_bias=False, nonlin=rectify):
    """Creates stacked Lasagne layers ConvLayer -> BN -> (ReLu)
    Parameters:
    ----------
    incoming_layer : instance of Lasagne layer
        Parent layer
    names : list of string
        Names of the layers in block
    num_filters : int
        Number of filters in convolution layer
    filter_size : int
        Size of filters in convolution layer
    stride : int
        Stride of convolution layer
    pad : int
        Padding of convolution layer
    use_bias : bool
        Whether to use bias in conlovution layer
    nonlin : function
        Nonlinearity type of Nonlinearity layer
    Returns
    -------
    tuple: (net, last_layer_name)
        net : dict
            Dictionary with stacked layers
        last_layer_name : string
            Last layer name
    """
    net = []
    net.append((
            names[0],
            ConvLayer(incoming_layer, num_filters, filter_size, stride, pad,
                      flip_filters=False, nonlinearity=None) if use_bias
            else ConvLayer(incoming_layer, num_filters, filter_size, stride, pad, b=None,
                           flip_filters=False, nonlinearity=None)
        ))

    net.append((
            names[1],
            BatchNormLayer(net[-1][1])
        ))
    if nonlin is not None:
        net.append((
            names[2],
            NonlinearityLayer(net[-1][1], nonlinearity=nonlin)
        ))

    return dict(net), net[-1][0]


def build_residual_block(incoming_layer, ratio_n_filter=1.0, ratio_size=1.0, has_left_branch=False,
                         upscale_factor=4, ix=''):
    """Creates two-branch residual block
    Parameters:
    ----------
    incoming_layer : instance of Lasagne layer
        Parent layer
    ratio_n_filter : float
        Scale factor of filter bank at the input of residual block
    ratio_size : float
        Scale factor of filter size
    has_left_branch : bool
        if True, then left branch contains simple block
    upscale_factor : float
        Scale factor of filter bank at the output of residual block
    ix : int
        Id of residual block
    Returns
    -------
    tuple: (net, last_layer_name)
        net : dict
            Dictionary with stacked layers
        last_layer_name : string
            Last layer name
    """
    simple_block_name_pattern = ['res%s_branch%i%s', 'bn%s_branch%i%s', 'res%s_branch%i%s_relu']

    net = {}

    # right branch
    net_tmp, last_layer_name = build_simple_block(
        incoming_layer, list(map(lambda s: s % (ix, 2, 'a'), simple_block_name_pattern)),
        int(lasagne.layers.get_output_shape(incoming_layer)[1]*ratio_n_filter), 1, int(1.0/ratio_size), 0)
    net.update(net_tmp)

    net_tmp, last_layer_name = build_simple_block(
        net[last_layer_name], list(map(lambda s: s % (ix, 2, 'b'), simple_block_name_pattern)),
        lasagne.layers.get_output_shape(net[last_layer_name])[1], 3, 1, 1)
    net.update(net_tmp)

    net_tmp, last_layer_name = build_simple_block(
        net[last_layer_name], list(map(lambda s: s % (ix, 2, 'c'), simple_block_name_pattern)),
        lasagne.layers.get_output_shape(net[last_layer_name])[1]*upscale_factor, 1, 1, 0,
        nonlin=None)
    net.update(net_tmp)

    right_tail = net[last_layer_name]
    left_tail = incoming_layer

    # left branch
    if has_left_branch:
        net_tmp, last_layer_name = build_simple_block(
            incoming_layer, list(map(lambda s: s % (ix, 1, ''), simple_block_name_pattern)),
            int(lasagne.layers.get_output_shape(incoming_layer)[1]*4*ratio_n_filter), 1, int(1.0/ratio_size), 0,
            nonlin=None)
        net.update(net_tmp)
        left_tail = net[last_layer_name]

    net['res%s' % ix] = ElemwiseSumLayer([left_tail, right_tail], coeffs=1)
    net['res%s_relu' % ix] = NonlinearityLayer(net['res%s' % ix], nonlinearity=rectify)

    return net, 'res%s_relu' % ix

def main(option, threshold):
    from lasagne.layers import DropoutLayer
    import lasagne
    from lasagne.layers import InputLayer
    from lasagne.layers import Conv2DLayer as ConvLayer
    from lasagne.layers import BatchNormLayer
    from lasagne.layers import Pool2DLayer as PoolLayer
    from lasagne.layers import NonlinearityLayer
    from lasagne.layers import ElemwiseSumLayer
    from lasagne.layers import DenseLayer
    from lasagne.nonlinearities import rectify, softmax
    # define network structure

    net = {}
    net['input'] = InputLayer((None, 3, 224, 224))
    sub_net, parent_layer_name = build_simple_block(
        net['input'], ['conv1', 'bn_conv1', 'conv1_relu'],
        64, 7, 2, 3, use_bias=True)
    net.update(sub_net)
    net['pool1'] = PoolLayer(net[parent_layer_name], pool_size=3, stride=2, pad=0, mode='max', ignore_border=False)
    block_size = list('abc')
    parent_layer_name = 'pool1'
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1, 1, True, 4, ix='2%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='2%s' % c)
        net.update(sub_net)

    block_size = list('abcd')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='3%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='3%s' % c)
        net.update(sub_net)

    block_size = list('abcdef')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='4%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='4%s' % c)
        net.update(sub_net)

    block_size = list('abc')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='5%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='5%s' % c)
        net.update(sub_net)
    net['pool5'] = PoolLayer(net[parent_layer_name], pool_size=7, stride=1, pad=0,
                             mode='average_exc_pad', ignore_border=False)
    net['fc1000'] = DenseLayer(net['pool5'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc1000'], nonlinearity=softmax)

    import caffe
    net_caffe = caffe.Net('ResNet-50-deploy.prototxt', 'ResNet-50-model.caffemodel', caffe.TEST)
    layers_caffe = dict(zip(list(net_caffe._layer_names), net_caffe.layers))

    for name, layer in net.items():
        try:
            layer.W.set_value(layers_caffe[name].blobs[0].data)
            layer.b.set_value(layers_caffe[name].blobs[1].data)
        except AttributeError:
            continue

    all_param = lasagne.layers.get_all_param_values(net['prob'])

    #for i, value in enumerate(all_param):
     #   print value.shape

    layers = [net['conv1'],
              #net['res2a_branch1'],
              net['res2a_branch2a'], net['res2a_branch2b'], net['res2a_branch2c'],
              net['res2b_branch2a'], net['res2b_branch2b'], net['res2b_branch2c'],
              net['res2c_branch2a'], net['res2c_branch2b'], net['res2c_branch2c'],
             # net['res3a_branch1'],
              net['res3a_branch2a'], net['res3a_branch2b'], net['res3a_branch2c'],
              net['res3b_branch2a'], net['res3b_branch2b'], net['res3b_branch2c'],
              net['res3c_branch2a'], net['res3c_branch2b'], net['res3c_branch2c'],
              net['res3d_branch2a'], net['res3d_branch2b'], net['res3d_branch2c'],
              #net['res4a_branch1'],
              net['res4a_branch2a'], net['res4a_branch2b'], net['res4a_branch2c'],
              net['res4b_branch2a'], net['res4b_branch2b'], net['res4b_branch2c'],
              net['res4c_branch2a'], net['res4c_branch2b'], net['res4c_branch2c'],
              net['res4d_branch2a'], net['res4d_branch2b'], net['res4d_branch2c'],
              net['res4e_branch2a'], net['res4e_branch2b'], net['res4e_branch2c'],
              net['res4f_branch2a'], net['res4f_branch2b'], net['res4f_branch2c'],
              #net['res5a_branch1'],
              net['res5a_branch2a'], net['res5a_branch2b'], net['res5a_branch2c'],
              net['res5b_branch2a'], net['res5b_branch2b'], net['res5b_branch2c'],
              net['res5c_branch2a'], net['res5c_branch2b'], net['res5c_branch2c']]
    shape_info = lasagne.layers.get_output_shape(layers)
    layer_all = [all_param[0],
                 #all_param[6],
                 all_param[11], all_param[16], all_param[21],
                 all_param[26], all_param[31], all_param[36],
                 all_param[41], all_param[46], all_param[51],
                 #all_param[56],
                 all_param[61], all_param[66], all_param[71],
                 all_param[76], all_param[81], all_param[86],
                 all_param[91], all_param[96], all_param[101],
                 all_param[106], all_param[111], all_param[116],
                 #all_param[121],
                 all_param[126], all_param[131], all_param[136],
                 all_param[141], all_param[146], all_param[151],
                 all_param[156], all_param[161], all_param[166],
                 all_param[171], all_param[176], all_param[181],
                 all_param[186], all_param[191], all_param[196],
                 all_param[201], all_param[206], all_param[211],
                 #all_param[216],
                 all_param[221], all_param[226], all_param[231],
                 all_param[236], all_param[241], all_param[246],
                 all_param[251], all_param[256], all_param[261]]

    xaxis_labels_all = [[7.5e6, 8.0e6],
                        [1e7, 1.05e7], [7.5e6, 7.7e6, 7.9e6], [3.7e6, 3.8e6, 3.9e6],
                        [5.0e6, 5.1e6], [3.1e6, 3.2e6],
                        [1.25e6, 1.28e6], [1.25e6, 1.28e6], [6.2e5, 6.4e5],
                        [2.04e4, 2.06e4], [3.56e3, 3.58e3, 3.60e3], [2.9e3, 3.0e3],
                        [2.0e7, 2.2e7, 2.4e7], [3.0e7, 3.2e7],
                        [1.5e7, 1.6e7], [1.5e7, 1.6e7], [7.5e6, 8.0e6],
                        [1e7, 1.05e7], [7.5e6, 7.7e6, 7.9e6], [3.7e6, 3.8e6, 3.9e6],
                        [5.0e6, 5.1e6], [3.1e6, 3.2e6],
                        [1.25e6, 1.28e6], [1.25e6, 1.28e6], [6.2e5, 6.4e5],
                        [2.04e4, 2.06e4], [3.56e3, 3.58e3, 3.60e3], [2.9e3, 3.0e3],
                        [2.0e7, 2.2e7, 2.4e7], [3.0e7, 3.2e7],
                        [1.5e7, 1.6e7], [1.5e7, 1.6e7], [7.5e6, 8.0e6],
                        [1e7, 1.05e7], [7.5e6, 7.7e6, 7.9e6], [3.7e6, 3.8e6, 3.9e6],
                        [5.0e6, 5.1e6], [3.1e6, 3.2e6],
                        [1.25e6, 1.28e6], [1.25e6, 1.28e6], [6.2e5, 6.4e5],
                        [2.04e4, 2.06e4], [3.56e3, 3.58e3, 3.60e3], [2.9e3, 3.0e3],
                        [5.0e6, 5.1e6], [3.1e6, 3.2e6],
                        [1.25e6, 1.28e6], [1.25e6, 1.28e6], [6.2e5, 6.4e5]]
    degree_plot(layer_all, 'layer', threshold, shape_info, option, xaxis_labels_all)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--option", type=str, dest="option",
                        default="orig", help="plot type: can be orig or ccdf")
    parser.add_argument("--threshold", type=float, dest="threshold",
                        default=0.3, help="fractions to prune the connections")
    args = parser.parse_args()

    main(**vars(args))