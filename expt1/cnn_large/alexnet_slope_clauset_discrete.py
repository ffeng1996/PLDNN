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

   
    p = plt.figure(figsize=(6, 4), dpi=80)
    p3 = p.add_subplot(111)
    p3.set_xscale("log")
    p3.set_yscale("log")
    p3.set_xlim(best_xmin, best_xmax)
    p3.set_ylim(1e-4, 1)
    p3.set_xlabel('degree', fontsize=20)
    p3.set_ylabel('p(X>=x)', fontsize=20)

    p3.plot(np.arange(len(best_counts)) + best_xmin, best_Actual_CCDF, "o", color='b', linewidth=5)
    p3.tick_params(axis="x", which='major', bottom='off', top='off', labelbottom='off')
    p3.tick_params(axis="both", which='minor', labelsize=14)

    p3.set_xticks([best_xmin,best_xmax], minor=True)  # xaxis by hand
    p3.plot(np.arange(best_xmin, best_xmax), S(best_xmin, best_xmax, np.arange(best_xmin, best_xmax), best_a),
            color='r', linewidth=5)

    plt.tight_layout()
    p.savefig(
        'figures/alexnet_s_layer/powerlaw_clauset/{3}_imagenet_{0}_{1}_{2}_{3}.png'.format(option, str1, str2, str3))


def degree_plot(data, name, prune_fraction, shape_info, option, xaxis_labels_all):
    thres = []
    for i in range(len(data)):
        data_current = np.abs(data[i])
        vec_data = data_current.flatten()
        a = int(prune_fraction * data_current.size)
        thres.append(np.sort(vec_data)[a])

    #print thres

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

        if i < len(data) - 1 and len(data_current.shape) == len(
                data_next.shape):  # neglect the conv_fc connection layer nodes
            data_next[data_next <= thres[i + 1]] = 0
            data_next[data_next > thres[i + 1]] = 1
            if len(data_next.shape) > 2:
                next_degree = np.sum(data_next, axis=(0, 2, 3)).astype(int)
                next_degree = next_degree * shape_info[i + 1][2] * shape_info[i + 1][3]
            else:
                next_degree = np.sum(data_next, axis=0).astype(int)

            if next_degree.size < current_degree.size:
                next_degree = np.concatenate((next_degree, next_degree))

            current_degree = current_degree + next_degree
            current_degree = current_degree[np.nonzero(current_degree)]
        plot(current_degree.tolist(), name, str(i + 1), str(prune_fraction).split('.')[1], option, xaxis_labels_all[i])
        degree = degree + current_degree.tolist()


def main(option, threshold):
    import lasagne
    from lasagne.layers import InputLayer
    from lasagne.layers import DenseLayer
    from lasagne.layers import NonlinearityLayer
    from lasagne.layers import DropoutLayer
    from lasagne.layers import Pool2DLayer as PoolLayer
    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
    from lasagne.nonlinearities import softmax
    from lasagne.layers import LocalResponseNormalization2DLayer as LRNLayer

    # define network structure
    net = {}
    net['input'] = InputLayer((None, 3, 227, 227))
    net['conv1'] = ConvLayer(net['input'], num_filters=96, filter_size=(11, 11), stride=4)
    net['norm1'] = LRNLayer(net['conv1'], n=5, alpha=0.0001 / 5.0, beta=0.75, k=1)
    net['pool1'] = PoolLayer(net['norm1'], pool_size=(3, 3), stride=2)

    net['conv2'] = ConvLayer(net['pool1'], num_filters=256, filter_size=(5, 5), pad=2)
    net['norm2'] = LRNLayer(net['conv2'], n=5, alpha=0.0001 / 5.0, beta=0.75, k=1)
    net['pool2'] = PoolLayer(net['norm2'], pool_size=(3, 3), stride=2)

    net['conv3'] = ConvLayer(net['pool2'], num_filters=384, filter_size=(3, 3), pad=1)
    net['conv4'] = ConvLayer(net['conv3'], num_filters=384, filter_size=(3, 3), pad=1)
    net['conv5'] = ConvLayer(net['conv4'], num_filters=128, filter_size=(3, 3), pad=1)
    net['pool5'] = PoolLayer(net['conv5'], pool_size=(3, 3), stride=2)

    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc7'] = DenseLayer(net['fc6'], num_units=4096)
    net['fc8'] = DenseLayer(net['fc7'], num_units=1000)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)


    import caffe
    net_caffe = caffe.Net('bvlc_alexnet_deploy.prototxt', 'alexnet.caffemodel', caffe.TEST)
    layers_caffe = dict(zip(list(net_caffe._layer_names), net_caffe.layers))

    for name, layer in net.items():
        try:
            layer.W.set_value(layers_caffe[name].blobs[0].data)
            layer.b.set_value(layers_caffe[name].blobs[1].data)
        except AttributeError:
            continue

    all_param = lasagne.layers.get_all_param_values(net['prob'])

    #print("Shape of each layer: ")
    #for i, value in enumerate(all_param):
     #   print value.shape

    #print("Sparsity of each layer: ")
    #for i, value in enumerate(all_param):
     #   if i % 2 == 0:
     #       sparsity = np.float(np.nonzero(all_param[i])[0].size) / np.prod(value.shape)
     #       print sparsity

    layers = [net['conv1'], net['conv2'], net['conv3'], net['conv4'], net['conv5']]
    shape_info = lasagne.layers.get_output_shape(layers)

    layer_all = [all_param[0], all_param[2], all_param[4], all_param[6], all_param[8], all_param[10],
                 all_param[12], all_param[14]]

    xaxis_labels_all = [[2.6e6, 2.8e6, 3.0e6], [5e5, 6e5, 7e5], [3.4e5, 3.6e5, 3.8e5], [2.4e5, 2.6e5],
                        [1.1e5, 1.2e5], [1.15e3, 1.2e3, 1.25e3], [6.4e2, 6.8e2], [1e3, 1.1e3]]

    degree_plot(layer_all, 'layer', threshold, shape_info, option, xaxis_labels_all)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--option", type=str, dest="option",
                        default="orig", help="plot type: can be orig or ccdf")
    parser.add_argument("--threshold", type=float, dest="threshold",
                        default=0.0, help="fractions to prune the connections")
    args = parser.parse_args()

    main(**vars(args))