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


def plot(degree, option, layer, adjust_axis, xaxis_labels):
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
            results = minimize(precise(xmin, xmax, x), init_params, method='L-BFGS-B', bounds=((1.001, 10),))
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

    print min(degree), max(degree), best_xmin, best_xmax, best_a, best_num_x, p_value

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

    p3.set_xticks(xaxis_labels, minor=True)  # xaxis by hand
    p3.plot(np.arange(best_xmin, best_xmax), S(best_xmin, best_xmax, np.arange(best_xmin, best_xmax), best_a),
            color='r', linewidth=5)

    plt.tight_layout()
    p.savefig('cifar/figures/powerlaw_clauset/cifar05_{0}_{1}.png'.format(option, layer), bbox_inches='tight')


def degree_plot(data, shape_info, option, adjust_axis, xaxis_labels_all):
    degree = []
    for i in range(len(data)):
        data_current = np.abs(data[i])
        if i < len(data) - 1:
            data_next = np.abs(data[i + 1])

        data_current[data_current > 0.] = 1
        if len(data_current.shape) > 2:
            current_degree = np.sum(data_current, axis=(1, 2, 3)).astype(int)
            current_degree = current_degree * shape_info[i][2] * shape_info[i][3]
        else:
            current_degree = np.sum(data_current, axis=0).astype(int)
        if i < len(data) - 1 and len(data_current.shape) == len(data_next.shape):
            data_current[data_current > 0.0] = 1
            if len(data_next.shape) > 2:
                next_degree = np.sum(data_next, axis=(0, 2, 3)).astype(int)
                next_degree = next_degree * shape_info[i + 1][2] * shape_info[i + 1][3]
            else:
                next_degree = np.sum(data_next, axis=1).astype(int)

            current_degree = current_degree + next_degree
            current_degree = current_degree[np.nonzero(current_degree)]

        plot(current_degree.tolist(), option, str(i + 1), adjust_axis, xaxis_labels_all[i])
        degree = degree + current_degree.tolist()


def main(filename, option, adjust_axis):
    import lasagne
    from lasagne.layers import InputLayer
    from lasagne.layers import DenseLayer
    from lasagne.layers import NonlinearityLayer
    from lasagne.layers import Pool2DLayer as PoolLayer
    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
    from lasagne.nonlinearities import softmax
    # define network structure
    net = {}
    net['input'] = InputLayer((None, 3, 32, 32))
    net['conv1'] = ConvLayer(net['input'], 32, 3, pad=1)
    net['conv2'] = ConvLayer(net['conv1'], 32, 3, pad=1)
    net['pool1'] = PoolLayer(net['conv1'], 2)
    net['conv3'] = ConvLayer(net['pool1'], 64, 3, pad=1)
    net['conv4'] = ConvLayer(net['conv3'], 64, 3, pad=1)
    net['pool2'] = PoolLayer(net['conv4'], 2)
    net['fc5'] = DenseLayer(net['pool2'], num_units=512)
    net['fc6'] = DenseLayer(net['fc5'], num_units=10, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc6'], softmax)

    with np.load(filename) as f:
        param_values_sparse = [f['arr_%d' % i] for i in range(len(f.files))]

    layers = [net['conv1'], net['conv2'], net['conv3'],net['conv4'],net['fc5']]
    shape_info = lasagne.layers.get_output_shape(layers)
    # plot except the last layer

    testList = [param_values_sparse[0], param_values_sparse[2], param_values_sparse[4], param_values_sparse[6],param_values_sparse[8]]
    print "#############################"
    print testList[0].shape
    print "#############################"
    print testList[1].shape
    print "#############################"
    print testList[2].shape
    print "#############################"
    print testList[3].shape
    print "#############################"
    print testList[4].shape
    print "#############################"
    '''
print testList[5].shape
    print "#############################"
    print testList[6].shape
    print "#############################"
    print testList[7].shape
    print "#############################"
    print testList[8].shape
    print "#############################"
    print testList[9].shape
    print "#############################"
    print testList[10].shape
'''
    xaxis_labels_all = [[9.0e4, 1.0e5], [1.3e5, 1.4e5], [6.5e4, 7.0e4, 7.5e5],[4.0e4, 6.0e4, 8.0e4],[1.24e3, 1.3e3, 1.4e3]]
    xaxis_labels_all = [[2048, 7168], [10752, 25856], [2816, 12288],[2048,12544],[163,299]]
    degree_plot([param_values_sparse[0],param_values_sparse[2], param_values_sparse[4], param_values_sparse[6],param_values_sparse[8]],
                shape_info, option, adjust_axis, xaxis_labels_all)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--filename", type=str, dest="filename",
                        default='cifar/model/sparse_cnn_0.95.npz')
    parser.add_argument("--option", type=str, dest="option",
                        default="orig", help="orig or ccdf")
    parser.add_argument("--adjust_axis", type=str, dest="adjust_axis",
                        default="y", help="whether to adjust axis when plotting")
    args = parser.parse_args()

    main(**vars(args))
