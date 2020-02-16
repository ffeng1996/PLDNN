#### set xticks by hand

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from argparse import ArgumentParser
from scipy import optimize

from matplotlib import ticker
import powerlaw
#import pylab

#pylab.rcParams['xtick.major.pad'] = '16'
#pylab.rcParams['ytick.major.pad'] = '16'
# pylab.rcParams['font.sans-serif']='Arial'
from scipy.optimize import minimize
from matplotlib import rc

rc('font', family='sans-serif')
rc('font', size=16.0)
rc('text', usetex=False)


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
    degree = []
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
        D = np.max(np.abs(CCDF_diff))  ### this is the same as the cdf diff
        if D > KS:
            num_larger = num_larger + 1

        p_value = num_larger / N

    return p_value


def plot(degree, option, num_layers, hidden_units, layer, adjust_axis, xaxis_labels, xaxis_tick_labels):
    degree = sorted(degree, reverse=True)

    KS_max = 1
    cut_fraction1 = 0.30
    cut_fraction2 = 0.30
    cutting_number1 = int(len(
        degree) * cut_fraction1)  ### choose xmin in the smallest xmin_fraction degrees, and xmax in the largest xmax_fraction degrees
    cutting_number2 = int(len(degree) * cut_fraction2)
    for i in range(2, cutting_number1):  ## iterate for xmin
        for j in range(cutting_number2):  ### iterate for xmax
            x = degree[j:-i + 1]  ##### from large to small
            xmin = min(x)
            xmax = max(x)

            xmax_for_fitting = xmax
            # get the actual results
            bins = np.arange(min(x) - 0.5, max(x) + 1, 1)
            h, bins = np.histogram(x, density=True, bins=bins)
            counts = h * len(x)
            counts = np.cumsum(counts[::-1])[::-1]
            Actual_CCDF = counts / counts[0]

            # get the fitting results
            init_params = 10
            results = minimize(precise(xmin, xmax_for_fitting, x), init_params, method='L-BFGS-B',
                               bounds=((1.001, 10),))

            a = results.x[0]
            # Theoretical_CCDF = S(xmin, xmax, np.arange(min(x), max(x)+1), a)
            Theoretical_CCDF = S(xmin, xmax_for_fitting, np.arange(xmin, xmax + 1), a)
            # get KS
            CCDF_diff = Theoretical_CCDF - Actual_CCDF
            D = np.max(np.abs(CCDF_diff))  ### this is the same as the cdf diff

            if D < KS_max:
                KS_max = D
                best_xmin = xmin
                best_xmax = xmax
                best_xmax_for_fitting = xmax_for_fitting
                # best_x = x
                best_a = a
                best_counts = counts
                best_Actual_CCDF = Actual_CCDF
                # best_Theoretical_CCDF = Theoretical_CCDF
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

    p3.set_xticks(xaxis_labels, minor=True)  ### xaxis by hand
    p3.set_xticklabels(xaxis_tick_labels, minor=True)  ### xaxis by hand

    # p3.plot(np.arange(best_xmin, best_xmax), S(best_xmin, best_xmax,np.arange(best_xmin, best_xmax), best_a), color='r',linewidth=5)  #############
    p3.plot(np.arange(best_xmin, best_xmax),
            S(best_xmin, best_xmax_for_fitting, np.arange(best_xmin, best_xmax), best_a), color='r',
            linewidth=5)  #############

    plt.tight_layout()
    # plt.subplots_adjust(top=0.9)
    p.savefig(
        'figures/layer/powerlaw_clauset/mnist_{0}_{1}_{2}_{3}.png'.format(option, num_layers, hidden_units, layer),
        bbox_inches='tight')


def degree_plot(data, hidden_units, num_layers, shape_info, option, adjust_axis, xaxis_labels_all,
                xaxis_tick_labels_all):
    degree = []
    for i in range(len(data)):
        data_current = np.abs(data[i])
        if i < len(data) - 1:
            data_next = np.abs(data[i + 1])
        data_current[data_current > 0.] = 1
        # input nodes
        if i < 1:
            first_degree = np.sum(data_current, axis=1).astype(int)
            first_degree = first_degree[np.nonzero(first_degree)]
            # import ipdb; ipdb.set_trace()
            plot(first_degree.tolist(), option, str(num_layers), hidden_units, str(i), adjust_axis, xaxis_labels_all[i],
                 xaxis_tick_labels_all[i])
            degree = degree + first_degree.tolist()

        # hidden nodes
        current_degree = np.sum(data_current, axis=0).astype(int)

        if i < len(data) - 1 and len(data_current.shape) == len(data_next.shape):
            data_next[data_next > 0.0] = 1
            next_degree = np.sum(data_next, axis=1).astype(int)
            current_degree = current_degree + next_degree

        current_degree = current_degree[np.nonzero(current_degree)]
        plot(current_degree.tolist(), option, str(num_layers), hidden_units, str(i + 1), adjust_axis,
             xaxis_labels_all[i + 1], xaxis_tick_labels_all[i + 1])
        degree = degree + current_degree.tolist()


def main(filename, option, num_layers, hidden_units, adjust_axis):
    with np.load(filename) as f:
        param_values_sparse = [f['arr_%d' % i] for i in range(len(f.files))]

    shape_info = []
    for i in param_values_sparse:
        shape_info.append(i.shape)

    # this is for (30,30)
    if "_8_" in filename:
        xaxis_labels_all = [[2e2, 4e2], [2e2, 4e2, 6e2], [2e2, 4e2]]
        xaxis_tick_labels_all = [[r'$2 \times 10^2$', r'$4 \times 10^2$'],
                                 [r'$2 \times 10^2$', r'$4 \times 10^2$', r'$6 \times 10^2$'],
                                 [r'$2 \times 10^2$', r'$4 \times 10^2$']]
    else:
		xaxis_labels_all = [[8e1, 1e2, 1.2e2], [1.6e2, 1.8e2, 2e2], [1e2, 1.2e2]]
		xaxis_tick_labels_all = [[r'$8 \times 10^1$', r'$1 \times 10^2$', r'$1.2 \times 10^2$'],
								 [r'$1.6 \times 10^2$', r'$1.8 \times 10^2$', r'$2 \times 10^2$'],
								 [r'$1 \times 10^2$', r'$1.2 \times 10^2$']]

    # plot except the last layer
    degree_plot([param_values_sparse[0], param_values_sparse[2]],
                str(hidden_units), num_layers, shape_info, option, adjust_axis, xaxis_labels_all, xaxis_tick_labels_all)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--filename", type=str, dest="filename",
                        default='model/sparse_model_0.9_2_1024.npz')
    parser.add_argument("--option", type=str, dest="option",
                        default="orig", help="orig or ccdf")
    parser.add_argument("--num_layers", type=int, dest="num_layers",
                        default=2, help="number of layers")
    parser.add_argument("--hidden_units", type=int, dest="hidden_units",
                        default=1024, help="number of hidden units in each layer")
    parser.add_argument("--adjust_axis", type=str, dest="adjust_axis",
                        default="y", help="whether to adjust axis when plotting")
    # parser.add_argument("--starting_node",  type=int, dest="starting_node",
    # 			default=0, help="help decide to diagard some beginning nodes in the ccdf plot")
    # parser.add_argument("--end_node",  type=int, dest="end_node",
    # 			default=0, help="help decide to diagard some ending nodes in the ccdf plot")
    args = parser.parse_args()

    main(**vars(args))
