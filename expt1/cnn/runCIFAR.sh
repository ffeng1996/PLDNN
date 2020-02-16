## train a dense network for cifar
python cifar_dense.py
## prune 70% connections
python cifar_sparse.py --num_epochs=100 --prune_fraction=0.7
# plot the ccdf plot
python degree_plot_slope_cifar_clauset_discrete.py --option="ccdf"