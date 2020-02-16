## train a dense cnn model for mnist
python mnist_dense.py
## prune 70% connections
python mnist_sparse.py --num_epochs=200 --prune_fraction=0.7
# plot the ccdf plot
python degree_plot_slope_clauset_discrete.py --option="ccdf"

