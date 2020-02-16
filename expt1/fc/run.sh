## train a dense network
# python mnist_dense.py custom_mlp:2,1024,0.0,0.0 200
## prune 90% connections and get a sparse network
# python mnist_sparse.py custom_mlp:2,1024,0.0,0.0 200 0.9

# plot the ccdf and fitting
python degree_plot_clauset_discrete.py --filename="model/sparse_model_0.9_2_1024.npz" --option="ccdf" --hidden_units=1024
