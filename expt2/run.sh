## train a dense network
# python mnist_dense.py custom_mlp:2,1024,0.0,0.0 200
## prune 90% connections and get a sparse network
# python mnist_sparse.py custom_mlp:2,1024,0.0,0.0 200 0.9
# add anohter 10% coonnections and fine-tune on task B.
python mnist_cont.py --model="custom_mlp:2,1024,0.0,0.0" --num_epochs=10 --add_fraction=0.1 --permute_size=8 --sparsity=0.9

# plot the difference
python diff_plot_layer.py --filename="model/sparse_model_0.9_2_1024.npz" --filename1="cont_model/cont_sparse_fc_0.1_0.9_8_2_1024.npz" --hidden_units=1024 --permute_size=8

# plot the ccdf and fitting
python degree_plot_clauset_discrete.py --filename="model/sparse_model_0.9_2_1024.npz" --option="ccdf" --hidden_units=1024

# plot ccdf for the second task
python degree_plot_clauset_discrete.py --filename="cont_model/cont_sparse_fc_0.1_0.9_8_2_1024.npz" --option="ccdf" --hidden_units=1024