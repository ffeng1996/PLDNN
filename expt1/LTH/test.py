import torch
import numpy as np
file = r'/home/eric/Desktop/LTH/Lottery-Ticket-Hypothesis-in-Pytorch/saves/lenet5/mnist/2_model_lt.pth.tar'

net=torch.load(file)
# net: class
para = net.state_dict()
# para: dict, each value in para is a tensor
#print (len(para))

#for k in para.keys():
#    print (k)
#f = open ("outputModel.txt",'w+')
#f2 = open ("modelStr.txt",'w+')
#print (net,file=f2)
#print (net.state_dict(),file=f)

#f3=open("weightTesting,txt",'w+')

#print (para['features.0.weight'],file=f3)
#print(type(para['features.0.weight']))


#print (para['features.0.weight'].shape) #[64,1,3,3]
'''
for t in para['features.0.weight']:
    print ("tensor name:", t)
    print (t.shape) #(1,3,3)
    
    '''

#param_name=['','']

r = np.load("/home/eric/Desktop/LTH/Lottery-Ticket-Hypothesis-in-Pytorch/sparse_cnn_0.7.npz")
for k in r.keys():
    print (k)
    print (r[k].shape)
