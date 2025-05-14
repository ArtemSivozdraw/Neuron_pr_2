from Neuron import Neuron_Network
#from sklearn import (datasets, model_selection)
import math


def sigma_activasion(s) :
    return 1 / (1 + math.exp(-s))

def sigma_diff_activasion(a) :
    return  a * (1 - a)

def RelU_activasion(s) :
    if s > 0 :
        return s
    else :
        return 0
    
def RelU_diff_activasion(a) :
    if a > 0 :
        return 1
    else :
        return 0

def leaky_relu(x, alpha=0.01):
    return x if x > 0 else alpha * x

def diff_leaky_relu(x, alpha=0.01):
    return 1 if x > 0 else alpha

filename = "image_set.txt"
train_param = []
train_result = []
with open(filename) as file :
    for image_number in range (5) :
        temp = []
        for line in range (6) :
            temp+=(list(map(int,file.readline()[:-1].split(" "))))
        train_param.append(temp)
        train_result.append(list(map(int,file.readline()[:-1].split(" "))))


network = Neuron_Network(sigma_activasion, sigma_diff_activasion, 36)

network.add_layer(10)
network.add_layer(5)


network.train_network(train_param,train_result,train_param,train_result,epoch=50000,step=0.01, valid_cost=0.01)
