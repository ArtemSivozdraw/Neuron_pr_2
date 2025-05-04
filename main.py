from Neuron import Neuron_Network
import math


def activasion(s) :
    return 1/(1 + math.exp(-s))

def diff_activasion(a) :
    return a * (1-a)

network = Neuron_Network(activasion, diff_activasion)
network.set_inputs([2.57,4.35,1.27])
network.add_layer(1)

network.train_network([[2.57,4.35,1.27]],[[0.5]],None,None,1,0.1)