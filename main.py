from Neuron import Neuron_Network
    
def activasion(s) :
    return s * 20

def diff_activasion(s) :
    return s/20

network = Neuron_Network(activasion, diff_activasion)
network.add_inputs([1,2,3,4,5])
network.add_layer(2)
print(network.get_layer(1))