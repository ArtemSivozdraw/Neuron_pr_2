from Neuron import Neuron_Network
import math


def activasion(s) :
    return 6 / (1 + math.exp(-s))

def diff_activasion(a) :
    s = a / 6
    return 6 * s * (1 - s)

time_array = [2.57, 4.35, 1.27, 5.46, 1.30, 4.92, 1.31, 4.14, 1.97, 5.67, 0.92, 4.76, 1.72, 4.44, 1.49]

training_param,training_results = [],[]

for i in range(len(time_array)-5) :
    training_param.append(time_array[i:i+3])
    training_results.append([time_array[i+3]])

print(training_param)
print(training_results)
network = Neuron_Network(activasion, diff_activasion,3)
network.add_layer(1)

network.print_layer_array()
network.train_network(training_param,training_results,None,None,80000,0.001)
network.print_layer_array()