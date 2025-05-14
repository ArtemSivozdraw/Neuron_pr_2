import numpy as np


class Layer :
    def __init__(self, weigths, biases,output) :
        self.weigths = weigths
        self.biases = biases
        self.output = output
    
    @staticmethod
    def create_empty_layer(number_of_neurons, previus_number_of_neuron, weights_filler, numaration) :
        biases = np.random.uniform(-1, 1, number_of_neurons).tolist()
        output = [None] * number_of_neurons
        weigths = []

        for _ in range (number_of_neurons) :
            temp = np.random.uniform(-1, 1, previus_number_of_neuron).tolist()
            weigths.append(temp)

        return Layer(weigths,biases, output)
    
    def get_all(self) :
        return self.weigths, self.biases, self.output

    def get_length(self) :
        if self.biases :
            return len(self.biases)
        return len(self.output)