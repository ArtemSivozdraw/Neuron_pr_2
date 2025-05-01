class Layer :
    def __init__(self, weigths, biases,output) :
        self.weights = weigths
        self.biases = biases
        self.output = output
    
    @staticmethod
    def create_empty_layer(number_of_neurons, previus_number_of_neuron) :
        biases = [0] * number_of_neurons
        weigths = [[0]*previus_number_of_neuron]*number_of_neurons
        return Layer(weigths,biases, None)
    
    def get_all(self) :
        return self.weights, self.biases, self.output

    def get_length(self) :
        if self.biases :
            return len(self.biases)
        return len(self.output)