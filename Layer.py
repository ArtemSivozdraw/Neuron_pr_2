class Layer :
    def __init__(self, weigths, biases,output) :
        self.weigths = weigths
        self.biases = biases
        self.output = output
    
    @staticmethod
    def create_empty_layer(number_of_neurons, previus_number_of_neuron, weights_filler, numaration) :
        biases = [0] * number_of_neurons
        output = [None] * number_of_neurons
        weigths = []
        s=0
        for _ in range (number_of_neurons) :
            temp = []
            for i in range(previus_number_of_neuron) :
                temp.append(weights_filler+s*numaration)
                s+=0.01
            weigths.append(temp)
        #weigths = [[weights_filler]*previus_number_of_neuron]*number_of_neurons
        return Layer(weigths,biases, output)
    
    def get_all(self) :
        return self.weigths, self.biases, self.output

    def get_length(self) :
        if self.biases :
            return len(self.biases)
        return len(self.output)