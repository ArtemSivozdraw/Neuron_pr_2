from Layer import Layer

class Neuron_Network :
    def __init__(self, activasion, diff_activasion) :
        self.activasion_function = activasion
        self.differential_activasion_function = diff_activasion
        self.layer_array = []
    
    def run_activasion(self,s) :
        return self.activasion_function(s)
    
    def add_inputs(self,inputs) :
        self.layer_array.append(Layer(None,None,inputs))
    
    def add_layer(self, number_of_neuron) :
        previus_layer_lenght = self.layer_array[-1].get_length()
        print(previus_layer_lenght)
        self.layer_array.append(Layer.create_empty_layer(number_of_neuron,previus_layer_lenght))
    
    def get_inputs(self) :
        return self.layer_array[0].output
    
    def get_layer(self,number_of_layer) :
        return self.layer_array[number_of_layer].get_all()