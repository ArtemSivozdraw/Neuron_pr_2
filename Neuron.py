from Layer import Layer

class Neuron_Network :
    def __init__(self, activasion, diff_activasion, number_of_inputs) :
        self.activasion_function = activasion
        self.differential_activasion_function = diff_activasion
        self.layer_array = [Layer(None, None, [0]*number_of_inputs)]
    
    def run_activasion(self,s) :
        return self.activasion_function(s)
    
    def run_dif_activasion(self,a) :
        return self.differential_activasion_function(a)
    
    def set_inputs(self,inputs) :
        self.layer_array[0]=Layer(None,None,inputs)
    
    def add_layer(self, number_of_neuron) :
        previus_layer_lenght = self.layer_array[-1].get_length()
        self.layer_array.append(Layer.create_empty_layer(number_of_neuron,previus_layer_lenght,len(self.layer_array),1))
    
    def get_inputs(self) :
        return self.layer_array[0].output
    
    def get_layer(self,number_of_layer) :
        return self.layer_array[number_of_layer].get_all()
    
    def get_layer_array_frame(self) :
        alt = []
        for i in range (1, len(self.layer_array)) :
            alt.append(Layer.create_empty_layer(len(self.layer_array[i].weigths),len(self.layer_array[i].weigths[0]), 0, 0))
        return alt

    def calculate_output(self) :

        for layer in self.layer_array :
            if layer.biases == None :
                inputs = layer.output 
                continue
            
            #print(inputs)
            for i in range (layer.get_length()) :
                sum = 0
                for j in range (len(inputs)) :
                    sum += inputs[j] * layer.weigths[i][j]
                sum+=layer.biases[i]
                layer.output[i] = self.run_activasion(sum)
            
            inputs = layer.output
            #print(f"Вихід нейронів {self.layer_array.index(layer)} шару ")
            #print(layer.get_all())

        return inputs
    
    def calculate_cost(self, actuall, expected) :
        if len(actuall) != len(expected) :
            raise TypeError(f"Очікується кількість виходів {len(expected)} : {expected}. Подається на вхід {len(actuall)} : {actuall}")
        cost = 0
        for i in range (len(actuall)) :
            cost += (actuall[i] - expected[i])**2
        return cost
    
    def calculate_output_delta(self, result) :
        for i in range (len(self.alt_layer_array[-1].biases)) :
            dif_cost = 2 * (self.layer_array[-1].output[i] - result[i])
            dif_func_activasion = self.run_dif_activasion(self.layer_array[-1].output[i])
            #print(f"2 * ({self.layer_array[-1].output[i]}-{result[i]}) * {self.layer_array[-1].output[i]} * (1-{self.layer_array[-1].output[i]})")
            self.alt_layer_array[-1].output[i] = dif_cost * dif_func_activasion
            self.alt_layer_array[-1].biases[i] += self.alt_layer_array[-1].output[i]

    def calculate_hiden_delta(self) :
        if len(self.layer_array) == 2 :
            return

        for i in range (len(self.layer_array)-2,0,-1) :
            for j in range (len(self.layer_array[i].biases)) :
                self.alt_layer_array[i-1].output[j] = 0
                weigths,deltas = self.get_deltaXweight_array(i,j)
                diff_a = self.run_dif_activasion(self.layer_array[i].output[j])
                for k in range (len(weigths)) :
                    self.alt_layer_array[i-1].output[j] += weigths[k] * deltas[k] * diff_a

                self.alt_layer_array[i-1].biases[j] += self.alt_layer_array[i-1].output[j]
        
    def get_deltaXweight_array (self,layer_index,neuron_index) :
        weights_array = []
        delta_array = []
        for i in range (len(self.layer_array[layer_index+1].weigths)) :
            weights_array.append(self.layer_array[layer_index+1].weigths[i][neuron_index])
            delta_array.append(self.alt_layer_array[layer_index].biases[i])
        
        return weights_array, delta_array

    def devine_alt_array_by(self, number) :
        for i in range(len(self.alt_layer_array)) :
            for j in range (len(self.alt_layer_array[i].weigths)) :
                for k in range (len(self.alt_layer_array[i].weigths[j])) :
                    self.alt_layer_array[i].weigths[j][k] /= number
            for j in range (len(self.alt_layer_array[i].biases)) :
                self.alt_layer_array[i].biases[j] /= number

    def calculate_weigths(self) :
        for i in range (len(self.alt_layer_array)) :
            for j in range (len(self.alt_layer_array[i].weigths)) :
                for k in range (len(self.alt_layer_array[i].weigths[j])) :
                    self.alt_layer_array[i].weigths[j][k] += self.alt_layer_array[i].output[j] * self.layer_array[i+1].weigths[j][k]

    def calculate_update(self, data, result) :
        avrg_cost = 0
        self.alt_layer_array = self.get_layer_array_frame()
        self.set_inputs(data[0])
        for i in range (len(result)) :
            if len(data[i]) != len(self.layer_array[0].output) :
                raise TypeError("Невірна кількість нейронів вхідного шару")
            
            network_output = self.calculate_output()
            cost = self.calculate_cost(network_output,result[i])
            avrg_cost+=cost

            self.calculate_output_delta(result[i])

            self.calculate_hiden_delta()

            self.calculate_weigths()

        
        avrg_cost /= len(result)
        self.devine_alt_array_by(len(result))

        return avrg_cost

    def update_network(self, step) :
        for i in range (len(self.alt_layer_array)) :
            for j in range (len(self.alt_layer_array[i].weigths)) :
                for k in range (len(self.alt_layer_array[i].weigths[j])) :
                    self.layer_array[i+1].weigths[j][k] -= step * self.alt_layer_array[i].weigths[j][k]
            for j in range (len(self.alt_layer_array[i].biases)) :
                self.layer_array[i+1].biases[j] -= step * self.alt_layer_array[i].biases[j]

    def train_network(self, train_data, train_result, test_data, test_result, epoch, valid_cost) :

        for current in range (epoch) :
            avarage_cost = self.calculate_update(train_data,train_result)
            self.update_network(0.1)
            print(f"!!! Cost = {avarage_cost} !!!")