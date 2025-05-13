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


net = Neuron_Network(sigma_activasion, sigma_diff_activasion, 2)

# Приклад 1:
inputs1 = [1.0, 0.0]
expected1 = [1.0, 0.0]

# Приклад 2:
inputs2 = [0.0, 1.0]
expected2 = [0.0, 1.0]

# Загальні масиви:
train_data = [inputs1, inputs2]
train_result = [expected1, expected2]

# Додаємо прихований шар (2 нейрони)
net.add_layer(2)

# Додаємо вихідний шар (2 нейрони)
net.add_layer(2)

# --- Встановлюємо ваги та зсуви вручну (для повторюваності) ---
net.layer_array[1].weigths = [
    [0.1, 0.2],  # weights to h1
    [0.3, 0.4],  # weights to h2
]
net.layer_array[1].biases = [0.0, 0.0]

net.layer_array[2].weigths = [
    [0.5, 0.6],  # weights from h1,h2 to o1
    [0.7, 0.8],  # weights from h1,h2 to o2
]
net.layer_array[2].biases = [0.0, 0.0]

net.calculate_update(train_data,train_result)

for i, layer in enumerate(net.alt_layer_array):
    print(f"\n--- Δ Layer {i+1} ---")
    print("Δ ваги:")
    for row in layer.weigths:
        print(row)
    print("Середнє Δ зсувів:")
    print(layer.biases)
    print("Δ (останній приклад):")
    print(layer.output)

#network.print_layer_array(alt=False)
#network.train_network(train_param,train_result,None,None,epoch=1,step=0.1, valid_cost=0.2)
#network.print_layer_array(alt=False)