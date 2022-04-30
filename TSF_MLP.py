# -*- coding: utf-8 -*-

import numpy as np
import random 

#range [0, 200]
def sigmoid_func(x, sigma):
    return 200 * (1 / (1 + np.exp(-1 * sigma * x)))

#range [0, 200]
def derivative_sigmoid_func(x, sigma):
    return sigma * sigmoid_func(x, sigma) * (200 - sigmoid_func(x, sigma))

def relu_func(x):
     if x > 0:
         return x
     else:
         return 0
    
def derivative_relu_func(x):
     if x > 0:
         return 1
     else:
         return 0
    
def random_weight_generator(input_dense, output_dense):
    weights = []
    for i in range(output_dense):
        child = []
        for j in range(input_dense + 1):
            child.append(round(random.uniform(0, 0.5), 4))
        weights.append(child)    
    return weights

def MLP(input_dense, hidden_layer_dense, train_set, alpha, activation, loop_range):
    roundTo = 5
    # v 
    first_layer_weights = random_weight_generator(input_dense, hidden_layer_dense)
    # w
    second_layer_weights = random_weight_generator(hidden_layer_dense, 1)

    for loop in range(0, loop_range):
        for ith_set in range(0, len(train_set)):
            z_in = []
            for i in range(0, len(first_layer_weights)):
                sum_on_weights = 0;
                for j in range(0, len(first_layer_weights[0])):
                    sum_on_weights += first_layer_weights[i][j] * train_set[ith_set][j]
                
                z_in.append(sum_on_weights)
            
            z_out = [1] #set bias value inside first index
            for i in range(0, len(z_in)):
                if activation['func'] == 'sigmoid':
                    z_out.append(sigmoid_func(z_in[i], activation['sigma']))
                if activation['func'] == 'relu':
                    z_out.append(relu_func(z_in[i]))
            
            y_in = []
            for i in range(0, len(second_layer_weights)):
                sum_on_weights = 0;
                for j in range(0, len(second_layer_weights[0])):
                    sum_on_weights += second_layer_weights[i][j] * z_out[j]
                
                y_in.append(sum_on_weights)
                            
            y_out = []
            for i in range(0, len(y_in)):
                if activation['func'] == 'sigmoid':
                    y_out.append(sigmoid_func(y_in[i], activation['sigma']))
                if activation['func'] == 'relu':
                    y_out.append(relu_func(y_in[i]))
        
            delta_k = []
            targetIndex = len(train_set[0]) - 1
            for k in range(0, len(y_out)):
                if activation['func'] == 'sigmoid':
                    delta_k.append((train_set[ith_set][targetIndex]-y_out[k]) * derivative_sigmoid_func(y_in[k], activation['sigma']))
                if activation['func'] == 'relu':
                    delta_k.append((train_set[ith_set][targetIndex]-y_out[k]) * derivative_relu_func(y_in[k]))    
            
            delta_in_j = []
            for j in range(1, len(z_out)):
                sigmaDeltaOnK = 0
                for k in range(0, len(delta_k)):
                    sigmaDeltaOnK += delta_k[k] * second_layer_weights[k][j]
                
                delta_in_j.append(sigmaDeltaOnK)
                        
            delta_out_j = []
            for j in range(0, len(delta_in_j)):
                if activation['func'] == 'sigmoid':
                    delta_out_j.append(delta_in_j[j] * derivative_sigmoid_func(z_in[j], activation['sigma']))
                if activation['func'] == 'relu':
                    delta_out_j.append(delta_in_j[j] * derivative_relu_func(z_in[j]))
            
            # change wieghts:
            for i in range(0, len(first_layer_weights[0])):
                for j in range(0, len(first_layer_weights)):
                    first_layer_weights[j][i] += alpha * delta_out_j[j] * train_set[ith_set][i]
                        
            for i in range(0, len(second_layer_weights[0])):
                for j in range(0, len(delta_k)):
                    second_layer_weights[j][i] += alpha * delta_k[j] * z_out[i]
        
    for i in range(0, len(first_layer_weights)):
        for j in range(0, len(first_layer_weights[i])):
            first_layer_weights[i][j] = round(first_layer_weights[i][j], roundTo)
            
    for i in range(0, len(second_layer_weights)):
        for j in range(0, len(second_layer_weights[i])):
            second_layer_weights[i][j] = round(second_layer_weights[i][j], roundTo)
            
    return {
            'first_layer': first_layer_weights, 
            'hidden_layer': second_layer_weights,
            'activation': activation
            }

   
def check_ANN(test_set, weights):
    roundTo = 5
    
    if len(test_set) != len(weights['first_layer'][0]) - 1:
        return 'This MLP network has '+ str(len(weights['first_layer'][0]) - 1) + ' nodes. Your Test-Set has ' + str(len(test_set)) + ' nodes.'
    
    hidden_nodes = [1]
    test_set.insert(0, 1)
    first_layer_weights = weights['first_layer']
    second_layer_weights = weights['hidden_layer']
    activation = weights['activation']
    for i in range(0, len(first_layer_weights)):
        input_pulse = 0;
        for j in range(0, len(test_set)):
            input_pulse += first_layer_weights[i][j] * test_set[j]
        
        if activation['func'] == 'sigmoid':
            hidden_nodes.append(sigmoid_func(input_pulse, activation['sigma']))
        if activation['func'] == 'relu':
            hidden_nodes.append(relu_func(input_pulse))
    
    input_pulse = 0
    for i in range(0, len(hidden_nodes)):
        input_pulse += second_layer_weights[0][i] * hidden_nodes[i]
    
    if activation['func'] == 'sigmoid':
        return round(sigmoid_func(input_pulse, activation['sigma']), roundTo)
    if activation['func'] == 'relu':
        return round(relu_func(input_pulse), roundTo)
    
    
    
#train set -> [bias, x_1, x_2, ..., X_i, target]
# =============================================================================
# sample = [
#          [1, 1, 1, 0],
#          [1, 1, 0, 1],
#          [1, 0, 1, 1],
#          [1, 0, 0, 0]
#          ]
# =============================================================================

sample = [
         [1, 10, 20, 30, 40],
         [1, 20, 30, 40, 50],
         [1, 30, 40, 50, 60],
         [1, 40, 50, 60, 70],
         [1, 50, 60, 70 ,80],
         [1, 60, 70, 80, 90]        
         ]

alpha = 1
    
activation_sigmoid = {
        'func': 'sigmoid',
        'sigma': 1000
        }

activation_relu = {
        'func': 'relu'
        }

weights = MLP(3, 100, sample, alpha, activation_sigmoid, 2000)
print('predict: ' + str(check_ANN([70, 80, 90], weights)))
