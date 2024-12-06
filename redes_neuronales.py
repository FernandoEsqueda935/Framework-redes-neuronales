import numpy
import functions
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
import scipy.io
import pandas as pd 

#clase capa, es un objeto capa que contiene:
# pesos aumentados de la capa
# la funcion de activacion de la capa
# la derivada de la funcion de activacion de la capa
# el tama;o de los valores de entrada a esa capa
# El valor de n de la capa
# el valor de la salida de la capa
# el valor de delta de la capa
# el valor del gradiente de esa capa. 

#este objeto estara contenido en un array de capas para formar una red neuronal
class layer:
    def __init__(self, neurons, input_size, activation_function, derivative_activation_function):
        #
        self.weights = numpy.random.rand(neurons, input_size + 1)
        self.activation_function = activation_function
        self.derivative_activation_function = derivative_activation_function
        self.inputs = 0
        self.n = 0
        self.a = 0
        self.delta = []
        self.gradient = []
        

#clase de la red neuronal, creara un objeto en el cual habran diferentes metodos a utilizar
#el plan es que con esos metodos obtener el gradiente de la red y optimizarla con un optimizador externo 
class network_architecture:
    def __init__(self,classifier, neurons_layers, input_size, activation_functions, derivate_activation_functions, dropout):
        self.classifier = classifier
        self.layers = self.calculate_layers(neurons_layers, input_size, activation_functions, derivate_activation_functions)
        self.output = 0
        self.e = 0
        self.inputs = 0
        self.MSE = 0
        self.dropout_value = dropout

    def cross_entropy_categorical(self, target):
        epsilon = 1e-15
        predictions = numpy.clip(self.output, epsilon, 1 - epsilon).T
        loss = -numpy.mean(numpy.sum(target.T * numpy.log(predictions), axis=0))
        return loss
    
    def cross_entropy_binary(self, target):
        epsilon = 1e-15 
        predictions = numpy.clip(self.output, epsilon, 1 - epsilon)
        loss = -numpy.mean(target.T * numpy.log(predictions) + (1 - target.T) * numpy.log(1 - predictions))
        return loss

    def dropout(self, p, a):
        mask = numpy.random.binomial(1, p, size=a.shape)
        return a * mask
        

    def calculate_layers(self, neurons_layers, input_size, activation_functions, derivate_activation_functions):
        layers = []
        if self.classifier == 0:
            last_neurons = input_size
            for neurons, activation_function, derivate_activation_function in zip(neurons_layers, activation_functions, derivate_activation_functions):
                layers.append(layer(neurons, last_neurons, activation_function, derivate_activation_function))
                last_neurons = neurons
        else:
            last_neurons = input_size
            for neurons, activation_function, derivate_activation_function in zip(neurons_layers, activation_functions, derivate_activation_functions):
                layers.append(layer(neurons, last_neurons, activation_function, derivate_activation_function))
                last_neurons = neurons
            if self.classifier == 2:
                layers[len(layers)-1].activation_function = functions.sigmoid
            else:
                layers[len(layers)-1].activation_function = functions.softmax
        return layers
    
    def foward_propagation(self, inputs, target):
        
        self.inputs = inputs
        inputs_h = numpy.vstack((inputs.T, numpy.ones((1, inputs.T.shape[1]))))

        cnt_layer = 0
        for layer in self.layers:
            n= layer.weights @ inputs_h
            a = layer.activation_function(n)
            layer.n = n
            layer.a = a
            
            if self.dropout_value != 0:
                if cnt_layer < len(self.layers)-1:
                    layer.a = self.dropout(self.dropout_value, a)

            inputs_h = numpy.vstack((a, numpy.ones((1, a.shape[1]))))

            cnt_layer += 1
        self.output = a

        if self.classifier == 0:
            self.e =  target.T - self.output 
        elif self.classifier == 2:
            self.e = self.output - target.T
            self.MSE = self.cross_entropy_binary(target.T)
        else:
            self.e = self.output - target.T
            self.MSE = self.cross_entropy_categorical(target.T)

        

    def back_propagation(self):

        for layer_index in reversed(range(len(self.layers))):
            
            if layer_index == len(self.layers)-1:
                if self.classifier == 0:
                    self.layers[layer_index].delta = numpy.array(-2 * numpy.multiply(self.e , self.layers[layer_index].derivative_activation_function(self.layers[layer_index].n)))
                else:
                    self.layers[layer_index].delta = self.e
            else:
                self.layers[layer_index].delta = numpy.multiply( self.layers[layer_index].derivative_activation_function(self.layers[layer_index].n) , (self.layers[layer_index+1].weights[:, :-1].T @ self.layers[layer_index+1].delta))
            
            
            if layer_index > 0:
                self.layers[layer_index].gradient = self.layers[layer_index].delta @ numpy.vstack((self.layers[layer_index-1].a, numpy.ones(self.layers[layer_index-1].a.shape[1]))).T
            else:
                self.layers[layer_index].gradient = self.layers[layer_index].delta @ numpy.hstack((self.inputs , numpy.ones((self.inputs.shape[0], 1))))
    
    
    def train_nn(self, inputs, target):
        self.foward_propagation(inputs, target)
        self.back_propagation()


    def optimize_sdg(self, inputs, target, q, epochs, lr):
        self.lr = lr
        for epoch in range(epochs):
            data_index = 0
            while (data_index < inputs.shape[0]):
                try:
                    train_data = inputs[data_index: data_index + q]
                    target_data = target[data_index: data_index + q]
                
                except:
                    train_data = inputs[data_index:]
                    target_data = target[data_index:]

                data_index += q        

                
                self.train_nn(train_data, target_data)
                
                for layer in self.layers:
                        
                    layer.weights -= lr * layer.gradient
                        
                
                self.MSE += numpy.sum(self.e ** 2)
            
            self.MSE = self.MSE / inputs.shape[0]

            if (epoch + 1)%(epochs/10) == 0:
                print("output", self.layers[0].gradient)
                print("Epoch: ", epoch + 1)
                if self.classifier == 0:
                    print("MSE: ", self.MSE)
                else: 
                    print("Cross Entropy: ", self.MSE)
            self.MSE = 0
        

    def optimize_adamax(self, inputs, target, q, epochs, lr, b1, b2, lambd, epsilon):
        self.lr = lr
        mu=0
        momentum=0

        for epoch in range(epochs):
            data_index = 0
            while (data_index < inputs.shape[0]):
                try:
                    train_data = inputs[data_index: data_index + q]
                    target_data = target[data_index: data_index + q]
                
                except:
                    train_data = inputs[data_index:]
                    target_data = target[data_index:]

                data_index += q        

                
                self.train_nn(train_data, target_data)
                
                w, gX = self.get_all_weights()

                normgX = numpy.linalg.norm(gX)    
                if lambd!=0:
                    gX=gX*(lambd*w)
                momentum=b1*momentum + (1+b1)*gX
                mu=numpy.maximum(b2*mu,normgX+epsilon)
                w=w-((lr*momentum)/((1-b1)*mu))
                
                self.reshape_params(w)

                        
                self.MSE += numpy.sum(self.e ** 2)
            
            self.MSE = self.MSE / inputs.shape[0]

            if (epoch + 1)%(epochs/10) == 0:
                print("Epoch: ", epoch + 1)
                if self.classifier == 0:
                    print("MSE: ", self.MSE)
                else:
                    print("Cross Entropy: ", self.MSE)
            self.MSE = 0


    
    def get_all_weights(self):
        all_weights = numpy.concatenate([layer.weights.flatten() for layer in self.layers])
        all_gradients = numpy.concatenate([layer.gradient.flatten() for layer in self.layers])
        return all_weights, all_gradients
    
    def reshape_params(self, params):
        for layer in self.layers:
            layer.weights = params[:layer.weights.size].reshape(layer.weights.shape)
            params = params[layer.weights.size:]
    
    def test_opt_regression(self, inputs_test, target_test):
        self.foward_propagation(inputs_test, target_test)
        
        print(targets_test.shape)
        print(numpy.array(self.output).shape)

        r2= r2_score(target_test, numpy.array(self.output).T)

        print("R2: ", r2)
    

    def test_opt_classification_binary(self, inputs_test, target_test):
        self.foward_propagation(inputs_test, target_test)
        for i in range(self.output.shape[0]):
            for j in range(self.output.shape[1]):
                if self.output[i][j] > 0.5:
                    self.output[i][j] = 1
                else:
                    self.output[i][j] = 0

        accuracy = accuracy_score(target_test.flatten(), numpy.array(self.output).flatten())

        print("Accuracy: ", accuracy)

    def test_opt_classification_multiclass(self, inputs_test, target_test):
        self.foward_propagation(inputs_test, target_test)
        predicted_classes = numpy.argmax(self.output, axis=0)

        true_classes = numpy.argmax(target_test, axis=1) 

        accuracy = accuracy_score(true_classes, predicted_classes)
        print("Accuracy: ", accuracy)


        


nn = network_architecture( 6 , #clases a clasificar, si es 0 es regresion, 
#si es 2 es clasificacion binaria, si es mayor a 3, es el numero de clases a clasificar
[8, 8, 6], #numero de neuronas por capa, ocultas y la ultima de salida
34 , #numero de entradas a la red
[functions.swish, functions.swish,  functions.softmax], #funciones de activacion de las 
#capas, si es clasificacion binaria o multiclase, se pone automatica softmax o sigmoid en la ultima capa
[functions.swish_derivative, functions.swish_derivative,  functions.softmax_derivative] , 
#las derivadas de las funciones de activacion
0.5) #la probabilidad de la funcion de dropout, si es 0 no se aplica

entrada  = numpy.array([ [1] , [1] , [1] ])
target = numpy.array([[0], [0] , [0]])

"""data = pd.read_csv("bezdekIris.data", header=None)

inputs = data.iloc[:, 0:4].values
targets = data.iloc[:, 4].values

one_hot_encoder = OneHotEncoder(sparse_output=False)
targets = one_hot_encoder.fit_transform(targets.reshape(-1, 1))"""


"""inputs = numpy.loadtxt("linreg_dataset.dat", usecols=(0, 1))  # Lee las columnas 0 y 1
targets = numpy.loadtxt("linreg_dataset.dat", usecols=2).reshape(-1 , 1)  # Lee la columna 2"""

"""inputs = numpy.loadtxt("xor_dataset.dat", usecols=(0, 1))  # Lee las columnas 0 y 1
targets = numpy.loadtxt("xor_dataset.dat", usecols=2) # Lee la columna 2"""

data = numpy.loadtxt("dermatology.dat")  # Lee las columnas 0 y 1

targets = data[:, 34]
inputs = data[:, :34]

one_hot_encoder = OneHotEncoder(sparse_output=False)
targets = one_hot_encoder.fit_transform(targets.reshape(-1, 1))



"""



print("engine inputs ", inputs[:10, :])
print("engine targets ", targets[:10, :])"""

"""mat = scipy.io.loadmat('engine_dataset.mat')

inputs = mat['engineInputs'].T
targets = mat['engineTargets'].T"""


"""ScalerInputs = RobustScaler()
ScalerTargets = RobustScaler()

inputs = ScalerInputs.fit_transform(inputs)
targets = ScalerTargets.fit_transform(targets)"""

inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.4, random_state=None)

"""print("engine inputs ", inputs[:10, :])
print("engine targets ", targets[:10, :])


print("shapes", inputs.shape, targets.shape)


print("inputs_train", inputs_train[:10, :])
"""
"""inputs_train = inputs_train
inputs_test = inputs_test
targets_train = targets_train
targets_test = targets_test """

# Normalización de las entradas
"""min_train = numpy.min(inputs_train[:, 0:1], axis=0)
max_train = numpy.max(inputs_train[:, 0:1], axis=0)
inputs_train[:, 0:1] = 2 * ((inputs_train[:, 0:1] - min_train) / (max_train - min_train)) - 1

min_train = numpy.min(inputs_train[:, 1:2], axis=0)
max_train = numpy.max(inputs_train[:, 1:2], axis=0)
inputs_train[:, 1:2] = 2 * ((inputs_train[:, 1:2] - min_train) / (max_train - min_train)) - 1

min_test = numpy.min(inputs_test[:, 0:1], axis=0)
max_test = numpy.max(inputs_test[:, 0:1], axis=0)
inputs_test [: , 0:1] = 2 * ((inputs_test[:, 0:1] - min_test) / (max_test - min_test)) - 1

min_test = numpy.min(inputs_test[:, 1:2], axis=0)
max_test = numpy.max(inputs_test[:, 1:2], axis=0)
inputs_test [: , 1:2] = 2 * ((inputs_test[:, 1:2] - min_test) / (max_test - min_test)) - 1"""

##nn.optimize_sdg(inputs_train, targets_train, 90, 10000, 0.001)

nn.optimize_adamax(inputs_train, targets_train, 60, 10000, 0.001, 0.9, 0.999, 0, 1e-8,)



##nn.test_opt_regression(inputs_test, targets_test)

nn.test_opt_classification_multiclass(inputs_test, targets_test)

##nn.test_opt_classification_binary(inputs_test, targets_test)
#nn.test_opt(inputs_test, targets_test)



