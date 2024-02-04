"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

import matplotlib.pyplot as plt 


class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes) # Aqui cuenta cuantas capas tiene la red neuronal (incluyendo los inputs)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] # Genera un array por cada capa de neuronas
        # que tengamos y cada array tendra tantos numeros random como neuronas tenga esa capa 
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])] # Genera un array por cada capa de neuronas donde 
        # cada array esta formado por un lista de n listas con m elementos por lista, donde n es la cantidad de neuronas
        #de la capa y m es la cantidad de inputs (activaciones) que va a recibir de la capa anterior

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights): #El zip "une" los arrays de los pesos y los bias
            # y el for va tomando los bias de las neuronas de una capa (l) y las listas que contienen los pesos
            # que se le asignaron a las activaciones de las neuronas de la capa l-1
            a = sigmoid(np.dot(w, a)+b) #Es la activacion que se va a obtener si se toma a como el input
        return a #Nos regresa el valor de la activacion de la capa en la que va, nos regresa un a que se usara
    # de input para la siguiente capa

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        cost = []
        if test_data:
            test_data = list(test_data) #Se crea una lista con los datos de prueba
            n_test = len(test_data) #Cuenta cuantos datos de prueba hay

        training_data = list(training_data) #Se crea una lista con los datos de entrenamiento
        n = len(training_data) #Cuenta cuantos datos de entrenamiento hay
        for j in range(epochs):
            random.shuffle(training_data) #Revuelve aleatoriamente los datos de entrenamiento
            mini_batches = [
                training_data[k:k+mini_batch_size] 
                for k in range(0, n, mini_batch_size)]#Separa los datos de entrenamiento en listas del tamaño 
                # del minibatch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2} with cost: {3}".format(
                    j, self.evaluate(test_data), n_test, self.cost_function(test_data))) #Si diste datos de prueba te regresa el procentaje de aciertos
                cost.append(self.cost_function(test_data))
                 
            else:
                print("Epoch {0} complete".format(j))#Si no diste datos te prueba solo te regresa que la epoca se termino

        num_epochs = list(range(len(cost)))

        fig, ax = plt.subplots()

        ax.plot( num_epochs, cost)

        ax.set(xlim=(0, len(cost)),
            ylim=(0, max(cost)*1.2))
        plt.show()

    

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases] #Crea una lista con arrays del mismo tamaño que el array de los bias
        nabla_w = [np.zeros(w.shape) for w in self.weights] #Crea una lista con arrays del mismo tamaño que el array de los pesos
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) #Iguala  delta_nabla_b, delta_nabla_w a la tupla
            #que nos regresa backprop, la variables delbackprop(x,y) son los los datos de input(x) y el resultado
            #correcto de a dicho input(y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] #Al array que teniamos con ceros
            #se le va sumar a cada elemento el elemento del array delta_nabla_b correspondiente
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]#lo mismo que el de arriba
            #solo que este con los pesos, al final tendremos un array del mismo tamaño que el de los pesos, o bias,
            #segun corresponda
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)] #Aqui es donde se aplica el SGD, a los pesos actuales
        #se les resta la derivada de la funcion de costo respecto al peso correspondiente multiplicada por la tasa de
        #aprendizaje dividida entre el numero de datos de entrenamiento que tiene el minibatch
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)] #Lo mismo que elde arriba solo que para los bias

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases] #Crea una lista con arrays del mismo tamaño que el array de los bias
        nabla_w = [np.zeros(w.shape) for w in self.weights] #Crea una lista con arrays del mismo tamaño que el array de los pesos
        # feedforward
        activation = x #x va a ser los datos de input
        activations = [x] # lista para guardar todas las activaciones, capa por capa
        zs = [] # lista para guardar todos los vectores z, capa por capa
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z) #Calculamos los inputs "pesados" y los vamos formando en una lista
            activation = sigmoid(z) #Se le aplica la funcion sigmoide a los inputs "pesados", lo que nos va a dar 
            #la activacion de la capa
            activations.append(activation)#Le anexa la activacion de la capa a la lista que tiene todas las activaciones de las capas
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1]) #Es el error de la ultima capa
        nabla_b[-1] = delta #Como la parcial del costo respecto a los bias es igual al error 
        #Igualamos la nabla_b de la ultima capa y su error
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) #Recordando nabla_w de la capa l es igual al error
        #de la capa l por la activacion de las capas anteriores
        for l in range(2, self.num_layers):
            z = zs[-l] #Va a empezar tomando el vector z que corresponde a la penultima capa hasta la segunda capa
            sp = sigmoid_prime(z)#Va a calcular la derivada de la sigmoide para los vectores z que saca a linea anterior
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp #Aqui aplica la definicion que obtuvimos
            #para el error de una capa que no sea la ultima
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w) #Nos regresa una tupla donde los elementos nabla_b son todos los nabla_b
        #de las capas y de manera similar nabla_w
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data] #Va a hacer una lista de tuplas que para cada
        #test data se va a tener el resultado que obtuvo la red neuronal y la respuesta correcta
        return sum(int(x == y) for (x, y) in test_results) #Suma un uno cada vez que el resultado de la red neuronal sea correcto

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y) #Es la funcion de costo cuadratica derivada
    

    def cost_function(self, test_data): #Aqui defini la funcion de costo para poder graficarla despues
        cost_x = [0.5*(np.square(np.argmax(self.feedforward(x)) - y))
                        for (x, y) in test_data] 
        cost_epoch = np.average(cost_x)

        return(cost_epoch)



#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z)) #Se define la funcion sigmoide

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z)) #Nos regresa el valor de evaluar z en la derivada de la funcion sigmoide
