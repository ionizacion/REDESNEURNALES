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

    def SGD(self, training_data, epochs, mini_batch_size, eta, phi,
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
        velocity = [np.zeros(w.shape) for w in self.weights] #Crea una lista con arrays del mismo tamaño que el array de los pesos
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
                self.update_mini_batch(mini_batch, eta, phi, velocity)#Aqui llama a la funcion que actualiza los valores de los pesos. eta es el learning rate
                # y phi es el momentum
            if test_data:
                print("Epoch {0}: {1} / {2} with cost: {3}".format(
                    j, self.evaluate(test_data), n_test, self.cost_function_cross_entropy(test_data))) #Si diste datos de prueba te regresa el procentaje de aciertos
                cost.append(self.cost_function_cross_entropy(test_data))#Usa la funcion que defini y crea una lista donde va anexando el costo
                # de la epocas. Esta lista su usa para graficarlas
            else:
                print("Epoch {0} complete".format(j))#Si no diste datos te prueba solo te regresa que la epoca se termino

        num_epochs = list(range(len(cost))) #cuenta cuantas epocas hubo para usar ese dato para el eje x de la grafica

        fig, ax = plt.subplots()

        ax.plot( num_epochs, cost) #Aqui digo que tome la lista num_epochs como el eje x y la lista de los costos en el eje y

        ax.set(xlim=(0, len(cost)-1),
            ylim=(0, max(cost)*1.2)) #Le pongo limites alos ejes
        
        ax.set_xlabel('Epocas')
        ax.set_ylabel('Valor de la funcion de costo')
        ax.set_title("Costo con cross-entropy. Learning rate = " + str(eta) + "Momentum =" + str(phi))

        plt.show() #Muestro la grafica

    

    def update_mini_batch(self, mini_batch, eta, phi, velocity):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases] #Crea una lista con arrays del mismo tamaño que el array de los bias
        nabla_w = [np.zeros(w.shape) for w in self.weights] #Crea una lista con arrays del mismo tamaño que el array de los pesos
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) #Iguala  delta_nabla_b, delta_nabla_w a la tupla
            #que nos regresa backprop, la variables del backprop(x,y) son los los datos de input(x) y el resultado
            #correcto de a dicho input(y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] #Al array que teniamos con ceros
            #se le va sumar a cada elemento el elemento del array delta_nabla_b correspondiente
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]#lo mismo que el de arriba
            #solo que este con los pesos, al final tendremos un array del mismo tamaño que el de los pesos, o bias,
            #segun corresponda
        velocity = [phi*v - eta*nw for v, nw in zip(velocity, nabla_w)] #Una vez se tienen los arrays nabla_w completos (para el minibatch)
        #se le restan a la velocidad que se tenia y obtenemos la velocidad nueva
        self.weights = [w+(1/len(mini_batch))*v
                        for w, v in zip(self.weights,velocity)] #Aqui es donde se aplica el SGD con momentum. A los pesos se les
        #suma la velocidad divida entre el tamaño del minibatch
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
        delta = self.delta_cross_entropy(activations[-1], y) #Es el error de la ultima capa para el caso de la funcion de costo cross-entropy
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
    
    def delta_cross_entropy(self, output_activations, y):
        return (output_activations-y) #Cuando se ocupa la funcion de costo cross_entropy este es el valor del error de la ultima capa
    

    def cost_function_cuadratic(self, test_data): #Aqui defini la funcion de costo cuadratica para poder graficarla despues
        cost_x = [0.5*np.linalg.norm(np.argmax(self.feedforward(x)) - y)**2
                        for (x, y) in test_data] 
        cost_epoch = np.average(cost_x)

        return(cost_epoch)

    def vector_y(self, j): #Esto es para vectorizar los resultados correctos de los test_data
        e = np.zeros((10, 1)) #Hace una matriz 10 por 1 de puros ceros
        e[j] = 1.0 #Hace el indice del numero correcto igual a 1
        return e

    def cost_function_cross_entropy(self, test_data): #Aqui defini la funcion de costo cross-entropy para poder graficarla despues
        cost_x = []
        for (x, y) in test_data:
             y = self.vector_y(y)
             cost_x.append(np.nan_to_num(-y*np.log(self.feedforward(x)) - (1-y)*np.log(1-self.feedforward(x)))) #calculo el error
             # de una dato de prueba. Aqui se ocupa el resultado vectorizado
        cost_epoch = np.average(cost_x) #Saca el costo de la epoca calculando el promedio del costo de todos los valores del test_data
        return(cost_epoch)
    

    

    

    


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z)) #Se define la funcion sigmoide

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z)) #Nos regresa el valor de evaluar z en la derivada de la funcion sigmoide
