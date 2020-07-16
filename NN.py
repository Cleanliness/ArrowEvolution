import numpy as np

"""modification of NN.py from NumberNN to work with ArrowEvolution. Does not require mnist, output layer neurons 
'fire' once they reach a threshold activation."""

class NN:

    def __init__(self, layers):
        """
        creates a new neural network/multilayer perceptron.

        format layers like this: [# of neurons in layer1, # in layer 2, ... ], must have at least 3 elements, only
        accept positive integers.

        neuron activation and bias list is set up like this:
        [a1, a2, ... an], where a1,... an are arrays containing neuron activations/biases at the nth layer
        ** bias matrix does not include first layer

        weight matrices are formatted like this, for some layer l:

        [w1a1, w2a1, ... w_n a_1]
        [w1a2, w2a2, ... w_n a_2]
        [           .           ]
        [           .           ]
        [           .           ]
        [w1 a_m, ....... w_n a_m]

        w_n a_m represents a weight connecting the the nth weight from the l-1th layer to the mth neuron
        """
        # vectorizing activation function and its derivative
        self.act_f = np.vectorize(sigmoid)
        self.act_f_prime = np.vectorize(sigmoid_prime)

        # set up neuron activation and random bias list
        self.activations = [np.array([0 for i in range(0, layers[0])])]
        self.biases = []

        # set up hidden layer activation and biases
        for l in layers[1:-1]:
            self.activations.append(np.array([0 for i in range(0, l)]))
            self.biases.append(np.array([np.random.random_sample()*2 - 1 for i in range(0, l)]))

        # set up output layer activations and biases
        self.activations.append(np.array([0 for i in range(0, layers[-1])]))
        self.biases.append(np.array([np.random.random_sample()*2 - 1 for i in range(0, layers[-1])]))

        # setting up neuron weight matrices w/ random weights
        self.weights = []
        for i in range(1, len(self.activations)):
            mat = []
            for r in range(0, len(self.activations[i])):
                row = np.array([2*(np.random.random_sample()*2 - 1) for i in range(0, len(self.activations[i-1]))])
                mat.append(row)
            self.weights.append(np.array(mat))

        # setting up sum array with dummy values
        self.sum_arr = [1 for i in self.activations]

    def classify(self, img):
        """
        Forward propagates the NN given a list of inputs

        img must have same number of elements as activations"""
        if len(img) != len(self.activations[0]):
            return None

        arr_img = np.array(img)

        # set input layer to img, update sum array at first layer
        self.activations.pop(0)
        self.activations.insert(0, self.act_f(arr_img))

        self.sum_arr.pop(0)
        self.sum_arr.insert(0, arr_img)

        # update activations and sums for each layer
        for layer in range(1, len(self.activations)):
            self.activations.pop(layer)
            w = self.weights[layer - 1]
            a = self.activations[layer - 1]

            # calculate weighted sum of current layer
            l_sum = w.dot(a) + self.biases[layer-1]
            self.sum_arr.pop(layer)
            self.sum_arr.insert(layer, l_sum)

            # apply sigmoid to all sums and add it to the current activation matrix
            self.activations.insert(layer, self.act_f(l_sum))

        return self.activations[-1]

    def load(self, fname="NN_save.txt"):
        """loads weights and biases from a text file"""
        f = open(fname, 'r')
        lines = f.readlines()

        self.__init__(eval(lines[1]))
        self.biases = eval(lines[3])
        self.weights = eval(lines[5])

    def save(self, fname="NN_save.txt"):
        """saves weights and biases to a text file.
        Formatting:

        a,b,c\n --> (a,b,c = # of neurons in corresponding hidden layers)

        [np.array(a), np.array(b), ... ]\n --> string representation of biases in NN where a,b are string reps of lists
                                                containing the biases of each layer(element of this list)

        [np.array([np.array(c,d)), np.array(d,e)], ... ]\n --> string rep of weights in NN, c,d are string reps of
                                                            lists containing the rows of each weight matrix
        """
        f = open(fname, 'w')

        # writing number of hidden layers
        f.write('----------------------Layers---------------------\n')
        f.write("[")
        for i in range(0, len(self.activations)):
            f.write(str(len(self.activations[i])) + ",")
        f.write("]\n")

        # writing string rep of bias array
        f.write('----------------------biases---------------------\n')

        f.write("[")
        for i in range(0, len(self.biases)):
            s = np.array2string(self.biases[i], separator=",")
            f.write("np.array(" + np.array2string(self.biases[i], separator=",").replace("\n", "") + "),")
        f.write("]\n")

        f.write('----------------------weights---------------------\n')
        # writing string rep of weight array
        f.write("[")
        for wm in range(0, len(self.weights)):
            f.write("np.array([")
            for row in range(0, len(self.weights[wm])):
                f.write("np.array(" + np.array2string(self.weights[wm][row], separator=",").replace("\n", "") + "),")
            f.write("]),")
        f.write("]")

        f.close()


def sigmoid(x):
    """activation function for the network"""
    return 1 / (1 + np.exp(-1 * x))


def sigmoid_prime(x):
    """derivative of the sigmoid function with respect to its input"""
    return sigmoid(x) * (1 - sigmoid(x))



