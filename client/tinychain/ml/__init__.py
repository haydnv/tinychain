from abc import abstractmethod, ABC
from tinychain.collection.tensor import einsum, Dense
from tinychain.ref import After
from tinychain.state import Tuple


class Activation(ABC):
    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass


class Sigmoid(Activation):
    def forward(self, Z):
        return 1 / (1 + (-Z).exp())

    def backward(self, dA, Z):
        sig = self.forward(Z=Z)
        return dA * sig * (1 - sig)


class ReLU(Activation):
    def forward(self, Z):
        return Z * (Z > 0)

    def backward(self, dA, Z):
        return dA * (Z > 0)


def layer(weights, activation):
    class Layer(Tuple):
        def eval(self, inputs):
            return activation.forward(einsum("ij,ki->kj", [self[0], inputs]))

        def gradients(self, A_prev, dA, Z, m):
            dZ = activation.backward(dA, Z).copy()
            d_weights = einsum("ki,kj->ij", [A_prev, dZ]) / m
            dA_prev = einsum("ij,kj->kj", [dZ, self[0]])
            return dA_prev, d_weights

        def train_eval(self, inputs):
            Z = einsum("ij,ki->kj", [self[0], inputs])
            A = activation.forward(Z)
            return A, Z

        def update(self, d_weights):
            new_weights = weights - d_weights
            return Dense(self[0]).overwrite(new_weights)

    return Layer([weights])


def neural_net(layers):
    num_layers = len(layers)

    class NeuralNet(Tuple):
        def eval(self, inputs):
            state = layers[0].eval(inputs)
            for i in range(1, len(layers)):
                state = layers[i].eval(state)

            return state

        def train(self, inputs, cost):
            A = [inputs]
            Z = [None]

            for layer in layers:
                A_l, Z_l = layer.train_eval(A[-1])
                A.append(A_l)
                Z.append(Z_l)

            dA = cost(A[-1]).copy()

            updates = [(A[-1], dA)]
            for i in reversed(range(0, num_layers)):
                dA_prev, d_weights = layers[i].gradients(A[i], dA, Z[i + 1], num_layers)
                update = layers[i].update(d_weights)
                updates.append(After(update, dA_prev))

            return After(updates, A[-1])

    return NeuralNet(layers)
