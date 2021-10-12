from abc import abstractmethod, ABC
from tinychain.collection.tensor import einsum, Dense
from tinychain.ref import After
from tinychain.state import Tuple


class Activation(ABC):
    @abstractmethod
    def forward(self, Z):
        pass

    @abstractmethod
    def backward(self, dA, Z):
        pass


class Sigmoid(Activation):
    def forward(self, Z):
        return 1 / (1 + (-Z).exp())

    def backward(self, dA, Z):
        sig = self.forward(Z=Z)
        return sig * (1 - sig) * dA


class ReLU(Activation):
    def forward(self, Z):
        return Z * (Z > 0)

    def backward(self, dA, Z):
        return (Z > 0) * dA


def layer(weights, bias, activation):
    # dimensions (for einsum): k = number of examples, i = weight input dim, j = weight output dim

    class Layer(Tuple):
        def eval(self, inputs):
            return activation.forward(einsum("ij,ki->kj", [self[0], inputs])) + self[1]

        def gradients(self, A_prev, dA, Z):
            dZ = activation.backward(dA, Z).copy()
            dA_prev = einsum("kj,ij->ki", [dZ, self[0]])
            d_weights = einsum("kj,ki->ij", [dZ, A_prev])
            d_bias = dZ.sum(0)
            return dA_prev, d_weights, d_bias

        def train_eval(self, inputs):
            Z = einsum("ij,ki->kj", [self[0], inputs])
            A = activation.forward(Z) + self[1]
            return A, Z

        def update(self, d_weights, d_bias):
            w = Dense(self[0])
            b = Dense(self[1])
            return w.overwrite(w - d_weights), b.overwrite(b - d_bias)

    return Layer([weights, bias])


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
                A.append(A_l.copy())
                Z.append(Z_l)

            m = inputs.shape()[0]
            dA = cost(A[-1]).sum() / m

            updates = []
            for i in reversed(range(0, num_layers)):
                dA, d_weights, d_bias = layers[i].gradients(A[i], dA, Z[i + 1])
                update = layers[i].update(d_weights, d_bias)
                updates.append(update)

            return After(updates, A[-1])

    return NeuralNet(layers)
