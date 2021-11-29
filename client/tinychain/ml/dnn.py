# Constructors for a generic deep neural network.
#
# Prefer this implementation if no more domain-specific neural net architecture is needed.

from tinychain.collection.tensor import einsum, Dense
from tinychain.ml import Layer, NeuralNet
from tinychain.ref import After


def layer(weights, bias, activation):
    """Construct a new layer for a deep neural network."""

    # dimensions (for `einsum`): k = number of examples, i = weight input dim, j = weight output dim


    class DNNLayer(Layer):
        '''
        Layer= ( Weights , Bias , Activation_Function)
        type(Weights) :tc.tensor.Dense
        type(Bias):tc.tensor.Dense
        type(Activation_Function):(e.g.: tc.ml.sigmoid)
        
        '''
        @property
        def bias(self):  
            '''
            We use property accessor so we can recall self.bias in stead of Dense(self[1])
            '''
            return Dense(self[1])

        @property
        def weights(self):  
            '''
            We use property accessor so we can recall self.weights in stead of Dense(self[0])
            '''
            return Dense(self[0])

        def eval(self, inputs):
            '''
            eval function applies Activation Function on the dot product 
            between the inputs  the corresponding weights_matrix + bias
            type(inputs) :tc.tensor.Dense
            
            lets understand einsum :-
            
            einsum('ij,ki->kj',[self.weights,inputs]) 
            
            let:-
            ----
            ij = (3,5)       3 neurons input layer , 5  neurons Next layer
            ki = (1000,3)    1000 rows of data , 3 neurons this layer
                             3,5  * 1000,3 -> 1000,5
            kj   k = 1000 , j = 5
            so basically einstein sum here is just simple np.dot(self.inputs,weights)
            '''
            
            return activation.forward(einsum("ij,ki->kj", [self.weights, inputs])) + self.bias


        def gradients(self, A_prev, dA, Z):
            
            '''
            gradients function :it calculates delta values in Weights , Bias and in previous layer.
            A_prev : Previous Input.
            Z : Current Output before Activation.
            dA: detla_Error.
            dZ : delta Error in the Current Layer.
            dA_Prev: delta Error in Previous Layer.
            d_weights: delta Weights.
            d_bias : delta Bias.
            activation.backward(dA,Z) : performs derivative version of your activation function
                                          to apply backward propagation.
            
            '''
            dZ = activation.backward(dA, Z).copy()
            dA_prev = einsum("kj,ij->ki", [dZ, self.weights])  
            d_weights = einsum("kj,ki->ij", [dZ, A_prev])
            d_bias = dZ.sum(0)
            return dA_prev, d_weights, d_bias

        def train_eval(self, inputs):
            '''
            train_eval function you can use it on any layer to get its
            output after dot product and after applying activation function on that output
            
            Z : Output of dot product between weight and inputs .
            A : returning the output after applying activation function on it.
            '''
            Z = einsum("ij,ki->kj", [self.weights, inputs])
            A = activation.forward(Z) + self.bias
            return A, Z

        def update(self, d_weights, d_bias):
            '''
            update function is used to ensure that update operations are done
            New_Weight = Old Weight - Delta Weight
            New_Bias = Old Bias - Delta Bias
            '''
            return self.weights.write(self.weights - d_weights), self.bias.write(self.bias - d_bias)
                

    return DNNLayer([weights, bias])

    
    """
    
    To Build Adam Optimizer or w/e you need just follow these steps:-
    1. Create Your Class with Your Optimizer Name.
    2. Build `Create` method as a class method that would initialize everything you need to set before optimizing
    3. Build `Optimize` method to be called on each layer so it can return delta weights and delta bias also if 
       you need to update some iterative parameters you might need to map all of them in function return.
    
    """
    
class Adam_Optimizer():
    @classmethod
    def create(cls):
        return tc.Map({"l_m":tc.Number(0) ," l_v":tc.Number(0) , "t":tc.Number(0)})
    def optimize (self, A_prev, dA, Z ,decay_rate_1 : tc.Number, decay_rate_2 : tc.Number,
                  epsilon :tc.Number , l_m,l_v,t):  
                          
         dZ = activation.backward(dA, Z).copy()          
         # Gradients for each layer
         g = einsum("kj,ki->ij", [dZ, A_prev])              
         t=t+1           
         # Computing 1st and 2nd moment for each layer
         l_m = l_m * decay_rate_1 + (1- decay_rate_1) * g            
         l_v = l_v * decay_rate_2 + (1- decay_rate_2) * (g ** 2)            
         l_m_corrected = l_m / (1-(decay_rate_1 ** t))
         l_v_corrected = l_v / (1-(decay_rate_2 ** t))
         
         # Update Weights
         d_weights = l_m_corrected / ((l_v_corrected)**(0.5) + epsilon)                       
         d_bias = dZ.sum(0)
         return tc.Map({'l_m':l_m,
                        'l_v':l_v,
                        't':t,
                        'd_weights': d_weights,
                        'd_bias':d_bias})
     




def neural_net(layers):
    """Construct a new deep neural network with the given layers."""

    num_layers = len(layers)

    class DNN(NeuralNet):
        def eval(self, inputs):
            '''
            eval function it will help you to get the result
            of each layer
            
            for the first layer its exception we eval the inputs
            
            type(inputs) : tc.tensor.Dense
            state : is the Result from evaluating  cascaded layers 
            '''
            state = layers[0].eval(inputs) 
            for i in range(1, len(layers)):
                state = layers[i].eval(state)

            return state
       


        def train(self, inputs, cost,optimizer):
            '''
            train function will take the inputs and cost function
            and give back last layer output after activation.
            
            A : list of Tensors starts with Inputs Tensor the Result of each layer after activation.
            Z : list of Output Tensors before getting applied with activation function.
            
            m : number of data rows
            
            cost : is a cost function which takes the output and actual output as input
                 and calculate the difference (e.g. : lambda output:((output- actual_labels)**2)* learning_Rate)
                 
           optimizer : is your selected optimizer to tune your Neural Network learning paramters.           
            
           updates : list of updated values for weights and biases.
           dA : Average Delta Error 
           A[i]: Previous Layer Tensor.
           Z[i+1] : Current Layer Tensor.
           A[-1]: Last Layer after activation ( Prediction ).
           After(updates, A[-1]) : Output of Last Layer after applying Activation and updating weights.
            '''

            A = [inputs]
            Z = [None]

            for layer in layers:
                A_l, Z_l = layer.train_eval(A[-1])
                A.append(A_l.copy())
                Z.append(Z_l)

            m = inputs.shape[0]   
            dA = cost(A[-1]).sum() / m

            updates = []
            

            l_m,l_v,t=optimizer['l_m'],optimizer['l_v'],optimizer['t']
            
            for i in reversed(range(0, num_layers)):
                
                updated_values = optimizer.optimize(A[i],dA,Z[i+1],decay_rate_1=0.9,
                                                    decay_rate_2=0.99,epsilon=(10e-8) ,l_m ,l_v ,t)
                l_m,l_v,t=updated_values['l_m'],updated_values['l_v'],updated_values['t']
                d_weights,d_bias=updated_values['d_weights'],updated_values['d_bias']               
                update= layers[i].update(d_weights,d_bias)
                updates.append(update)
                
            optimizer_state = l_m,l_v,t
               

            return After(updates, A[-1]) , optimizer_state

    return DNN(layers)