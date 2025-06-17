#same model from the ipynb file
#creating a neural network layer
import numpy as np

class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


# ReLU activation
class Activation_ReLU:
    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify original variable,
        # let's make a copy of values first
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0


# Softmax activation
class Activation_Softmax:
    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


# Common loss class
class Loss:
    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):
        # print("here")
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        # Return loss
        return data_loss


# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        # print("here1")
        samples = len(y_pred)
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # print("y_pred:",y_pred_clipped)

        # Probabilities for target values - only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy:
    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # # If labels are one-hot encoded,
        # # turn them into discrete values
        # if len(y_true.shape) == 2:
        #     y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Optimizer_SGD:
    # Initialize optimizer - set settings
    # Learning rate of 1.0 is default for this optimizer
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    # Update parameters
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

#creating CNN layer

class CNN_layer:
    def __init__(self, in_channels=1, out_channels=1, kernel_size=3):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Initialize filters: [out_channels, in_channels, kernel_size, kernel_size]
        # Each output channel has its own set of filters for all input channels
        self.filters = 0.1 * np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        
        # One bias per output channel
        self.biases = 0.1 * np.random.randn(out_channels)
        


    def forward(self, input):
       
        self.input = input  

        # Add batch dimension if missing

        batch_size, channels, h, w = input.shape
        assert channels == self.in_channels, f"Expected {self.in_channels} channels, got {channels}"

        out_h = h - self.kernel_size + 1
        out_w = w - self.kernel_size + 1

        # Feature maps for each output channel and batch
        self.output = np.zeros((batch_size, self.out_channels, out_h, out_w))

        # Apply each output filter for each sample in batch
        for b in range(batch_size):
            for out_ch in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        conv_sum = 0
                        for in_ch in range(self.in_channels):
                            patch = input[b, in_ch, i:i+self.kernel_size, j:j+self.kernel_size]
                            conv_sum += np.sum(patch * self.filters[out_ch, in_ch])
                        self.output[b, out_ch, i, j] = conv_sum + self.biases[out_ch]
    
    def backward(self, doutput):
        input = self.input
        batch_size, channels, h, w = input.shape
        out_h = h - self.kernel_size + 1
        out_w = w - self.kernel_size + 1

        self.dfilters = np.zeros_like(self.filters)
        self.dbiases = np.zeros_like(self.biases)
        self.dinputs = np.zeros_like(input)

        for b in range(batch_size):
            for out_ch in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        for in_ch in range(self.in_channels):
                            patch = input[b, in_ch, i:i+self.kernel_size, j:j+self.kernel_size]
                            self.dfilters[out_ch, in_ch] += doutput[b, out_ch, i, j] * patch
                            self.dinputs[b, in_ch, i:i+self.kernel_size, j:j+self.kernel_size] += doutput[b, out_ch, i, j] * self.filters[out_ch, in_ch]
                            self.dbiases[out_ch] += doutput[b, out_ch, i, j]

        # Remove batch dimension if input was single sample
        if len(self.input.shape) == 3:
            self.dinputs = self.dinputs[0]
    
    def optimize(self,learning_rate=0.01):
        # Update filters and biases using gradients computed in backward pass
        self.filters -= learning_rate * self.dfilters
        self.biases -= learning_rate * self.dbiases



class maxPool:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        self.input = x
        batch_size, channels, h, w = x.shape
        out_h = (h - self.kernel_size) // self.stride + 1
        out_w = (w - self.kernel_size) // self.stride + 1
        self.output = np.zeros((batch_size, channels, out_h, out_w))
        self.max_indices = np.zeros_like(self.output, dtype=int)

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size
                        patch = x[b, c, h_start:h_end, w_start:w_end]
                        self.output[b, c, i, j] = np.max(patch)
                        self.max_indices[b, c, i, j] = np.argmax(patch)
        return self.output

    def backward(self, doutput):
        """
        doutput: gradient of loss w.r.t. output, same shape as self.output
        """
        batch_size, channels, h, w = self.input.shape
        self.dinputs = np.zeros_like(self.input)
        out_h, out_w = doutput.shape[2], doutput.shape[3]

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    
                    for j in range(out_w):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size
                        patch = self.input[b, c, h_start:h_end, w_start:w_end]
                        max_idx = self.max_indices[b, c, i, j]
                        idx = np.unravel_index(max_idx, patch.shape)
                        self.dinputs[b, c, h_start:h_end, w_start:w_end][idx] += doutput[b, c, i, j]


class flatten_layer:
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)    #batch_size, then all rest of layers flattened

    def backward(self, doutput):
        self.dinputs=doutput.reshape(self.input_shape)
    

cnn1=CNN_layer(in_channels=1, out_channels=4, kernel_size=3) #28x28-> 26x26
relu1=Activation_ReLU()
mp1=maxPool(kernel_size=2, stride=2)#26x26-> 13x13
cnn2=CNN_layer(in_channels=4, out_channels=8, kernel_size=3)#13x13-> 11x11
relu2=Activation_ReLU() 
mp2=maxPool(kernel_size=2, stride=2)# 11x11-> 5x5
f=flatten_layer()
nn1=Layer_Dense(n_inputs=8*5*5, n_neurons=10) # 8 channels, 10 feature maps
lsfn=Activation_Softmax_Loss_CategoricalCrossentropy()
# Optimizer
optimizer = Optimizer_SGD(learning_rate=0.01)

optimizer=Optimizer_SGD(learning_rate=0.1)

import json

# Load model parameters from JSON file
with open('model_para.json', 'r') as parameter:
    loaded_para = json.load(parameter)


# Assign loaded parameters back to the model (convert lists to numpy arrays)
cnn1.filters = np.array(loaded_para['cnn1_filters'])
cnn1.biases = np.array(loaded_para['cnn1_biases'])
cnn2.filters = np.array(loaded_para['cnn2_filters'])
cnn2.biases = np.array(loaded_para['cnn2_biases'])
nn1.weights = np.array(loaded_para['nn1_weights'])
nn1.biases = np.array(loaded_para['nn1_biases'])

def predict(x):
    with open('model_para.json', 'r') as parameter:
        loaded_para = json.load(parameter)
    # Assign loaded parameters back to the model (convert lists to numpy arrays)
    cnn1.filters = np.array(loaded_para['cnn1_filters'])
    cnn1.biases = np.array(loaded_para['cnn1_biases'])
    cnn2.filters = np.array(loaded_para['cnn2_filters'])
    cnn2.biases = np.array(loaded_para['cnn2_biases'])
    nn1.weights = np.array(loaded_para['nn1_weights'])
    nn1.biases = np.array(loaded_para['nn1_biases'])
    cnn1.forward(x)
    relu1.forward(cnn1.output)
    mp1.forward(relu1.output)
    cnn2.forward(mp1.output)
    relu2.forward(cnn2.output)
    mp2.forward(relu2.output)
    f_output = f.forward(mp2.output)
    nn1.forward(f_output)
    return np.argmax(nn1.output, axis=1)