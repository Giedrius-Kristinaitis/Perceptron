from file_reader import read_file
import matplotlib.pyplot as plt
import neurolab as nl
import pylab as pl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def get_inputs_outputs(data_set, input_count, range_start, range_end):
    data_inputs = []
    data_outputs = []

    for i in range(range_start, range_end):
        data_input = []

        if i + input_count + 1 >= range_end:
            break

        for j in range(i, i + input_count):
            data_input.append(data_set[j])

        data_inputs.append(data_input)
        data_outputs.append([data_set[i + input_count]])

    return data_inputs, data_outputs

# read sunspot data
attribute_types = {
    "Metai": "numeric",
    "Aktyvumas": "numeric"
}

data = read_file("sunspot.txt", attribute_types)

# plot expected output for activity inputs with 2 inputs
inputs1 = []
inputs2 = []
outputs = []

for i in range(0, 200, 2):
    inputs1.append(data[1].values[i])
    inputs2.append(data[1].values[i + 1])
    outputs.append(data[1].values[i + 2])

ax = plt.axes(projection='3d')
ax.scatter3D(inputs1, inputs2, outputs)
ax.set_title("Outputs for given inputs")
ax.set_xlabel("Input 1")
ax.set_ylabel("Input 2")
ax.set_zlabel("Output")
plt.show()

# plot input data
plt.plot(data[0].values, data[1].values)
plt.title("Sunspot activity")
plt.xlabel('Year')
plt.ylabel('Activity')
plt.grid()
plt.show()

# input count for a single output
n = 9

# learning rate
learning_rate = 0.00000075

# maximum number of epochs
epochs = 10000

# after how many epochs to display error
epochs_to_show = 20

# get sunspot activity value range
min_value = data[1].min_value()
max_value = data[1].max_value()

# get input ranges
input_ranges = []

for i in range(n):
    input_ranges.append([min_value, max_value])

# create a perceptron with n inputs
net = nl.net.newp(input_ranges, 1)

# set transfer function to linear
net.layers[0].transf = nl.trans.PureLin()

# set error function to MSE
net.errorf = nl.error.MSE()

# set training error goal
net.trainf.defaults["goal"] = 200

# get training input and output
training_input, training_output = get_inputs_outputs(data[1].values, n, 0, 200)

# train the perceptron and get training error matrix
error = net.train(training_input, training_output, epochs=epochs, lr=learning_rate, show=epochs_to_show)

# display neuron parameters
print(f"Neuron weight vector after training: {net.layers[0].np['w']}")
print(f"Neuron bias: {net.layers[0].np['b']}")

# plot error
pl.plot(error)
pl.title('MSE')
pl.xlabel('Epoch number')
pl.ylabel('Train error')
pl.grid()
pl.show()

# make predictions with trained net for the remaining data
print("Testing perceptron predictions...")

predictionInputData, expectedPredictions = get_inputs_outputs(data[1].values, n, 200, len(data[1].values))
actualPredictions = [i[0] for i in net.sim(predictionInputData)]
expectedPredictions = [i[0] for i in expectedPredictions]

# plot expected and actual predictions
plt.plot(range(1900, 2005), actualPredictions, label="Actual predictions")
plt.plot(range(1900, 2005), expectedPredictions, label="Expected predictions")
plt.legend()
plt.title("Expected and actual predictions for year 1900-2005")
plt.xlabel('Year')
plt.ylabel('Activity')
plt.grid()
plt.show()

# get error vector between expected and actual predictions
error = []

for i in range(0, len(expectedPredictions)):
    error.append(abs(expectedPredictions[i] - actualPredictions[i]))

# plot error vector
plt.plot(range(1900, 2005), error, label="Error")
plt.legend()
plt.title("Error between expected and actual predictions")
plt.xlabel('Year')
plt.ylabel('Error')
plt.grid()
plt.show()

# calculate MAD
mad = np.median(error)

# calculate MSE
mse = np.square(np.subtract(expectedPredictions, actualPredictions)).mean()

print(f"MAD = {mad}")
print(f"MSE = {mse}")

# plot error histogram
plt.title("Error between expected and actual predictions histogram")
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.hist(np.hstack(error), 10, alpha=0.5, histtype='bar', ec='black')
plt.show()
