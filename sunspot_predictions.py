from file_reader import read_file

import neurolab as nl
import pylab as pl

# read sunspot data
attribute_types = {
    "Metai": "numeric",
    "Aktyvumas": "numeric"
}

data = read_file("sunspot.txt", attribute_types)

# input count for a single output
n = 9

# learning rate
learning_rate = 0.000005

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
net.trainf.defaults["goal"] = 100

# get training input and output
training_input = []
training_output = []

chunk_size = n + 1

for training_data in [data[1].values[i:i + chunk_size] for i in range(0, 200, chunk_size)]:
    training_input_set = []

    for index in range(len(training_data) - 1):
        training_input_set.append(training_data[index])

    training_input.append(training_input_set)
    training_output.append([training_data[len(training_data) - 1]])

# train the perceptron and get training error matrix
error = net.train(training_input, training_output, epochs=epochs, lr=learning_rate, show=epochs_to_show)

# plot error
pl.plot(error)
pl.xlabel('Epoch number')
pl.ylabel('Train error')
pl.grid()
pl.show()

# make predictions with trained net for the remaining data
print("Testing perceptron predictions...")

for prediction_data in [data[1].values[200:][i:i + chunk_size] for i in range(0, len(data[1].values[200:]), chunk_size)]:
    prediction_inputs = []

    for index in range(len(prediction_data) - 1):
        prediction_inputs.append(prediction_data[index])

    if len(prediction_inputs) != n:
        break

    expected_result = prediction_data[len(prediction_data) - 1]
    actual_result = net.sim([prediction_inputs])

    print('Expected prediction: {:>5}  Actual prediction: {:>5}'.format(expected_result, round(actual_result[0][0], 1)))
