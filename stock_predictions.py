from file_reader import read_file
import neurolab as nl
import numpy as np
from attributes import print_numeric_attribute_info

# read stock data
attribute_types = {
    "Lag1": "numeric",
    "Lag2": "numeric",
    "Lag3": "numeric",
    "Lag4": "numeric",
    "Lag5": "numeric",
    "Volume": "numeric",
    "Today": "numeric"
}

data = read_file("stock.csv", attribute_types)

print_numeric_attribute_info(data)

# learning rate
learning_rate = 0.000005

# maximum number of epochs
epochs = 500

# after how many epochs to display error
epochs_to_show = 100

# get input ranges
input_ranges = [
    [data[0].min_value(), data[0].max_value()],
    [data[1].min_value(), data[1].max_value()],
    [data[2].min_value(), data[2].max_value()],
    [data[3].min_value(), data[3].max_value()],
    [data[4].min_value(), data[4].max_value()],
    [data[5].min_value(), data[5].max_value()]
]


def perform_simulation(range_start: int, range_end: int) -> float:
    # create a perceptron with 6 inputs
    net = nl.net.newp(input_ranges, 1)

    # set transfer function to linear
    net.layers[0].transf = nl.trans.PureLin()

    # set error function to MSE
    net.errorf = nl.error.MSE()

    # set training error goal
    net.trainf.defaults["goal"] = 1

    # get training input and output
    training_input = []
    training_output = []

    for i in range(range_start, int(range_end - ((range_end - range_start) * 0.25))):
        training_input.append([
            data[0].values[i],
            data[1].values[i],
            data[2].values[i],
            data[3].values[i],
            data[4].values[i],
            data[5].values[i]
        ])

        training_output.append([data[6].values[i]])

    # train the perceptron
    net.train(training_input, training_output, epochs=epochs, lr=learning_rate, show=epochs_to_show)

    # make predictions with trained net for the remaining data
    prediction_inputs = []
    expected_predictions = []

    for i in range(int(range_end - ((range_end - range_start) * 0.25)), range_end):
        prediction_inputs.append([
            data[0].values[i],
            data[1].values[i],
            data[2].values[i],
            data[3].values[i],
            data[4].values[i],
            data[5].values[i]
        ])

        expected_predictions.append(data[6].values[i])

    actual_predictions = [i[0] for i in net.sim(prediction_inputs)]

    error = []

    for i in range(0, len(expected_predictions)):
        error.append(abs(expected_predictions[i] - actual_predictions[i]))

    return np.average(error)


errors = [
    perform_simulation(0, 100),
    perform_simulation(100, 200),
    perform_simulation(200, 300),
    perform_simulation(300, 400),
    perform_simulation(400, 500),
    perform_simulation(500, 600),
    perform_simulation(600, 700),
    perform_simulation(700, 800),
    perform_simulation(800, 900),
    perform_simulation(900, 1000),
]

print(f"Average error: {np.average(errors)}")
