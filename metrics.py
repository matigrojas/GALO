import os
import numpy as np
from sklearn import metrics
import neurolab as nl


def get_data(route_train,route_test):
    dataset_test = np.loadtxt(open(route_test, "rb"), delimiter=",", skiprows=0)
    dataset_train = np.loadtxt(open(route_train, "rb"), delimiter=",", skiprows=0)

    test_input = dataset_test[:,:-1]
    test_output = dataset_test[:, -1]

    train_input = dataset_train[:, :-1]
    train_output = dataset_train[:, -1]
    return test_input, test_output, train_input, train_output

def classify(net,X):
    y_pred = net.sim(X).reshape(len(X))
    y_pred = np.round(y_pred).astype(int)
    y_pred = np.clip(y_pred, 0, 1)
    return y_pred

def load_dataset_kfold(route_dataset, dataset_name, number_of_folds):
    test_input = []
    test_output = []
    train_input = []
    train_output = []

    for k in range(1,number_of_folds+1):
        test_dataset = dataset_name + f"Test_{k}.csv"
        train_dataset = dataset_name + f"Train_{k}.csv"

        data_test = os.path.join(route_dataset,test_dataset)
        data_train = os.path.join(route_dataset,train_dataset)

        x_test, y_test, x_train, y_train = get_data(data_train,data_test)

        test_input.append(x_test)
        test_output.append(y_test)
        train_input.append(x_train)
        train_output.append(y_train)

    return test_input, test_output, train_input, train_output

def define_network(fl,sl,solution):
    size_layers = [fl, sl, 1]

    net = nl.net.newff([[0, 1]] * fl, [sl, 1])

    split1 = size_layers[1] * size_layers[0]  # Hidden x input
    split2 = split1 + size_layers[1]
    split3 = split2 + size_layers[1]

    input_w = np.array(solution[0:split1]).reshape(size_layers[1], size_layers[0])
    layer_w = np.array(solution[split1:split2]).reshape(1, size_layers[1])
    input_bias = np.array(solution[split2:split3]).reshape(1, size_layers[1])
    bias_output = solution[split3:split3 + 1]

    net.layers[0].np['w'][:] = input_w
    net.layers[1].np['w'][:] = layer_w
    net.layers[0].np['b'][:] = input_bias
    net.layers[1].np['b'][:] = bias_output

    return net

def classification_metrics_kfold(solution, route_dataset,dataset_name, number_of_folds:int = 10):
    x_test,y_test,x_train,y_train = load_dataset_kfold(route_dataset,dataset_name,number_of_folds)

    acc = 0.0
    prec = 0.0
    tn, fp, fn, tp = 0.0, 0.0, 0.0, 0.0

    fl = np.shape(x_test[0])[1]  # First Layer
    sl = (fl * 2) + 1 #Second Layer

    net = define_network(fl,sl,solution)

    print("\nTRAIN:\n")
    for k in range(number_of_folds):
        y_pred = classify(net,x_train[k])
        acc += metrics.accuracy_score(y_train[k], y_pred, normalize=True)/number_of_folds
        prec += metrics.precision_score(y_train[k], y_pred)/number_of_folds
        conf_matrix = metrics.confusion_matrix(y_train[k], y_pred).flatten()
        tn += conf_matrix[0]/number_of_folds
        fp += conf_matrix[1]/number_of_folds
        fn += conf_matrix[2]/number_of_folds
        tp += conf_matrix[3]/number_of_folds

    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")

    print(f"TN: {tn} FP: {fp} FN: {fn/number_of_folds} TP: {tp/number_of_folds}")

    specifity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    print("Specifity: {}".format(specifity))
    print("Sensitivity: {}".format(sensitivity))
    print("G-Mean: {}".format(np.sqrt(specifity * sensitivity)))

    print("\nTEST:\n")

    acc = 0.0
    prec = 0.0
    tn, fp, fn, tp = 0.0, 0.0, 0.0, 0.0

    for k in range(number_of_folds):
        y_pred = classify(net, x_test[k])
        acc += metrics.accuracy_score(y_test[k], y_pred, normalize=True) / number_of_folds
        prec += metrics.precision_score(y_test[k], y_pred) / number_of_folds
        conf_matrix = metrics.confusion_matrix(y_test[k], y_pred).flatten()
        tn += conf_matrix[0] / number_of_folds
        fp += conf_matrix[1] / number_of_folds
        fn += conf_matrix[2] / number_of_folds
        tp += conf_matrix[3] / number_of_folds

    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"TN: {tn} FP: {fp} FN: {fn} TP: {tp}")

    specifity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    print("Specifity: {}".format(specifity))
    print("Sensitivity: {}".format(sensitivity))
    print("G-Mean: {}".format(np.sqrt(specifity*sensitivity)))