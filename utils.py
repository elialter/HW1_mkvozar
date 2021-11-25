import numpy as np
import math as m


def sigmoid(x):
    """
     Parameters
     ----------
     x : np.array input data

     Returns
     -------
     np.array
         sigmoid of the input
     """
    return 1 / (1 + m.exp(-x))
#    raise NotImplementedError("To be implemented")


def sigmoid_prime(x):
    """
         Parameters
         ----------
         x : np.array input data

         Returns
         -------
         np.array
             derivative of sigmoid of the input x
    """
    sig_prime = sigmoid(x) * (1 - sigmoid(x))
    return sig_prime
#    raise NotImplementedError("To be implemented")


def random_weights(sizes):
    """
         Parameters
         ----------
         sizes : list of sizes

         Returns
         -------
         list
             list of xavier initialized np arrays weight matrices
    """
    xavier_list = []
    loop_size = len(sizes) - 2
    for i in range(0, loop_size):
        xavier_list.append(xavier_initialization(sizes[i], sizes[i+1]))

    return xavier_list

#    raise NotImplementedError("To be implemented")


def zeros_weights(sizes):
    """
         Parameters
         ----------
         sizes : list of sizes

         Returns
         -------
         list
             list of zero np arrays weight matrices
    """
    zero_list = []
    loop_size = len(sizes) - 2
    for i in range(1, loop_size):
        zero_list.append(np.zeros(sizes[i], sizes[i+1]))

    return zero_list
#    raise NotImplementedError("To be implemented")


def zeros_biases(list):
    """
         Parameters
         ----------
         sizes : list of sizes

         Returns
         -------
         list
             list of zero np arrays bias matrices

    """
    zero_list = []
    loop_size = len(list) - 1
    for i in range(1, loop_size):
        zero_list.append(np.zeros(list[i]))

    return zero_list
#    raise NotImplementedError("To be implemented")


def create_batches(data, labels, batch_size):
    """
         Parameters
         ----------
         data : np.array of input data
         labels : np.array of input labels
         batch_size : int size of batch

         Returns
         -------
         list
             list of tuples of (data batch of batch_size, labels batch of batch_size)


    """
    data_list = []
    labels_list = []
    if len(data) % batch_size == 0:
        data_list = np.split(data, len(data) / batch_size)
        labels_list = np.split(labels, len(labels) / batch_size)

    else:
        eff_len = len(data) - (len(data) % batch_size)
        eff_data = data[0:eff_len]
        eff_labels = labels[0:eff_len]
        data_list = np.split(eff_data, eff_len / batch_size)
        data_list.append(data[eff_len:len(data)])
        labels_list = np.split(eff_labels, eff_len / batch_size)
        labels_list.append(labels[eff_len:len(data)])

    final_list = []
    for i in range(0, len(labels_list)):
        final_list.append((data_list[i], labels_list[i]))

    return final_list
#    raise NotImplementedError("To be implemented")


def add_elementwise(list1, list2):
    """
         Parameters
         ----------
         list1 : np.array of numbers
         list2 : np.array of numbers

         Returns
         -------
         list
             list of sum of each two elements by index
    """
    sum_list = []
    for i in range(0, len(list1)):
        sum_list += list1[i] + list2[i]

    return sum_list

#    raise NotImplementedError("To be implemented")

def xavier_initialization(m, n):
    xavier = 1 / (m ** 0.5)
    return np.random.uniform(low=-xavier, high=xavier, size=(m, n))
