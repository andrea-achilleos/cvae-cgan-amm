import pandas as pd
import numpy as np
import random

barLen = {  1: 0.062, 
            2: 0.092, 
            3: 0.112, 
            4: 0.132, 
            5: 0.152, 
            6: 0.162, 
            7: 0.171, 
            8: 0.191, 
            9: 0.221, 
            10: 0.241, 
            11: 0.251, 
            12: 0.271, 
            13: 0.281, 
            14: 0.301, 
            15: 0.321 
          }

barSpa = {  1: 0.216, 
            2: 0.212, 
            3: 0.207,
            4: 0.189, 
            5: 0.161, 
            6: 0.166, 
            7: 0.171, 
            8: 0.134, 
            9: 0.257, 
            10: 0.234, 
            11: 0.230, 
            12: 0.207, 
            13: 0.203, 
            14: 0.175, 
            15: 0.152 
          }

def read_x_data_100_fun(name, two_by_two, test, padding, number):

    if test == False:
        num_of_data = 2768
    else:
        num_of_data = 328

    with open(name) as f:
        lines = f.read()

    lines = lines.replace('\n', '')
    lines = lines.replace('[', '')
    lines = lines.replace(']', '')
    lines = lines.replace('  ', ' ')
    lines = lines.replace('  ', ' ')
    lines = lines.replace('  ', ' ')

    x = lines.split('.')[:-1]

    data = np.zeros((num_of_data, 2, 2, 2))

    for count, i in enumerate(x):
        if i != '':
            one, two, three, four = i.split()
            #print(one, two, three, four)
            data[count, 0, 0, 0] = barLen[int(one)]
            data[count, 0, 0, 1] = barLen[int(two)]
            data[count, 0, 1, 0] = barLen[int(three)]
            data[count, 0, 1, 1] = barLen[int(four)]

            data[count, 1, 0, 0] = barSpa[int(one)]
            data[count, 1, 0, 1] = barSpa[int(two)]
            data[count, 1, 1, 0] = barSpa[int(three)]
            data[count, 1, 1, 1] = barSpa[int(four)]

               
    if padding == True:

        array = np.zeros((number*num_of_data, 2, 2, 2))

        count = 0 
        for i in range(num_of_data):
            for j in range(number):
                array[count+j, :, :, :] = data[i, :, :, :]
            count += number

        data = array
    
    return data

def read_y_data_100_fun(name, padding, number):
    with open(name) as f:
        lines = f.read()
        lines = lines.replace('\n', '')
        lines = lines.replace('[', '')
        lines = lines.replace(']', '')
        lines = lines.replace('  ', ' ')

        x = lines.split(' &')[:-1]

        Y = np.zeros((len(x), 100))
        for i in range(len(x)):
            splitted = x[i].split(' ')
            splitted = [float(x) for x in splitted if x is not '']
            Y[i, :] = splitted

    count = 0
    if padding == True:

        array = np.zeros((number*len(x), 100))

        for i in range(len(x)):
            for j in range(number):
                padding_extend = np.random.randint(0, 99)
                randint_idx = random.sample(range(1,99), padding_extend)
                array[count+j, :] = Y[i, :]
                for z in range(padding_extend):
                    array[count+j, randint_idx[z]] = -1

            count += number

        Y = array
    return Y