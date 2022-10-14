import warnings
import numpy as np
from keras.layers import Input, Dense, Lambda
from keras.layers import Concatenate as concat
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import imageio
import os
from read_data import read_y_data_100_fun, read_x_data_100_fun
import keras
from plot_keras_history import plot_history
import pandas as pd
from bayes_opt import BayesianOptimization
from numpy.linalg import norm

warnings.filterwarnings('ignore')

from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

SECTION = 'cvae'
RUN_ID = 'main'
DATA_NAME = 'cell'
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER,'images'))

def read_one_y(name, padding, number, padding_extend):
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

        if padding == True:
            array_minus = []
            for i in range(padding_extend):
              array_minus.append(-1)
            randint_idx = np.random.randint(0, 100, size = padding_extend)
            print('size', randint_idx.shape)
            Yo = Y[0]
            for i in randint_idx:
              Yo[i] = -1
            
            Y = Yo

    return Y

def cm_to_inch(value):
    return value/2.54

leg = ['Generated', 'Input']

fig_sim, axs = plt.subplots(3, 3, figsize=(9,9))
fig_error, axs_error = plt.subplots(3, 3, figsize=(9,9))

pad = False
num = 1000
padding_extend = 1
Ygens = read_one_y('data/100_pressures_val.txt', padding = pad, number = num, padding_extend=padding_extend)
Yinps = 4785.90273 *read_one_y('data/Y_train_best.txt', padding = pad, number = num, padding_extend=padding_extend)

y_train = read_y_data_100_fun('data/p.txt', padding = pad, number = num)[:2600, :]
means = np.mean(y_train, axis=0)
standard_deviations = np.std(y_train, axis= 0)
la = means - (0.5*standard_deviations)
indexes = np.arange(0, 100, 1)
res = sorted(range(len(standard_deviations)), key = lambda sub: standard_deviations[sub])
means = means[res]
standard_deviations = standard_deviations[res]

under_line = means- (0.5*standard_deviations)
over_line = means+ (0.5*standard_deviations)

fig = plt.figure()
plt.plot(means.reshape(100), color = 'black')
plt.fill_between(indexes, under_line.reshape(100), over_line.reshape(100), alpha=1)
plt.ylabel('Mean abolute ppressure [Pa]', fontsize=14)
plt.xlabel('Points', fontsize=14)
fig.savefig(os.path.join(RUN_FOLDER, "images/standard_deviations.png"))
print(max(means))
print(max(standard_deviations))

last = np.sort(y_train[:, -1])
print(max(last))
fig = plt.figure()
plt.plot(last, color = 'red')
plt.ylabel('Absolute pressure [Pa]', fontsize=14)
plt.xlabel('Simulations used in training', fontsize=14)
fig.savefig(os.path.join(RUN_FOLDER, "images/last_point.png"))

fig = plt.figure()
plt.plot(y_train[1], color = 'red')
plt.plot(y_train[2], color = 'yellow')
plt.plot(y_train[3], color = 'green')
plt.plot(y_train[4], color = 'blue')
plt.plot(y_train[5], color = 'purple')
plt.ylabel('Absolute pressure [Pa]', fontsize=14)
plt.xlabel('Points', fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(RUN_FOLDER, "images/pres_5.png"))

fig = plt.figure()
plt.plot(y_train[4], color = 'blue')
plt.ylabel('Absolute pressure [Pa]', fontsize=14)
plt.xlabel('Points', fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(RUN_FOLDER, "images/pres.png"))

Ygen = Ygens[0]
Yinp = Yinps[0]

Ygen = Ygen[res]
Yinp = Yinp[res]

mean = np.mean(Yinp)
error = np.abs(Ygen - Yinp) / mean

cosine = np.dot(Ygen,Yinp)/(norm(Ygen)*norm(Yinp))
print("Cosine Similarity:", cosine)
print('Mean error:', np.mean(error), '+-', np.std(error))

fig = plt.figure(figsize=(cm_to_inch(20), cm_to_inch(20)))
plt.plot(Ygen, color='black', linewidth=2.0)
plt.plot(Yinp, color='red', linewidth=2.0)
plt.ylabel('Pressure [Pa]', fontsize=30)
plt.xlabel('Points', fontsize=30)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.legend(leg, prop={'size': 25})
plt.tight_layout()
plt.ylim(0, 600)
fig.savefig(os.path.join(RUN_FOLDER, "images/similarity_1.png"), dpi=1000)

fig = plt.figure(figsize=(cm_to_inch(20), cm_to_inch(20)))
plt.plot(error, color='red', linewidth=2.0)
plt.ylabel('Relative error [-]', fontsize=30)
plt.xlabel('Points', fontsize=30)
plt.ylim(0, 6)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.tight_layout()
fig.savefig(os.path.join(RUN_FOLDER, "images/error1.png"), dpi=1000)

axs[0, 0].plot(Ygen, color='black', linewidth=1.0)
axs[0, 0].plot(Yinp, color='red', linewidth=1.0)

axs_error[0, 0].plot(error, color='black', linewidth=1.0)
axs_error[0, 0].set_ylim(0, 3.8)

Ygen = Ygens[1]
Yinp = Yinps[1]

Ygen = Ygen[res]
Yinp = Yinp[res]

mean = np.mean(Yinp)
error = np.abs(Ygen - Yinp) / mean

cosine = np.dot(Ygen,Yinp)/(norm(Ygen)*norm(Yinp))
print("Cosine Similarity:", cosine)
print('Mean error:', np.mean(error), '+-', np.std(error))

fig = plt.figure(figsize=(cm_to_inch(20), cm_to_inch(20)))
plt.plot(Ygen, color='black', linewidth=2.0)
plt.plot(Yinp, color='red', linewidth=2.0)
plt.ylabel('Pressure [Pa]', fontsize=30)
plt.xlabel('Points', fontsize=30)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.legend(leg, prop={'size': 25})
plt.tight_layout()
plt.ylim(0, 600)
fig.savefig(os.path.join(RUN_FOLDER, "images/similarity_2.png"))

fig = plt.figure(figsize=(cm_to_inch(20), cm_to_inch(20)))
plt.plot(error, color='red', linewidth=2.0)
plt.ylabel('Relative error [-]', fontsize=30)
plt.xlabel('Points', fontsize=30)
plt.ylim(0, 6)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.tight_layout()
fig.savefig(os.path.join(RUN_FOLDER, "images/error2.png"))

axs[0, 1].plot(Ygen, color='black', linewidth=1.0)
axs[0, 1].plot(Yinp, color='red', linewidth=1.0)

axs_error[0, 1].plot(error, color='black', linewidth=1.0)
axs_error[0, 1].set_ylim(0, 3.8)

Ygen = Ygens[2]
Yinp = Yinps[2]

Ygen = Ygen[res]
Yinp = Yinp[res]

mean = np.mean(Yinp)
error = np.abs(Ygen - Yinp) / mean

cosine = np.dot(Ygen,Yinp)/(norm(Ygen)*norm(Yinp))
print("Cosine Similarity:", cosine)
print('Mean error:', np.mean(error), '+-', np.std(error))

fig = plt.figure(figsize=(cm_to_inch(20), cm_to_inch(20)))
plt.plot(Ygen, color='black', linewidth=2.0)
plt.plot(Yinp, color='red', linewidth=2.0)
plt.ylabel('Pressure [Pa]', fontsize=30)
plt.xlabel('Points', fontsize=30)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.legend(leg, prop={'size': 25})
plt.tight_layout()
plt.ylim(0, 600)
fig.savefig(os.path.join(RUN_FOLDER, "images/similarity_3.png"))

fig = plt.figure(figsize=(cm_to_inch(20), cm_to_inch(20)))
plt.plot(error, color='red', linewidth=2.0)
plt.ylabel('Relative error [-]', fontsize=30)
plt.xlabel('Points', fontsize=30)
plt.ylim(0, 6)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.tight_layout()
fig.savefig(os.path.join(RUN_FOLDER, "images/error3.png"))

axs[0, 2].plot(Ygen, color='black', linewidth=1.0)
axs[0, 2].plot(Yinp, color='red', linewidth=1.0)

axs_error[0, 2].plot(error, color='black', linewidth=1.0)
axs_error[0, 2].set_ylim(0, 3.8)

Ygen = Ygens[3]
Yinp = Yinps[3]

Ygen = Ygen[res]
Yinp = Yinp[res]

print(np.argmax(Yinp))
mean = np.mean(Yinp)
error = np.abs(Ygen - Yinp) / mean

cosine = np.dot(Ygen,Yinp)/(norm(Ygen)*norm(Yinp))
print("Cosine Similarity:", cosine)
print('Mean error:', np.mean(error), '+-', np.std(error))

fig = plt.figure(figsize=(cm_to_inch(20), cm_to_inch(20)))
plt.plot(Ygen, color='black', linewidth=2.0)
plt.plot(Yinp, color='red', linewidth=2.0)
plt.ylabel('Pressure [Pa]', fontsize=30)
plt.xlabel('Points', fontsize=30)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.legend(leg, prop={'size': 25})
plt.tight_layout()
plt.ylim(0, 600)
fig.savefig(os.path.join(RUN_FOLDER, "images/similarity_4.png"))

fig = plt.figure(figsize=(cm_to_inch(20), cm_to_inch(20)))
plt.plot(error, color='red', linewidth=2.0)
plt.ylabel('Relative error [-]', fontsize=30)
plt.xlabel('Points', fontsize=30)
plt.ylim(0, 6)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.tight_layout()
fig.savefig(os.path.join(RUN_FOLDER, "images/error4.png"))

print(max(error))

axs[1, 0].plot(Ygen, color='black', linewidth=1.0)
axs[1, 0].plot(Yinp, color='red', linewidth=1.0)

axs_error[1, 0].plot(error, color='black', linewidth=1.0)
axs_error[1, 0].set_ylim(0, 3.8)

Ygen = Ygens[4]
Yinp = Yinps[4]

Ygen = Ygen[res]
Yinp = Yinp[res]

mean = np.mean(Yinp)
error = np.abs(Ygen - Yinp) / mean

cosine = np.dot(Ygen,Yinp)/(norm(Ygen)*norm(Yinp))
print("Cosine Similarity:", cosine)
print('Mean error:', np.mean(error), '+-', np.std(error))

fig = plt.figure(figsize=(cm_to_inch(20), cm_to_inch(20)))
plt.plot(Ygen, color='black', linewidth=2.0)
plt.plot(Yinp, color='red', linewidth=2.0)
plt.ylabel('Pressure [Pa]', fontsize=30)
plt.xlabel('Points', fontsize=30)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.legend(leg, prop={'size': 25})
plt.tight_layout()
plt.ylim(0, 600)
fig.savefig(os.path.join(RUN_FOLDER, "images/similarity_5.png"))

fig = plt.figure(figsize=(cm_to_inch(20), cm_to_inch(20)))
plt.plot(error, color='red', linewidth=2.0)
plt.ylabel('Relative error [-]', fontsize=30)
plt.xlabel('Points', fontsize=30)
plt.ylim(0, 6)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.tight_layout()
fig.savefig(os.path.join(RUN_FOLDER, "images/error5.png"))

axs[1, 1].plot(Ygen, color='black', linewidth=1.0)
axs[1, 1].plot(Yinp, color='red', linewidth=1.0)

axs_error[1, 1].plot(error, color='black', linewidth=1.0)
axs_error[1, 1].set_ylim(0, 3.8)

Ygen = Ygens[5]
Yinp = Yinps[5]

Ygen = Ygen[res]
Yinp = Yinp[res]

mean = np.mean(Yinp)
error = np.abs(Ygen - Yinp) / mean

cosine = np.dot(Ygen,Yinp)/(norm(Ygen)*norm(Yinp))
print("Cosine Similarity:", cosine)
print('Mean error:', np.mean(error), '+-', np.std(error))

fig = plt.figure(figsize=(cm_to_inch(20), cm_to_inch(20)))
plt.plot(Ygen, color='black', linewidth=2.0)
plt.plot(Yinp, color='red', linewidth=2.0)
plt.ylabel('Pressure [Pa]', fontsize=30)
plt.xlabel('Points', fontsize=30)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.legend(leg, prop={'size': 25})
plt.tight_layout()
plt.ylim(0, 3800)
fig.savefig(os.path.join(RUN_FOLDER, "images/similarity_6.png"))

fig = plt.figure(figsize=(cm_to_inch(20), cm_to_inch(20)))
plt.plot(error, color='red', linewidth=2.0)
plt.ylabel('Relative error [-]', fontsize=30)
plt.xlabel('Points', fontsize=30)
plt.ylim(0, 6)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.tight_layout()
fig.savefig(os.path.join(RUN_FOLDER, "images/error6.png"))

axs[1, 2].plot(Ygen, color='black', linewidth=1.0)
axs[1, 2].plot(Yinp, color='red', linewidth=1.0)

axs_error[1, 2].plot(error, color='black', linewidth=1.0)
axs_error[1, 2].set_ylim(0, 3.8)

Ygen = Ygens[6]
Yinp = Yinps[6]

Ygen = Ygen[res]
Yinp = Yinp[res]

mean = np.mean(Yinp)
error = np.abs(Ygen - Yinp) / mean

cosine = np.dot(Ygen,Yinp)/(norm(Ygen)*norm(Yinp))
print("Cosine Similarity:", cosine)
print('Mean error:', np.mean(error), '+-', np.std(error))

fig = plt.figure(figsize=(cm_to_inch(20), cm_to_inch(20)))
plt.plot(Ygen, color='black', linewidth=2.0)
plt.plot(Yinp, color='red', linewidth=2.0)
plt.ylabel('Pressure [Pa]', fontsize=30)
plt.xlabel('Points', fontsize=30)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.legend(leg, prop={'size': 25})
plt.tight_layout()
plt.ylim(0, 3800)
fig.savefig(os.path.join(RUN_FOLDER, "images/similarity_7.png"))

fig = plt.figure(figsize=(cm_to_inch(20), cm_to_inch(20)))
plt.plot(error, color='red', linewidth=2.0)
plt.ylabel('Relative error [-]', fontsize=30)
plt.xlabel('Points', fontsize=30)
plt.ylim(0, 6)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.tight_layout()
fig.savefig(os.path.join(RUN_FOLDER, "images/error7.png"))

axs[2, 0].plot(Ygen, color='black', linewidth=1.0)
axs[2, 0].plot(Yinp, color='red', linewidth=1.0)

axs_error[2, 0].plot(error, color='black', linewidth=1.0)
axs_error[2, 0].set_ylim(0, 3.8)

Ygen = Ygens[7]
Yinp = Yinps[7]

Ygen = Ygen[res]
Yinp = Yinp[res]

mean = np.mean(Yinp)
error = np.abs(Ygen - Yinp) / mean

cosine = np.dot(Ygen,Yinp)/(norm(Ygen)*norm(Yinp))
print("Cosine Similarity:", cosine)
print('Mean error:', np.mean(error), '+-', np.std(error))

fig = plt.figure(figsize=(cm_to_inch(20), cm_to_inch(20)))
plt.plot(Ygen, color='black', linewidth=2.0)
plt.plot(Yinp, color='red', linewidth=2.0)
plt.ylabel('Pressure [Pa]', fontsize=30)
plt.xlabel('Points', fontsize=30)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.legend(leg, prop={'size': 25})
plt.tight_layout()
plt.ylim(0, 3800)
fig.savefig(os.path.join(RUN_FOLDER, "images/similarity_8.png"))

fig = plt.figure(figsize=(cm_to_inch(20), cm_to_inch(20)))
plt.plot(error, color='red', linewidth=2.0)
plt.ylabel('Relative error [-]', fontsize=30)
plt.xlabel('Points', fontsize=30)
plt.ylim(0, 6)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.tight_layout()
fig.savefig(os.path.join(RUN_FOLDER, "images/error8.png"))

axs[2, 1].plot(Ygen, color='black', linewidth=1.0)
axs[2, 1].plot(Yinp, color='red', linewidth=1.0)

axs_error[2, 1].plot(error, color='black', linewidth=1.0)
axs_error[2, 1].set_ylim(0, 3.8)

Ygen = Ygens[8]
Yinp = Yinps[8]

Ygen = Ygen[res]
Yinp = Yinp[res]

mean = np.mean(Yinp)
error = np.abs(Ygen - Yinp) / mean

cosine = np.dot(Ygen,Yinp)/(norm(Ygen)*norm(Yinp))
print("Cosine Similarity:", cosine)
print('Mean error:', np.mean(error), '+-', np.std(error))

fig = plt.figure(figsize=(cm_to_inch(20), cm_to_inch(20)))
plt.plot(Ygen, color='black', linewidth=2.0)
plt.plot(Yinp, color='red', linewidth=2.0)
plt.ylabel('Pressure [Pa]', fontsize=30)
plt.xlabel('Points', fontsize=30)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.legend(leg, prop={'size': 25})
plt.tight_layout()
plt.ylim(0, 600)
fig.savefig(os.path.join(RUN_FOLDER, "images/similarity_9.png"))

fig = plt.figure(figsize=(cm_to_inch(20), cm_to_inch(20)))
plt.plot(error, color='red', linewidth=2.0)
plt.ylabel('Relative error [-]', fontsize=30)
plt.xlabel('Points', fontsize=30)
plt.ylim(0, 6)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.tight_layout()
fig.savefig(os.path.join(RUN_FOLDER, "images/error9.png"))

axs[2, 2].plot(Ygen, color='black', linewidth=1.0)
axs[2, 2].plot(Yinp, color='red', linewidth=1.0)

axs_error[2, 2].plot(error, color='black', linewidth=1.0)
axs_error[2, 2].set_ylim(0, 3.8)

fig_sim.savefig(os.path.join(RUN_FOLDER, "images/similarities.png"))
fig_error.savefig(os.path.join(RUN_FOLDER, "images/errors.png"))

