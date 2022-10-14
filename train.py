import warnings
import numpy as np
from keras.layers import Input, Dense, Lambda
from keras.layers import concatenate as concat
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

pad = False
num = 100

X_train = read_x_data_100_fun('data/m.txt', True, False, padding = pad, number = num)
y_train = read_y_data_100_fun('data/p.txt', padding = pad, number = num)
X_test = read_x_data_100_fun('data/m_test.txt', True, True, padding = pad, number = num)
y_test = read_y_data_100_fun('data/p_test.txt', padding = pad, number = num)

print(X_train.shape) 
print(X_train[0])
print(y_train.shape) 

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

y_max = y_train.max()
with open(os.path.join(RUN_FOLDER, "images/Y_norm.txt"), mode = 'a') as file_object:
    print(y_max,'&', file=file_object)
y_train = y_train.astype('float32') / y_max
y_test = y_test.astype('float32') / y_max

n_pixels = np.prod(X_train.shape[1:])
X_train = X_train.reshape((len(X_train), n_pixels))
X_test = X_test.reshape((len(X_test), n_pixels))


m = 8 #batch size 
n_z = 2 # latent space size
encoder_dim1 = 512 # dim of encoder hidden layer
decoder_dim = 512 # dim of decoder hidden layer
decoder_out_dim = 8 #4 for 1 channel # dim of decoder output layer
activ = 'relu'
optim = Adam(lr=0.001)

n_x = X_train.shape[1]
n_y = y_train.shape[1]

n_epoch = 200

X = Input(shape=(n_x,))
label = Input(shape=(n_y,))

inputs = concat([X, label])

encoder_h = Dense(encoder_dim1, activation=activ)(inputs)
mu = Dense(n_z, activation='linear')(encoder_h)
l_sigma = Dense(n_z, activation='linear')(encoder_h)

def sample_z(args):
    mu, l_sigma = args
    eps = K.random_normal(shape=(m, n_z), mean=0., stddev=1.)
    return mu + K.exp(l_sigma / 2) * eps

z = Lambda(sample_z, output_shape = (n_z, ))([mu, l_sigma])

zc = concat([z, label])

decoder_hidden = Dense(decoder_dim, activation=activ)
decoder_out = Dense(decoder_out_dim, activation='sigmoid')
h_p = decoder_hidden(zc)
outputs = decoder_out(h_p)

def vae_loss(y_true, y_pred):
    recon = K.mean(K.square(y_pred - y_true))
    kl = 0.5 * K.sum(K.exp(l_sigma) + K.square(mu) - 1. - l_sigma, axis=-1)
    return recon + kl

def KL_loss(y_true, y_pred):
	return(0.5 * K.sum(K.exp(l_sigma) + K.square(mu) - 1. - l_sigma, axis=1))

def recon_loss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

cvae = Model([X, label], outputs)

encoder = Model([X, label], mu)

d_in = Input(shape=(n_z+n_y,))
d_h = decoder_hidden(d_in)
d_out = decoder_out(d_h)

decoder = Model(d_in, d_out)
cvae.summary()


cvae.compile(optimizer=optim, loss=vae_loss, metrics = [KL_loss, recon_loss])


histories = []

for _ in range(n_epoch):
    print('epoch: ', n_epoch+1)
    X_train = read_x_data_100_fun('data/m.txt', True, False, padding = pad, number = num)
    y_train = read_y_data_100_fun('data/p.txt', padding = pad, number = num)
    X_test = read_x_data_100_fun('data/m_test.txt', True, True, padding = pad, number = num)
    y_test = read_y_data_100_fun('data/p_test.txt', padding = pad, number = num)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    y_max = y_train.max()
    y_train = y_train.astype('float32') / y_max
    y_test = y_test.astype('float32') / y_max

    n_pixels = np.prod(X_train.shape[1:])
    X_train = X_train.reshape((len(X_train), n_pixels))
    X_test = X_test.reshape((len(X_test), n_pixels))
    history = cvae.fit([X_train, y_train], X_train, verbose = 1, batch_size=m, epochs=1,
							validation_data = ([X_test, y_test], X_test), shuffle = True)
    if _ == n_epoch - 1:
        histories.append(history)
hist_df = pd.DataFrame(history.history)

hist_csv_file = str(os.path.join(RUN_FOLDER, "images/history.csv"))
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

training_history = pd.read_csv(str(os.path.join(RUN_FOLDER, "images/history.csv")))
print(training_history)

cvae.save(str(os.path.join(RUN_FOLDER, "cvae_model")))
decoder.save(str(os.path.join(RUN_FOLDER, "decoder_model")))
encoder.save(str(os.path.join(RUN_FOLDER, "encoder_model")))

z_train = encoder.predict([X_train, y_train])
encodings= np.asarray(z_train)
encodings = encodings.reshape(X_train.shape[0], n_z)

def construct_numvec(digit, z = None):
    out = np.zeros((1, n_z + n_y))
    out[:, -100:] = digit
    if z is None:
        return(out)
    else: 
        for i in range(len(z)):
            out[:,i] = z[i]
        return(out)

r, c = 3, 3
idx = np.random.randint(0, y_train.shape[0], 32)
true_imgs = X_train[idx]

fig, axs = plt.subplots(r, c, figsize=(9,9))
fig1, axs1 = plt.subplots(r, c, figsize=(9,9))
cnt = 0
for i in range(r):
    for j in range(c):
        axs[i,j].imshow(true_imgs[cnt].reshape(2, 2, 2)[0], cmap = 'gray')
        axs1[i,j].imshow(true_imgs[cnt].reshape(2, 2, 2)[1], cmap = 'gray')
        with open(os.path.join(RUN_FOLDER, "images/X_train.txt"), mode = 'a') as file_object:
            print(true_imgs[cnt].reshape(2, 2, 2),'&', file=file_object)
        axs[i,j].axis('off')
        axs1[i,j].axis('off')
        cnt += 1
fig.savefig(os.path.join(RUN_FOLDER, "images/X_train1.png"))
fig1.savefig(os.path.join(RUN_FOLDER, "images/X_train2.png"))

fig, axs = plt.subplots(r, c, figsize=(9,9))
fig1, axs1 = plt.subplots(r, c, figsize=(9,9))
cnt = 0
for i in range(r):
    for j in range(c):
        sample_3 = construct_numvec(y_train[idx][cnt])
        with open(os.path.join(RUN_FOLDER, "images/Y_train.txt"), mode = 'a') as file_object:
            print(y_train[idx][cnt],'&', file=file_object)
        axs[i,j].imshow(decoder.predict(sample_3).reshape(2, 2, 2)[0], cmap = 'gray')
        axs1[i,j].imshow(decoder.predict(sample_3).reshape(2, 2, 2)[1], cmap = 'gray')
        with open(os.path.join(RUN_FOLDER, "images/prediction_train.txt"), mode = 'a') as file_object:
            print(decoder.predict(sample_3).reshape(2, 2, 2),'&', file=file_object)
        axs[i,j].axis('off')
        axs1[i,j].axis('off')
        cnt += 1
fig.savefig(os.path.join(RUN_FOLDER, "images/prediction_train1.png"))
fig1.savefig(os.path.join(RUN_FOLDER, "images/prediction_train2.png"))

#many reconstructions from test data
r, c = 3, 3
idx = np.random.randint(0, y_test.shape[0], 32)
true_imgs = X_test[idx]

fig, axs = plt.subplots(r, c, figsize=(9,9))
fig1, axs1 = plt.subplots(r, c, figsize=(9,9))
cnt = 0
for i in range(r):
    for j in range(c):
        axs[i,j].imshow(true_imgs[cnt].reshape(2, 2, 2)[0], cmap = 'gray')
        axs1[i,j].imshow(true_imgs[cnt].reshape(2, 2, 2)[1], cmap = 'gray')
        with open(os.path.join(RUN_FOLDER, "images/X_test.txt"), mode = 'a') as file_object:
            print(true_imgs[cnt].reshape(2, 2, 2),'&', file=file_object)
        axs[i,j].axis('off')
        axs1[i,j].axis('off')
        cnt += 1
fig.savefig(os.path.join(RUN_FOLDER, "images/X_test1.png"))
fig1.savefig(os.path.join(RUN_FOLDER, "images/X_test2.png"))

fig, axs = plt.subplots(r, c, figsize=(9,9))
fig1, axs1 = plt.subplots(r, c, figsize=(9,9))
cnt = 0
for i in range(r):
    for j in range(c):
        sample_3 = construct_numvec(y_test[idx][cnt])
        with open(os.path.join(RUN_FOLDER, "images/Y_test.txt"), mode = 'a') as file_object:
            print(y_test[idx][cnt],'&', file=file_object)
        axs[i,j].imshow(decoder.predict(sample_3).reshape(2, 2, 2)[0], cmap = 'gray')
        axs1[i,j].imshow(decoder.predict(sample_3).reshape(2, 2, 2)[1], cmap = 'gray')
        with open(os.path.join(RUN_FOLDER, "images/prediction_test.txt"), mode = 'a') as file_object:
            print(decoder.predict(sample_3).reshape(2, 2, 2),'&', file=file_object)
        axs[i,j].axis('off')
        axs1[i,j].axis('off')
        cnt += 1
fig.savefig(os.path.join(RUN_FOLDER, "images/prediction_test1.png"))
fig1.savefig(os.path.join(RUN_FOLDER, "images/prediction_test2.png"))

fig = plt.figure()
plt.plot(y_test[0]*y_max, color='black', linewidth=1.0)
plt.plot(y_test[1]*y_max, color='black', linewidth=1.0)
plt.plot(y_test[100]*y_max, color='black', linewidth=1.0)
plt.plot(y_test[101]*y_max, color='black', linewidth=1.0)
plt.plot(y_test[50]*y_max, color='black', linewidth=1.0)
plt.plot(y_max*np.sum(y_train, axis=0)/y_train.shape[0], color='red', linewidth=1.0)
plt.ylabel('pressure', fontsize=14)
plt.xlabel('points', fontsize=14)
fig.savefig(os.path.join(RUN_FOLDER, "images/pressures.png"))

average_pressures = np.sum(y_train, axis=0)/y_train.shape[0]
with open(os.path.join(RUN_FOLDER, "images/Y_average.txt"), mode = 'a') as file_object:
    print(average_pressures,'&', file=file_object)

fig, axs = plt.subplots(1, 1, figsize=(9,9))
fig1, axs1 = plt.subplots(1, 1, figsize=(9,9))

sample_3 = construct_numvec(average_pressures)
axs.imshow(decoder.predict(sample_3).reshape(2, 2, 2)[0], cmap = 'gray')
axs1.imshow(decoder.predict(sample_3).reshape(2, 2, 2)[1], cmap = 'gray')
with open(os.path.join(RUN_FOLDER, "images/prediction_average.txt"), mode = 'a') as file_object:
    print(decoder.predict(sample_3).reshape(2, 2, 2),'&', file=file_object)
axs.axis('off')
axs1.axis('off')
fig.savefig(os.path.join(RUN_FOLDER, "images/prediction_average1.png"))
fig1.savefig(os.path.join(RUN_FOLDER, "images/prediction_average2.png"))


# Visualize one metasurface
fig, axes = plt.subplots(nrows=1, ncols=2)
  
for i, ax in enumerate(axes.flat):
    im = ax.imshow(true_imgs[0].reshape(2, 2, 2)[i], cmap='gray')
    print('print', true_imgs[0].reshape(2, 2, 2)[i])
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    if i == 0:
        ax.title.set_text('Bar length, '+r'$\frac{b_\ell}{\lambda_0}$')
    else:
        ax.title.set_text('Inter-bar spacing, '+r'$\frac{b_s}{\lambda_0}$')

plt.colorbar(im, ax=axes.ravel().tolist(), label='Ratio of bar length/ inter-bar spacing to operating wavelegth, '+r'$\frac{b_\ell,s}{\lambda_0}$', orientation="horizontal")
plt.show()
fig.savefig(os.path.join(RUN_FOLDER, "images/data1.png"))

#plot training history
plot_history(
    histories,
    show_standard_deviation=False,
    show_average=True
)
plt.savefig(os.path.join(RUN_FOLDER,"images/training_history.png"))
plt.close()