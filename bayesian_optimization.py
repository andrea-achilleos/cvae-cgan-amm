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
from bayes_opt import BayesianOptimization

warnings.filterwarnings('ignore')

from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

SECTION = 'cvae'
RUN_ID = 'bayesian_opt'
DATA_NAME = 'cell'
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER,'images'))

m = 10 # batch size must be divisible with number of data for some reason
decoder_out_dim = 8 #4 for 1 channel # dim of decoder output layer
activ = 'relu'

n_x = 8 
n_y = 100

histories = pd.DataFrame(columns = ['loss', 'KL_loss', 'recon_loss', 'val_loss', 'val_KL_loss', 'val_recon_loss'])

pad = False
num = 1000

X_train = read_x_data_100_fun('data/m.txt', True, False, padding = pad, number = num)
y_train = read_y_data_100_fun('data/p.txt', padding = pad, number = num)
X_test = read_x_data_100_fun('data/m_test.txt', True, True, padding = pad, number = num)
y_test = read_y_data_100_fun('data/p_test.txt', padding = pad, number = num)

X_train = X_train[:2700]
y_train = y_train[:2700]
X_test = X_test[:300]
y_test = y_test[:300]



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
y_train = y_train.reshape((len(y_train), 100))
y_test = y_test.reshape((len(y_test), 100))

def fit_with(lr, n_z, dim):

    encoder_dim1 = dim
    decoder_dim = dim
    n_z = int(n_z)
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

    # merge latent space with label
    zc = concat([z, label])

    #Decoder
    decoder_hidden = Dense(decoder_dim, activation=activ)
    decoder_out = Dense(decoder_out_dim, activation='sigmoid')
    h_p = decoder_hidden(zc)
    outputs = decoder_out(h_p)

    cvae = Model([X, label], outputs)

    #encoder = keras.models.load_model(str(os.path.join(RUN_FOLDER, "encoder_model")),  compile=False) #continue training
    encoder = Model([X, label], mu)

    d_in = Input(shape=(n_z+n_y,))
    d_h = decoder_hidden(d_in)
    d_out = decoder_out(d_h)

    #decoder = keras.models.load_model(str(os.path.join(RUN_FOLDER, "decoder_model")),  compile=False) #continue training
    decoder = Model(d_in, d_out)
   
    def vae_loss(y_true, y_pred):
        recon = K.mean(K.square(y_pred - y_true))

        kl = 0.5 * K.sum(K.exp(l_sigma) + K.square(mu) - 1. - l_sigma, axis=-1)
        return recon + kl

    def KL_loss(y_true, y_pred):
        return(0.5 * K.sum(K.exp(l_sigma) + K.square(mu) - 1. - l_sigma, axis=1))

    def recon_loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true))

    optim = Adam(lr = lr)
    cvae.compile(optimizer=optim, loss=vae_loss, metrics = [KL_loss, recon_loss])

    history = cvae.fit([X_train, y_train], X_train, verbose = 1, batch_size=m, epochs=3,#n_epoch,
                            validation_data = ([X_test, y_test], X_test), shuffle = True) #I ADDED SHUFFLE
                            #callbacks = [EarlyStopping(monitor = 'val_recon_loss', patience = 20)]
                            
    val_loss = float(list(pd.DataFrame(history.history)['val_loss'])[-1])

    return -val_loss


pbounds = {'n_z': (2, 500), 'lr': (1e-6, 1e-1), 'dim' : (200, 1000)}

optimizer = BayesianOptimization(
    f=fit_with,
    pbounds=pbounds,
    verbose=2,  
    random_state=1,
)

optimizer.maximize(init_points=100, n_iter=100,)

print('max: ', optimizer.max)