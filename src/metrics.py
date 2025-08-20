##################
#python libraries#
##################
import numpy as np
import tensorflow.keras.backend as K
import tensorflow._api.v2.compat.v1 as tf
from tensorflow.keras.losses import mse

def rmse(y1, y2):
    y1, y2 = np.array(y1), np.array(y2)
    print('rmse:',np.sqrt(((y1-y2)**2).mean()))

def r2(y_true, y_pred):
    corr = np.corrcoef(y_true, y_pred)
    corr = corr[0,1]
    print('correlation:', corr)
    print('r2:', corr ** 2)
    
def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    print('mae:',np.mean(np.abs(y_true - predictions)))

def smape(y_true, predictions):
    numerador = np.abs(y_true - predictions)
    denominador =  (np.abs(y_true) + np.abs(predictions))
    return 100 * np.mean(numerador/denominador)

def mse_funct(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

def smape_loss(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    epsilon = 0.1
    summ = K.maximum(K.abs(y_true) + K.abs(y_pred) + epsilon, 0.5 + epsilon)
    smape = K.abs(y_pred - y_true) / summ * 2.0
    return smape
        
def vae_loss(por, reconstruction_por, rtp, reconstruction_rtp,
             ntg, reconstruction_ntg, perm, reconstruction_perm,
             z_mean, z_log_var):
    loss_recon_por = mse(K.flatten(por), K.flatten(reconstruction_por))
    loss_recon_rtp = mse(K.flatten(rtp), K.flatten(reconstruction_rtp))
    loss_recon_ntg = mse(K.flatten(ntg), K.flatten(reconstruction_ntg))
    loss_recon_perm = mse(K.flatten(perm), K.flatten(reconstruction_perm))
    loss_KL = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    total_loss = K.mean(loss_recon_por + loss_recon_rtp + loss_recon_ntg + loss_recon_perm + loss_KL)
    return total_loss

def correlation(x, y):    
    mx = tf.math.reduce_mean(x)
    my = tf.math.reduce_mean(y)
    xm, ym = x-mx, y-my
    r_num = tf.math.reduce_mean(tf.multiply(xm,ym))        
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
    return r_num / r_den