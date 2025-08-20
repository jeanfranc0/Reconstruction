import os
import joblib
import numpy as np

def denormalization_computing(scaler, data, normalizers):
    nsamples, timeStep, rows, cols, ch = data.shape
    feature = np.reshape(data, (nsamples, timeStep * ch * rows * cols))
    feature_denormalization = scaler.inverse_transform(feature)
    return np.reshape(feature_denormalization, (nsamples, timeStep, rows, cols, ch))

def inverse_normalization(args):
    path_to_normalizers, y_pred_uncertainty, normalizers = args
    print(path_to_normalizers + normalizers + '.pkl')
    scaler = joblib.load(path_to_normalizers + normalizers + '.pkl')
    x_test_denormalization = denormalization_computing(scaler, y_pred_uncertainty, normalizers)
    return x_test_denormalization

def denormalize(path_to_normalizers, y_pred):
    list_normalizers = ['POR3D', 'RTP3D', 'NTG3D', 'PERMI3D']
    results = []
    for y_pred_uncertainty, normalizers in zip(y_pred, list_normalizers):
        result = inverse_normalization((path_to_normalizers, y_pred_uncertainty, normalizers))
        results.append(result)
    return results