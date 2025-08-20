##################
#python libraries#
##################
import os
import time
import copy
import keras
import random   
import pickle
import argparse
import numpy as np
import tensorflow as tf
import plotly.express as px
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.metrics import r2_score
from tensorflow.keras import Input, Model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.callbacks import Callback, EarlyStopping

##################
#  my libraries  #
##################
# from src.models.ResidualVAE import ResidualVAE
# from src.models.ResidualVAE_tf_v2 import ResidualVAEtf
from src.notebook_util import setup_gpu
from src.metrics import smape, mse_funct
from src.denormalization import denormalize
# from src.models.drvae_RAF_anchor_tf_v2 import ResidualVAEtfv2
from src.models.drvae import DRVAE
from src.data_loader import loadUncertainties, loadProductionANDHistory, selectProductionCurve
from src.visualization import visualizationResults, plot_learning_curves, computeAndSaveSmapeAndNQDS
from src.original_reconstruction_visualizer import visualize_original_reconstruction, plot_original_reconstruction, plot_uncertainty
from utils import loadNpFile, load_pkl_file, clear_and_save, loadTXTFile, convertFoldertoZipFile, load_pkl_file, createPKLFile

def configure_callbacks():
    """
    Configura e retorna uma lista de callbacks para o treinamento.
    """
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto')
    return [earlystop]

def reshape_uncertainties(uncertainties):
    """Redimensiona incertezas de fluidos para adicionar uma dimensão."""
    return uncertainties.reshape(uncertainties.shape[0], uncertainties.shape[1], 1)

def random_initialization(seed=2):
    # 1. Set the seed for NumPy and TensorFlow
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # 2. Set TensorFlow to use deterministic operations
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
    # 3. Control TensorFlow settings for deterministic behavior
    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    
    # # Seed for reproducibility
    npr.seed(10)
    
def train(x_train, x_val, x_test, y_train, y_val, y_test):
    n_outputs = y_train.shape[1]
    input_shapes = [(x.shape[1], x.shape[2], x.shape[3], x.shape[4]) for x in x_train[:4]] + [(x_train[4].shape[1], 1)]
    
    n_epochs = 250
    n_ensembles = 2
    batch_size = 8
    y_pred = []
    y_pred_val = []
    for i in range(n_ensembles):
        model = DRVAE(*input_shapes, n_outputs).model
        history = model.fit(x=x_train, y=x_train[:-1],
                            batch_size=batch_size,
                            epochs=n_epochs,
                            shuffle=True,
                            callbacks=configure_callbacks(),
                            verbose=2, 
                            validation_data=(x_val, x_val[:-1]))
        metrics = model.evaluate(x_test, x_test[:-1], verbose=1)#, batch_size=1)
        print(metrics)
        loss = metrics[0]
        print('loss: %.3f' % loss)

        y_pred.append(model.predict(x = x_test))
        y_pred_val.append(model.predict(x = x_val))
    print()
    print('test set')
    reconstruct_metrics([], x_test[:-1], y_pred, job_number=job_number)
    print()
    print('validation set')
    reconstruct_metrics([], x_val[:-1], y_pred_val, job_number=job_number)

    return loss, loss, model

def preprocess_targets(targets):
    """
    Preprocessa os targets removendo e combinando colunas específicas.
    """
    processed_targets = targets[:28] + targets[56:63]
    processed_targets = np.delete(processed_targets, np.s_[14:21])
    return processed_targets

def load_or_process_targets(path, targets):
    """
    Carrega os targets de arquivos se um caminho for fornecido, caso contrário, processa-os.
    """
    if path:
        return np.concatenate([loadNpFile(path, target) for target in targets], axis=1)
    
    return targets

def normalize_targets(y_train, y_test, y_val, path_save_normalizer, normalize=False):
    """
    Normaliza os targets se necessário.
    """
    if normalize:
        scaler = MinMaxScaler()
        y_train =  scaler.fit_transform(y_train)
        y_test =  scaler.transform(y_test)
        y_val =  scaler.transform(y_val)
        print('asdasda')
    createPKLFile(path_save_normalizer, scaler)     
    return y_train, y_test, y_val, scaler

def train_model(x_train_por_3d, x_train_rtp_3d, x_train_ntg_3d, 
                x_train_permi_3d, x_train_fluid_uncertainties, x_test_por_3d, 
                x_test_rtp_3d, x_test_ntg_3d, x_test_permi_3d, 
                x_test_fluid_uncertainties, x_val_por_3d, x_val_rtp_3d, 
                x_val_ntg_3d, x_val_permi_3d, x_val_fluid_uncertainties, 
                y_train, y_test, y_val, path_model_to_save, 
                path_to_production=None, normalized_output=True):
                    
    # Preprocessamento dos targets
    y_train = preprocess_targets(y_train)
    y_test = preprocess_targets(y_test)
    y_val = preprocess_targets(y_val)

    # Carregamento ou processamento dos targets
    y_train = load_or_process_targets(path_to_production, y_train)
    y_test = load_or_process_targets(path_to_production, y_test)
    y_val = load_or_process_targets(path_to_production, y_val)
    
    # Normalização dos targets, se necessário
    path_save_normalizer = path_model_to_save + 'normalizer_target/normalizer'
    y_train, y_test, y_val, _ = normalize_targets(y_train, y_test, y_val, path_save_normalizer, normalized_output) 
    
    # Redimensionamento para conjuntos de treino, validação e teste
    x_train = [x_train_por_3d, x_train_rtp_3d, x_train_ntg_3d, x_train_permi_3d]
    x_val = [x_val_por_3d, x_val_rtp_3d, x_val_ntg_3d, x_val_permi_3d]
    x_test = [x_test_por_3d, x_test_rtp_3d, x_test_ntg_3d, x_test_permi_3d]

    x_train_fluid_uncertainties = reshape_uncertainties(x_train_fluid_uncertainties)
    x_val_fluid_uncertainties = reshape_uncertainties(x_val_fluid_uncertainties)
    x_test_fluid_uncertainties = reshape_uncertainties(x_test_fluid_uncertainties)
    
    x_train.append(x_train_fluid_uncertainties)
    x_val.append(x_val_fluid_uncertainties)
    x_test.append(x_test_fluid_uncertainties)
    
    # get the model and metrics results
    loss_error, loss_general, model = train(x_train, x_val, x_test, y_train, y_val, y_test)

    print(loss_error, loss_general)

def resize_to_original_dimension(uncertainty):
    return uncertainty[:, :, 5:-5, :]

def resize_and_normalization(uncertainty):
    return resize_to_original_dimension(uncertainty)

def ssim_check(y_true, y_pred):
    # Remove the last singleton dimension
    y_true = y_true.squeeze(-1)  # shape: (60, 24, 38, 72)
    y_pred = y_pred.squeeze(-1)

    ssim_scores = []

    for i in range(y_true.shape[0]):  # over samples
        for c in range(y_true.shape[1]):  # over channels
            true_img = y_true[i, c]  # shape: (38, 72)
            pred_img = y_pred[i, c]
            
            # Compute SSIM for 2D images
            score = ssim(true_img, pred_img, data_range=true_img.max() - true_img.min())
            ssim_scores.append(score)

    return np.mean(ssim_scores)
    # print("Average SSIM over all samples and channels:", mean_ssim)

def psnr(y_true, y_pred, max_value=1.0):
    # Peak Signal-to-Noise Ratio (PSNR) 
    # Ensure that the input images are in the range [0, 1]
    # Compute Mean Squared Error (MSE)
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Handle the case where MSE is zero (no difference between the images)
    if mse == 0:
        return np.inf  # If there is no error, PSNR is infinity
    
    # Compute PSNR
    psnr_value = 10 * np.log10(max_value ** 2 / mse)
    return psnr_value

def reconstruction_metrics(original_data, reconstructed_data):
    ssim_values = ssim_check(original_data, reconstructed_data)
    print(original_data.shape, reconstructed_data.shape)
    fid_values = calculate_fid(original_data, reconstructed_data)
    original_flattened = np.reshape(original_data, -1)
    reconstructed_flattened = np.reshape(reconstructed_data, -1)
    _, _, r, _, _ = linregress(original_flattened, reconstructed_flattened)
    mse_values = mse_funct(original_data, reconstructed_data)
    psnr_value = psnr(original_data, reconstructed_data)

    print('psnr:', round(psnr_value, 3), '\nfid:', round(fid_values, 3), '\ncorrelation:', round(r, 3),'\nssim:', round(ssim_values, 3),'\nmse:',  round(mse_values, 3))

# Convert 3D volumes to 2D slices along the depth axis
def volume_to_slices(volume_batch):
    """
    volume_batch: shape (N, D, H, W, C)
    returns: shape (N * D, H, W, C)
    """
    batch_size, depth, height, width, channels = volume_batch.shape
    return tf.reshape(volume_batch, [batch_size * depth, height, width, channels])

# Preprocessing for InceptionV3 (expects 299x299 RGB)
def preprocess_for_inception(images):
    """
    images: shape (N, H, W, 1)
    returns: shape (N, 299, 299, 3)
    """
    images = tf.image.resize(images, (299, 299))       # Resize to InceptionV3 input
    images = tf.image.grayscale_to_rgb(images)          # Convert to 3 channels
    return preprocess_input(images)


def calculate_fid(real_volumes, generated_volumes):
    """
    real_volumes, generated_volumes: shape (N, D, H, W, 1), dtype=float32, range [0, 1]
    """
    """
    real_volumes, generated_volumes: shape (N, D, H, W, 1), dtype=float32, range [0, 1]
    """
    # Build TensorFlow graph
    real_images_tensor = preprocess_for_inception(volume_to_slices(real_volumes))
    generated_images_tensor = preprocess_for_inception(volume_to_slices(generated_volumes))

    # Evaluate tensors to get NumPy arrays
    with tf.compat.v1.Session() as sess:
        real_images_np, generated_images_np = sess.run([real_images_tensor, generated_images_tensor])

    # Load InceptionV3 model
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    # Extract features from real and generated images
    real_features = model.predict(real_images_np, batch_size=32)
    gen_features = model.predict(generated_images_np, batch_size=32)

    # Compute mean and covariance statistics
    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(gen_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(gen_features, rowvar=False)

    # Compute Fréchet Distance
    covmean, _ = sqrtm(sigma_real @ sigma_gen, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = np.sum((mu_real - mu_gen) ** 2) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid

def recover_original_arrays(reshaped_array):
    reshaped_back = reshaped_array.reshape((240, 24, 48, 72, 1))
    array_list_recovered = np.split(reshaped_back, 4)
    return array_list_recovered

def process_reconstruction(x_mean_reconstruction, x_uncertainty, uncertainty_name, job_number):
    x_mean = resize_and_normalization(x_mean_reconstruction)
    x_uncertainty_norm = resize_and_normalization(x_uncertainty)
    reconstruction_metrics(x_mean, x_uncertainty_norm)
    visualize_original_reconstruction(x_mean, x_uncertainty_norm, uncertainty_name, job_number)

def reconstruct_metrics(method_means, y_true, y_pred, check_model=False, job_number=None):
    list_uncertainties = ['por_3d', 'rtp_3d', 'ntg_3d', 'permi_3d']

    print(method_means)
    if len(method_means) > 0:
        list_of_means = recover_original_arrays(method_means)
        print('======= mean results =======')
        for x_mean_reconstruction, x_uncertainty, uncertainty in zip(list_of_means, y_true, list_uncertainties):
            print(uncertainty)
            process_reconstruction(x_mean_reconstruction, x_uncertainty, uncertainty, job_number)

    for n_ensamble, y_ensemble in enumerate(y_pred):
        print(f'Ensamble: {n_ensamble}', len(y_ensemble), len(y_true))
        for x_mean_reconstruction, x_uncertainty, uncertainty in zip(y_ensemble, y_true, list_uncertainties):
            print('======= '+ uncertainty + ' =======')
            process_reconstruction(x_mean_reconstruction, x_uncertainty, uncertainty, job_number)
            
def setup_gpu_environment(gpu_id):
    """Configura o ambiente de GPU."""
    setup_gpu(gpu_id)

def load_data(uncertainties_path, production_path, history_path, is_some_days):
    """
    Carrega os datasets de incertezas, produção e histórico, considerando a configuração de dias específicos.
    
    Args:
        uncertainties_path (str): Caminho para os dados de incertezas.
        production_path (str): Caminho base para os dados de produção.
        history_path (str): Caminho base para os dados históricos.
        is_some_days (str): Sufixo para os caminhos, indicando se são dados de alguns dias específicos.
    
    Returns:
        Tuple: Datasets de treinamento, teste, validação e histórico carregados.
    """
    production_path_with_days = f"{production_path}{is_some_days}/"
    history_path_with_days = f"{history_path}{is_some_days}/"

    # Carrega incertezas
    x_train_porosity_3d, x_train_rtp_3d, x_train_ntg_3d, x_train_permi_3d, x_train_fluid_uncertainties, \
    x_test_porosity_3d, x_test_rtp_3d, x_test_ntg_3d, x_test_permi_3d, x_test_fluid_uncertainties, \
    x_val_porosity_3d, x_val_rtp_3d, x_val_ntg_3d, x_val_permi_3d, x_val_fluid_uncertainties = loadUncertainties(uncertainties_path)

    # Carrega dados de produção
    train_output = loadProductionANDHistory(production_path_with_days, String="Production", typeOfSet="train")
    test_output = loadProductionANDHistory(production_path_with_days, String="Production", typeOfSet="test")
    val_output = loadProductionANDHistory(production_path_with_days, String="Production", typeOfSet="validation")
    
    # Carrega dados históricos
    history_data = loadProductionANDHistory(history_path_with_days, String="History")



def pretrained(uncertainties_path, production_path, 
               history_path, days_txt, gpu_id, is_some_days, 
               path_model_to_save):
    """
    Treina o modelo de ponta a ponta com dados de incertezas, produção e histórico.
    
    Args:
        uncertainties_path (str): Caminho para os dados de incertezas.
        upload_production_path (str), save_history_path (str): Caminhos para upload de produção e salvamento de histórico.
        some_days_txt (str), gpu (int), is_some_days (str), results_path (str), start_to_train (float): Configurações diversas.
        opt_analysis (bool): Se a análise de otimização deve ser realizada.
    
    Returns:
        Tuple: Resultados do modelo treinado incluindo problema, y_test_sobol, x_test_encoder, new_model.
    """
    setup_gpu_environment(gpu_id)
    
    x_train, x_test, x_val, train_output, test_output, val_output, _ = load_data(
        uncertainties_path, production_path, history_path, is_some_days
    )
    
    train_model(
        *x_train, *x_test, *x_val,
        train_output, test_output, val_output, path_model_to_save,
        path_to_production=f"{production_path}{is_some_days}/"
    )


def parse_arguments():
    """
    Parse command-line arguments for the script.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run simulations and predictions.")

    # SIMULATOR
    parser.add_argument('uncertainties_file', 
                        type=str, help="File path to the uncertainties data")
    parser.add_argument('simulation_file', 
                        type=str, help="File path to the simulation data")
    
    # PRODUCTION
    parser.add_argument('history_file', 
                        type=str, help="File path to the production history")
    
    # SELECTED DAYS
    parser.add_argument('selected_days_file', 
                        type=str, help="File path to the selected days (txt format)")
    
    # NORMALIZATION
    parser.add_argument('gpu_id', 
                        type=str, help="GPU ID to be used for computations")
    
    # CHOOSE ALL OR SOME DAYS
    parser.add_argument('days_option', 
                        type=str, choices=['someDays', 'allDays'], 
                        help="Choose whether to use 'someDays' or 'allDays' of production data")
    
    # VISUALIZATION
    parser.add_argument('results_save_path', 
                        type=str, help="Directory to save the result outputs")

    return parser.parse_args()

def main():
    # Parse the command-line arguments
    args = parse_arguments()
    # Assign arguments to descriptive variable names
    uncertainties_file = args.uncertainties_file
    simulation_file = args.simulation_file
    history_file = args.history_file
    selected_days_file = args.selected_days_file
    gpu_id = args.gpu_id
    days_option = args.days_option
    results_save_path = args.results_save_path
    random_initialization()

    # Perform the pretrained prediction
    pretrained(
        uncertainties_path=uncertainties_file,
        production_path=simulation_file,
        history_path=history_file,
        days_txt=selected_days_file,
        gpu_id=gpu_id,
        is_some_days=days_option,
        path_model_to_save=results_save_path
    )


if __name__ == "__main__":
    main()
