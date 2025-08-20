import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib.colors import LinearSegmentedColormap

def create_custom_cmap():
    colors = [
        (0.027, 0.165, 0.992),  # #072afd
        (0, 0.773, 1),          # #00c5ff
        (0.086, 0.988, 0.545),  # #15fc8b
        (0.525, 0.988, 0.243),  # #86fd3e
        (0.871, 1, 0.125),      # #deff20
        (1, 0.855, 0.008),      # #ffda02
        (0.996, 0.466, 0.004),  # #fe7701
        (1, 0.055, 0.004)       # #ff0e01
    ]
    return LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

# def plot_uncertainty_layer(uncertainty, name, custom_cmap):    
#     uncertainty = np.fliplr(uncertainty)
    
#     uncertainty[uncertainty <= 0.00000001] = np.nan
    
#     vmin = min(np.nanmin(uncertainty), np.nanmin(uncertainty))
#     vmax = max(np.nanmax(uncertainty), np.nanmax(uncertainty))
    
#     fig, ax = plt.subplots(figsize=(5, 5))

#     im = ax.imshow(uncertainty, cmap=custom_cmap, vmin=vmin, vmax=vmax)
#     ax.set_title(name)
#     cbar = plt.colorbar(im, ax=ax, cax=ax.inset_axes([1.05, 0, 0.05, 1]), shrink=0.8)
    
#     plt.tight_layout()
#     plt.show()
    
# def plot_uncertainty(uncertainty, name):
#     custom_cmap = create_custom_cmap()
#     for i in range(32):
#         print(i)
#         plot_uncertainty_layer(
#             uncertainty[i], 
#             name + str(i + 1), 
#             custom_cmap
#         )

# def plot_image(ori, text, reconstruct, text2, custom_cmap):
#     ori_std = np.std(ori)
#     reconst_std = np.std(reconstruct)

#     ori = np.fliplr(ori)
#     reconstruct = np.fliplr(reconstruct)

#     slope, intercept, r_value, p_value, std_err = linregress(ori.flatten(), reconstruct.flatten())

#     ori[ori <= 0.4] = np.nan
#     reconstruct[reconstruct <= 0.4] = np.nan

#     vmin = min(np.nanmin(ori), np.nanmin(reconstruct))
#     vmax = max(np.nanmax(ori), np.nanmax(reconstruct))

#     fig, axs = plt.subplots(1, 2, figsize=(10, 5))

#     im1 = axs[0].imshow(ori, cmap=custom_cmap, vmin=vmin, vmax=vmax)
#     axs[0].set_title(text)
#     cbar1 = plt.colorbar(im1, ax=axs[0], cax=axs[0].inset_axes([1.05, 0, 0.05, 1]), shrink=0.8)

#     im2 = axs[1].imshow(reconstruct, cmap=custom_cmap, vmin=vmin, vmax=vmax)
#     axs[1].set_title(text2)
#     cbar2 = plt.colorbar(im2, ax=axs[1], cax=axs[1].inset_axes([1.05, 0, 0.05, 1]), shrink=0.8)

#     plt.tight_layout()
#     plt.show()
#     return ori_std, reconst_std

# def plot_histograms(original, reconstruction, text, text2):
#     fig, axs = plt.subplots(1, 2, figsize=(10, 5))

#     axs[0].hist(original.flatten(), bins=10)
#     axs[0].set_title(text, fontsize=18)

#     axs[1].hist(reconstruction.flatten(), bins=10)
#     axs[1].set_title(text2, fontsize=18)

#     plt.tight_layout()
#     plt.show()

# def plot_kde(original, reconstruction, text, text2, name_uncertainty):
#     plt.figure(figsize=(10, 6))
#     sns.kdeplot(original.flatten(), color='r', shade=True, label=text)
#     sns.kdeplot(reconstruction.flatten(), color='b', shade=True, label=text2)
#     plt.xlabel('Values', fontsize=18) 
#     plt.ylabel('Frequency', fontsize=18) 
#     plt.title(name_uncertainty, fontsize=20)
#     plt.legend(loc='best', fontsize=14)
#     plt.xticks(fontsize=20)
#     plt.yticks(fontsize=20)
#     plt.show()
    
# def plot_ridgeline(original, reconstruction, original_std, reconstruction_std, layer=None, type_uncertainty=None):
#     flattened_ori = original.flatten()
#     flattened_reconstruct = reconstruction.flatten()
#     data = {
#         'Values': np.concatenate([flattened_ori, flattened_reconstruct]),
#         'Type': np.concatenate([
#             np.repeat('Original', flattened_ori.shape[0]),
#             np.repeat('Reconstructed', flattened_reconstruct.shape[0])
#         ])
#     }
#     df = pd.DataFrame(data)
#     hue_order = ['Original', 'Reconstructed']
#     plt.figure(figsize=(10, 5))
#     plot_title = f'{type_uncertainty} Plot of Original and Reconstructed'
#     if layer:
#         plot_title += f' in {layer}'
    
#     sns.histplot(
#         df, x='Values', hue='Type', hue_order=hue_order,
#         element='step', stat='frequency', common_norm=False, fill=True
#     )
    
#     plt.title(plot_title, fontsize=18)
#     plt.xlabel('Values', fontsize=18)
#     plt.ylabel('Frequency', fontsize=18)
#     plt.xticks(fontsize=18)
#     plt.yticks(fontsize=18)
#     plt.legend(labels=['Original', 'Reconstructed'], loc='best', fontsize=15)
    
#     plt.show()
    
# def plot_global_histogram(original, reconstruction, name_uncertainty=None):
#     # Compute standard deviations
#     ori_std = np.std(original)
#     #reconst_std = np.std(reconstruction)
#     std_reconst = np.std(reconstruction)
    
#     # Set values lower than 0.4 to NaN
#     original[original <= 0.4] = np.nan
#     reconstruction[reconstruction <= 0.4] = np.nan

#     # Plot ridgeline plot
#     plot_ridgeline(
#         original, 
#         reconstruction, 
#         str(round(ori_std, 3)), 
#         str(round(std_reconst, 3)), 
#         type_uncertainty=name_uncertainty)
    
#     plot_histograms(
#             original, 
#             reconstruction, 
#             'Original', 
#             'Reconstructed'
#     )
    
#     plot_kde(
#             original, 
#             reconstruction, 
#             'Original', 
#             'Reconstructed', 
#             name_uncertainty + ' Plot Original and Reconstruction'
#         )

# def plot_original_reconstruction(original, text, reconstruction, text2, name_uncertainty=None):
#     custom_cmap = create_custom_cmap()
#     for i in range(11,15):
#         print(i)
#         ori_std, reconst_std = plot_image(
#             original[i], 
#             text + str(i + 1), 
#             reconstruction[i], 
#             text2 + str(i + 1), 
#             custom_cmap
#         )
#         plot_histograms(
#             original[i], 
#             reconstruction[i], 
#             text + str(i + 1), 
#             text2 + str(i + 1)
#         )
#         plot_kde(
#             original[i], 
#             reconstruction[i], 
#             text + str(i + 1), 
#             text2 + str(i + 1), 
#             name_uncertainty + ' Plot Original and Reconstruction in layer ' + str(i + 1)
#         )
#         plot_ridgeline(
#             original[i], 
#             reconstruction[i], 
#             str(round(ori_std, 3)), 
#             str(round(reconst_std, 3)), 
#             layer='layer '+str(i + 1), 
#             type_uncertainty=name_uncertainty
#         )
        
# def visualize_original_reconstruction(x_true, x_predict, uncertainty_name):
#     for original, reconstruction in zip(x_true, x_predict):
#         text, text2 = uncertainty_name + ' Original in Layer ', uncertainty_name +' Reconstruction in Layer '
#         plot_global_histogram(original, reconstruction, name_uncertainty=uncertainty_name)
#         plot_original_reconstruction(original, text, reconstruction, text2, uncertainty_name)    

import os
# import matplotlib.pyplot as plt
# import numpy as np

def save_plot(fig, job_number, plot_name):
    # Create the directory if it doesn't exist
    save_path = f"Save/{job_number}"
    os.makedirs(save_path, exist_ok=True)

    # Save the plot in the specified folder
    file_path = os.path.join(save_path, plot_name)
    fig.savefig(file_path)
    # print(f"Saved plot to {file_path}")

def plot_uncertainty_layer(uncertainty, name, custom_cmap, job_number):    
    uncertainty = np.fliplr(uncertainty)
    
    uncertainty[uncertainty <= 0.00000001] = np.nan
    
    vmin = min(np.nanmin(uncertainty), np.nanmin(uncertainty))
    vmax = max(np.nanmax(uncertainty), np.nanmax(uncertainty))
    
    fig, ax = plt.subplots(figsize=(5, 5))

    im = ax.imshow(uncertainty, cmap=custom_cmap, vmin=vmin, vmax=vmax)
    ax.set_title(name)
    cbar = plt.colorbar(im, ax=ax, cax=ax.inset_axes([1.05, 0, 0.05, 1]), shrink=0.8)
    
    plt.tight_layout()

    # Save the plot
    save_plot(fig, job_number, f"uncertainty_{name}.png")
    
    plt.show()
    
def plot_uncertainty(uncertainty, name, job_number):
    custom_cmap = create_custom_cmap()
    for i in range(32):
        # print(i)
        plot_uncertainty_layer(
            uncertainty[i], 
            name + str(i + 1), 
            custom_cmap, 
            job_number
        )

def plot_image(ori, text, reconstruct, text2, custom_cmap, job_number):
    ori_std = np.std(ori)
    reconst_std = np.std(reconstruct)

    ori = np.fliplr(ori)
    reconstruct = np.fliplr(reconstruct)

    slope, intercept, r_value, p_value, std_err = linregress(ori.flatten(), reconstruct.flatten())

    ori[ori <= 0.4] = np.nan
    reconstruct[reconstruct <= 0.4] = np.nan

    vmin = min(np.nanmin(ori), np.nanmin(reconstruct))
    vmax = max(np.nanmax(ori), np.nanmax(reconstruct))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    im1 = axs[0].imshow(ori, cmap=custom_cmap, vmin=vmin, vmax=vmax)
    axs[0].set_title(text)
    cbar1 = plt.colorbar(im1, ax=axs[0], cax=axs[0].inset_axes([1.05, 0, 0.05, 1]), shrink=0.8)

    im2 = axs[1].imshow(reconstruct, cmap=custom_cmap, vmin=vmin, vmax=vmax)
    axs[1].set_title(text2)
    cbar2 = plt.colorbar(im2, ax=axs[1], cax=axs[1].inset_axes([1.05, 0, 0.05, 1]), shrink=0.8)

    plt.tight_layout()

    # Save the plot
    save_plot(fig, job_number, f"comparison_{text}_{text2}.png")
    
    plt.show()
    return ori_std, reconst_std

def plot_histograms(original, reconstruction, text, text2, job_number):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].hist(original.flatten(), bins=10)
    axs[0].set_title(text, fontsize=18)

    axs[1].hist(reconstruction.flatten(), bins=10)
    axs[1].set_title(text2, fontsize=18)

    plt.tight_layout()

    # Save the plot
    save_plot(fig, job_number, f"histogram_{text}_{text2}.png")
    
    plt.show()

def plot_kde(original, reconstruction, text, text2, name_uncertainty, job_number):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(original.flatten(), color='r', shade=True, label=text)
    sns.kdeplot(reconstruction.flatten(), color='b', shade=True, label=text2)
    plt.xlabel('Values', fontsize=18) 
    plt.ylabel('Frequency', fontsize=18) 
    plt.title(name_uncertainty, fontsize=20)
    plt.legend(loc='best', fontsize=14)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Save the plot
    save_plot(plt, job_number, f"kde_{text}_{text2}.png")

    plt.show()
    
def plot_ridgeline(original, reconstruction, original_std, reconstruction_std, layer=None, type_uncertainty=None, job_number=None):
    flattened_ori = original.flatten()
    flattened_reconstruct = reconstruction.flatten()
    data = {
        'Values': np.concatenate([flattened_ori, flattened_reconstruct]),
        'Type': np.concatenate([
            np.repeat('Original', flattened_ori.shape[0]),
            np.repeat('Reconstructed', flattened_reconstruct.shape[0])
        ])
    }
    df = pd.DataFrame(data)
    hue_order = ['Original', 'Reconstructed']
    plt.figure(figsize=(10, 5))
    plot_title = f'{type_uncertainty} Plot of Original and Reconstructed'
    if layer:
        plot_title += f' in {layer}'
    
    sns.histplot(
        df, x='Values', hue='Type', hue_order=hue_order,
        element='step', stat='frequency', common_norm=False, fill=True
    )
    
    plt.title(plot_title, fontsize=18)
    plt.xlabel('Values', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    # Save the plot
    save_plot(plt, job_number, f"ridgeline_{plot_title}.png")
    
    plt.show()

def plot_global_histogram(original, reconstruction, name_uncertainty=None, job_number=None):
    ori_std = np.std(original)
    std_reconst = np.std(reconstruction)
    
    original[original <= 0.4] = np.nan
    reconstruction[reconstruction <= 0.4] = np.nan

    # Plot ridgeline plot
    plot_ridgeline(
        original, 
        reconstruction, 
        str(round(ori_std, 3)), 
        str(round(std_reconst, 3)), 
        type_uncertainty=name_uncertainty,
        job_number=job_number
    )
    
    plot_histograms(
            original, 
            reconstruction, 
            'Original', 
            'Reconstructed',
            job_number
    )
    
    plot_kde(
            original, 
            reconstruction, 
            'Original', 
            'Reconstructed', 
            name_uncertainty + ' Plot Original and Reconstruction',
            job_number
        )

def plot_original_reconstruction(original, text, reconstruction, text2, name_uncertainty=None, job_number=None):
    custom_cmap = create_custom_cmap()
    for i in range(11,15):
        # print(i)
        ori_std, reconst_std = plot_image(
            original[i], 
            text + str(i + 1), 
            reconstruction[i], 
            text2 + str(i + 1), 
            custom_cmap,
            job_number
        )
        plot_histograms(
            original[i], 
            reconstruction[i], 
            text + str(i + 1), 
            text2 + str(i + 1),
            job_number
        )
        plot_kde(
            original[i], 
            reconstruction[i], 
            text + str(i + 1), 
            text2 + str(i + 1), 
            name_uncertainty + ' Plot Original and Reconstruction in layer ' + str(i + 1),
            job_number
        )
        plot_ridgeline(
            original[i], 
            reconstruction[i], 
            str(round(ori_std, 3)), 
            str(round(reconst_std, 3)), 
            layer='layer '+str(i + 1), 
            type_uncertainty=name_uncertainty,
            job_number=job_number
        )
        
def visualize_original_reconstruction(x_true, x_predict, uncertainty_name, job_number):
    for original, reconstruction in zip(x_true, x_predict):
        text, text2 = uncertainty_name + ' Original in Layer ', uncertainty_name +' Reconstruction in Layer '
        plot_global_histogram(original, reconstruction, name_uncertainty=uncertainty_name, job_number=job_number)
        plot_original_reconstruction(original, text, reconstruction, text2, uncertainty_name, job_number=job_number)
