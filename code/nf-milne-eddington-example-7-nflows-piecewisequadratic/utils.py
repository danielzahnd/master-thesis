"""
This file contains several functions used in jupyter notebooks concerned with normalizing flows.
"""
# Import all important packages and set parameters
import emcee
import os
from tqdm import tqdm
from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm, amsmath, siunitx}'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = 14
from scipy.interpolate import interp1d
from astropy.io import fits
from scipy.interpolate import CubicSpline
import torch
import torch.nn as nn
import itertools
import numpy as np
import pandas as pd
from math import pi 
import seaborn as sns
import corner
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import LogisticNormal
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.nn import functional as F
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.coupling import PiecewiseQuadraticCouplingTransform
from nflows.transforms.permutations import ReversePermutation
from nflows import distributions, flows, transforms, utils
from nflows.nn import nets
from sklearn.model_selection import train_test_split
from matplotlib.patches import Rectangle

# Save current directory
current_directory = os.getcwd()

# Change working path to pyMilne directory and load Milne-Eddington code
os.chdir('/home/dz/pyMilne') # Reference to pyMilne directory
import MilneEddington

# Set path again to current directory
os.chdir(current_directory)

# Define function to read and interpolate penumbra formation maps
def read_map(filename):
    '''
    Reads a map of dimension (540,700,4,13).
    '''
    # Store observation name
    obs_name = filename.split('/')[-1][:27]

    # Read the image data
    map_inv = fits.getdata(filename, ext=0)
    hdulist = fits.open(filename)

    # Interpolate
    dtype = 'float32'
    regions = [[np.arange(13, dtype=dtype)*0.04 - 0.28 + 6302.4931, None]]
    regions_interp = regions[0][0][np.array(sorted(list(set(np.where(map_inv!=-1)[1]))))]
    map_inv = map_inv[np.where(map_inv!=-1)]
    map_inv = map_inv.reshape(4,12,540,700)
    cs_map = CubicSpline(regions_interp, map_inv, axis=1)
    map_inv_interp = cs_map(regions[0][0])
    map_inv = map_inv_interp.transpose(2,3,0,1)
    return map_inv, hdulist, regions

# Define function to read and interpolate penumbra formation maps
def read_map_pf(filename):
    '''
    Reads a map of dimension (550,600,4,13).
    '''
    # Store observation name
    obs_name = filename.split('/')[-1][:27]

    # Read the image data
    map_inv = fits.getdata(filename, ext=0)
    hdulist = fits.open(filename)

    # Interpolate
    dtype = 'float32'
    regions = [[np.arange(13, dtype=dtype)*0.04 - 0.28 + 6302.4931, None]]
    regions_interp = regions[0][0][np.array(sorted(list(set(np.where(map_inv!=-1)[1]))))]
    map_inv = map_inv[np.where(map_inv!=-1)]
    map_inv = map_inv.reshape(4,11,550,600)
    cs_map = CubicSpline(regions_interp, map_inv, axis=1)
    map_inv_interp = cs_map(regions[0][0])
    map_inv = map_inv_interp.transpose(2,3,0,1)
    return map_inv, hdulist, regions

# Function to read maps and store their data in a dictionary
def read_maps(frames):
    """
    Reads map data into a dictionary.
    """
    map_inv_dict = {}
    hdulist_dict = {}
    regions_dict = {}

    # Change base path if needed
    base_path = '/home/dz/maps/Complete dataset penumbra formation maps/Prepared_map_SIR_AR13010_nb_6302_2022-05-16T08_28_21_frame_{}.fits'
    
    for frame_number in frames:
        file_path = base_path.format(frame_number)
        map_inv_frame, hdulist_frame, regions_frame = read_map_pf(file_path)
        map_inv_dict[frame_number] = map_inv_frame
        hdulist_dict[frame_number] = hdulist_frame
        regions_dict[frame_number] = regions_frame
    
    return map_inv_dict, hdulist_dict, regions_dict

# Define function to invert maps
def invert_milne(map, regions):
    '''
    Inverts a map of dimension (n,m,4,13) using the Milne-Eddington algorithm with n and m integer numbers.
    '''
    # Initialize the Milne-Eddington object
    lines   = [6302]
    me = MilneEddington.MilneEddington(regions, lines, nthreads=2, precision='float32')

    # Set number of pixels on map
    ny = map.shape[0]
    nx = map.shape[1]

    # Set noise level
    noise_level = 5.e-3
    sig = np.zeros((4, me.get_wavelength_array().size), dtype='float64', order='c')
    sig += noise_level

    # Artificially increase the weight of Q, U and V by lowering their noise estimate
    sig[1:3] /= 7.
    sig[3]   /= 3.
    
    # Provide model with initial guess for the parameter values [|B| [G], theta [rad], varphi [rad], v_los [km/s], Delta lambda_D [Angstrom], eta_0, a, S0, S1]
    iGuessed = np.float64([500., 0.1, 0.1, 0.0, 0.04, 100, 0.5, 0.1, 1.0])
    guessed_initial  = me.repeat_model(iGuessed, ny, nx)

    # Invert map
    inverted_map, syn_out, chi2 = me.invert(guessed_initial, map, sig, nRandom = 5, nIter=20, chi2_thres=1.0, verbose=False)

    # Estimate uncertainties for the resulting parameters of the map inversion
    errors = me.estimate_uncertainties(inverted_map, map, sig, mu=1.0)

    # Return inverted map with errors and synthetic spectra
    return inverted_map, errors, syn_out

# Function to invert and store map data
def invert_store_and_save(frames, map_dict, regions_dict):
    """
    frames: List of frames to invert and store.
    map_dict: Dictionary with data.
    regions_dict: Dictionary with wavelength point data.
    """
    # Define dictionaries
    inverted_map_dict = {}
    errors_map_dict = {}
    syn_spectra_map_dict = {}

    # Loop over all frames
    for i in frames:
        inverted_map_save = 'inverted_map' + str(i) + '.npy'
        syn_spectra_map_save = 'syn_spectra_map' + str(i) + '.npy'
        errors_map_save = 'errors_map' + str(i) + '.npy'
        inverted_map, errors_map, syn_spectra_map = invert_milne(map_dict[i], regions_dict[i])
        inverted_map_dict[i] = inverted_map
        errors_map_dict[i] = errors_map
        syn_spectra_map_dict[i] = syn_spectra_map
        np.save(inverted_map_save, inverted_map)
        np.save(errors_map_save, errors_map)
        np.save(syn_spectra_map_save, syn_spectra_map)
    print('Data sucessfully inverted, stored and saved!')
    return inverted_map_dict, errors_map_dict, syn_spectra_map_dict

# Define function to load already inverted data
def load_inv_maps(frames):
    """
    Load previously inverted maps.
    
    frames: List of frames, for which inverted data is to be imported.
    """
    # Define dictionaries
    inverted_map_dict = {}
    errors_map_dict = {}
    syn_spectra_map_dict = {}

    # Loop over all frames
    for i in frames:
        inverted_map_load = 'inverted_map' + str(i) + '.npy'
        syn_spectra_map_load = 'syn_spectra_map' + str(i) + '.npy'
        errors_map_load = 'errors_map' + str(i) + '.npy'
        inverted_map_dict[i] = np.load(inverted_map_load)
        errors_map_dict[i] = np.load(errors_map_load)
        syn_spectra_map_dict[i] = np.load(syn_spectra_map_load)
    
    return inverted_map_dict, errors_map_dict, syn_spectra_map_dict

# Function to plot stokes I, Q, U and V
def plot_stokes(spectra_true, spectra_syn, wavelengths, pixel_x, pixel_y, labels, savename=None, title=None):
    """
    Plot Stokes vector.
    """
    plt
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))  # Create a 1x4 grid of subplots
    for i in range(4):
        if i == 0:
            title = 'Stokes $I$'
        else:
            title = 'Stokes ' + labels[i] 
        axes[i].set_title(title)
        axes[i].plot(wavelengths, spectra_true[pixel_x, pixel_y, i, :], label='True')
        axes[i].plot(wavelengths, spectra_syn[pixel_x, pixel_y, i, :], label='Synthetic', color='orangered')
        axes[i].set_xlabel('$\lambda$ $[\\si{\\angstrom}]$')
        axes[i].set_ylabel(labels[i])
        axes[i].legend(loc='best')
    plt.tight_layout()  # Ensure proper spacing between subplots
    if savename != None:
        plt.savefig(savename)
    plt.show()

# Function to plot stokes I, Q, U and V
def plot_stokes_simple(spectra, wavelengths, spectra_idx, labels, points=13, savename=None, title=None):
    """
    Plot Stokes vector.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))  # Create a 1x4 grid of subplots
    if title is not None:
        plt.suptitle(title)
    spectras = spectra[spectra_idx,:].reshape(4,points)
    for i in range(4):
        if i == 0:
            title = 'Stokes $I$'
        else:
            title = 'Stokes ' + labels[i] 
        axes[i].set_title(title)
        axes[i].plot(wavelengths, spectras[i, :])
        axes[i].set_xlabel('$\lambda$ $[\\si{\\angstrom}]$')
        axes[i].set_ylabel(labels[i])
    plt.tight_layout()  # Ensure proper spacing between subplots
    if savename != None:
        plt.savefig(savename)
    plt.show()

# Function to plot stokes I, Q, U and V
def plot_stokes_single(spectrum, wavelengths, labels, points=13, savename=None, title=None):
    """
    Plot Stokes vector.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))  # Create a 1x4 grid of subplots
    if title is not None:
        plt.suptitle(title)
    spectras = spectrum.reshape(4,points)
    for i in range(4):
        if i == 0:
            title = 'Stokes $I$'
        else:
            title = 'Stokes ' + labels[i] 
        axes[i].set_title(title)
        axes[i].plot(wavelengths, spectras[i, :])
        axes[i].set_xlabel('$\lambda$ $[\\si{\\angstrom}]$')
        axes[i].set_ylabel(labels[i])
    plt.tight_layout()  # Ensure proper spacing between subplots
    if savename != None:
        plt.savefig(savename)
    plt.show()

# Function for observations plotting
def plot_four_images(array, wl_idx, labels, savename=None):
    """
    Plots four images next to each other for each Stokes parameter (I, Q, U, V) with entries (n, m, i, 0) 
    where i is in {0, 1, 2, 3} and n and m are integer numbers.
    array: Input np.array of shape (n, m, 4, 13).
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 3.7), sharex=True, sharey=True, gridspec_kw={'height_ratios': [1] * 1, 'width_ratios': [1, 1, 1, 1]})
    
    for i in range(4):
        im = axes[i].imshow(array[:, :, i, wl_idx], cmap='binary', aspect='equal')  # You can change the colormap as needed
        axes[i].set_title(f'{labels[i]}')
        cbar = fig.colorbar(im, ax=axes[i], orientation='vertical')
    
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename)
    plt.show()

# Define function to resize a 1-dimensional array to the three sigma range
def three_sigma(array):
    """
    Resizes a one-dimensional array of arbitrary length to the three sigma range.
    """
    # Calculate mean and standard deviation
    mean = np.mean(array)
    std = np.std(array)

    # Define lower and upper bounds
    lower_bound = mean - 3*std
    upper_bound = mean + 3*std

    # Resize array within the three sigma range
    resized_array = np.clip(array, lower_bound, upper_bound)

    return resized_array

# Define function to plot density plots
def plot_dens(inp, n, inp_names=None, savename=None, title=None):
    """
    Plot n density plots of inp with 5 plots at most each row, all row entries for a given column of inp are taken as the distribution.
    """
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 3*nrows))
    if title is not None:
        plt.suptitle(title)
    for i, ax in enumerate(axes.flat):
        if i < n:
            sns.kdeplot(three_sigma(inp[:,i]), bw_method=0.2, ax=ax, label=('Characteristics:\n$\\mu =$ %s, $\\sigma =$ %s' %(round(np.mean(inp[:,i]),2), round(np.std(inp[:,i]),2))))           
            if inp_names != None:
                ax.set_title(f'{inp_names[i]}')
            else:
                ax.set_title(f'Density {i+1}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend(loc='best')
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename)
    else:
        plt.show()

# Define function to plot histograms
def plot_hist(inp, n, inp_names=None, savename=None, title=None):
    """
    Plot n histograms of inp with 5 plots at most each row, all row entries for a given column of inp are taken as the distribution.
    """
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 3*nrows))
    if title is not None:
        plt.suptitle(title)
    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.hist(three_sigma(inp[:,i]), bins=200, label=('Characteristics:\n$\\mu =$ %s, $\\sigma =$ %s' %(round(np.mean(inp[:,i]),2), round(np.std(inp[:,i]),2))))
            if inp_names != None:
                ax.set_title(f'{inp_names[i]}')
            else:
                ax.set_title(f'Histogram {i+1}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Number of values per bin')
            ax.legend(loc='best')
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename)
    else:
        plt.show()

# Define function to plot histograms for train data, such as to find possible holes in the train data
def plot_histograms(data, labels, title, savename=None):
    """
    Plots histograms for each column in the input data array.
    
    data (numpy.ndarray): Input array of shape [n, 9].
    labels (list): List of 9 strings representing labels for each column.
    title (str): Title for the entire plot.
    """
    num_columns = data.shape[1]
    fig, axs = plt.subplots(3, 3, figsize=(7, 7))
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle(title, fontsize=13)

    for i in range(num_columns):
        row = i // 3
        col = i % 3
        ax = axs[row, col]

        # Plot histogram for the current column
        ax.hist(data[:, i], bins=20)
        ax.set_title(labels[i], fontsize=12)
        ax.set_xlabel('Values', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.tick_params(axis='both', labelsize=8)

    # If there are fewer than 9 columns, remove empty subplots
    for i in range(num_columns, 9):
        fig.delaxes(axs.flatten()[i])

    plt.tight_layout()
    if savename != None:
        plt.savefig(savename)

# Two-column histogram plot
def plot_histograms2(data1, data2, labels, unit_labels, title, label_data1, label_data2, savename=None, true_dens=False):
    """
    Plots histograms for each column in two input data arrays side by side.

    data1 (numpy.ndarray): First input array of shape [n, 9].
    data2 (numpy.ndarray): Second input array of shape [n, 9].
    labels (list): List of 9 strings representing labels for each plot.
    unit_labels: List of 9 strings representing unit labels for each plot.
    title (str): Title for the entire plot.
    savename (str, optional): File name to save the plot as an image, default is None.
    true_dens: Plot normalized frequency or not, default is False.
    """

    num_columns = data1.shape[1]
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle(title, fontsize=16)

    for i in range(num_columns):
        row = i // 3
        col = i % 3
        ax = axs[row, col]

        # Plot histogram for the current column from data1
        ax.hist(data1[:, i], bins=40, color='skyblue', alpha=0.5, label=label_data1, density=true_dens)

        # Overlay histogram for the current column from data2
        ax.hist(data2[:, i], bins=40, color='orange', alpha=0.5, label=label_data2, density=true_dens)

        ax.set_title(labels[i])
        ax.set_xlabel('Values ' + unit_labels[i], fontsize=9)
        ax.set_ylabel('Frequency (normalized)', fontsize=9)
        ax.legend(fontsize=9, loc='upper right')

    # If there are fewer than 9 columns, remove empty subplots
    for i in range(num_columns, 9):
        fig.delaxes(axs.flatten()[i])

    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename)
    plt.show()  # Added to display the plot if savename is None

# Two-column density plot
def plot_density_plots2(data1, data2, labels, title, label_data1, label_data2, savename=None):
    """
    Plots density plots for each column in two input data arrays side by side.

    data1 (numpy.ndarray): First input array of shape [n, 9].
    data2 (numpy.ndarray): Second input array of shape [n, 9].
    labels (list): List of 9 strings representing labels for each column.
    title (str): Title for the entire plot.
    label_data1 (str): Label for the first dataset.
    label_data2 (str): Label for the second dataset.
    savename (str, optional): File name to save the plot as an image, default is None.
    """
    num_columns = data1.shape[1]
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle(title, fontsize=16)

    for i in range(num_columns):
        row = i // 3
        col = i % 3
        ax = axs[row, col]

        # Create density plot for the current column from data1
        sns.kdeplot(data1[:, i], color='skyblue', label=label_data1, ax=ax)

        # Overlay density plot for the current column from data2
        sns.kdeplot(data2[:, i], color='orange', label=label_data2, ax=ax)

        ax.set_title(labels[i])
        ax.set_xlabel('Values')
        ax.set_ylabel('Density')
        ax.legend(fontsize=6)

    # If there are fewer than 9 columns, remove empty subplots
    for i in range(num_columns, 9):
        fig.delaxes(axs.flatten()[i])

    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename)
    plt.show()  # Added to display the plot if savename is None

# Define function to find value x of a probability density function p(x) for which p(x) = max
def find_max_dens(array, multidim=False, num_bins=50):
    '''
    Finds the values of maximal density in a potentially multidimensional array. Returns n maximal density values, 
    if the array consists of n columns and m rows.
    '''
    if multidim == True:
        max_density_values = np.zeros([array.shape[1]])
        for i in range(array.shape[1]):
            # Create histogram
            counts, edges = np.histogram(array[:,i], bins=num_bins, density=True)

            # Find the bin with maximal probability density
            max_density_bin_index = np.argmax(counts)

            # Calculate the mean of the values in the bin with maximal density
            bin_start = edges[max_density_bin_index]
            bin_end = edges[max_density_bin_index + 1]
            mean_of_max_density_bin = (bin_start + bin_end) / 2
            max_density_values[i] = mean_of_max_density_bin
        return max_density_values
    else:
        # Create histogram
        counts, edges = np.histogram(array, bins=num_bins, density=True)

        # Find the bin with maximal probability density
        max_density_bin_index = np.argmax(counts)

        # Calculate the mean of the values in the bin with maximal density
        bin_start = edges[max_density_bin_index]
        bin_end = edges[max_density_bin_index + 1]
        mean_of_max_density_bin = (bin_start + bin_end) / 2
        return mean_of_max_density_bin

# Define function to plot Milne-Eddington-inverted data against normalizing-flow-inverted data
def plot_colorplots(params1, params2, title1, title2, labels, savename=None, samescale=True):
    '''
    Create colorplots for a variable with dimensions (n, m, 9), where n and m are integer numbers.
    '''
    # Set font size
    mpl.rcParams['font.size'] = 7

    # Create 9 subplots in a 9x2 grid with shared x and y axes
    fig, axes = plt.subplots(9, 2, figsize=(5, 18), sharex=True, sharey=True)

    for i in range(0,9):
        # Extract plotting data
        param_data_1 = params1[:, :, i]
        param_data_2 = params2[:, :, i]

        if samescale == False:
            # Create a colorplot for the parameter in column 1
            im1 = axes[i, 0].imshow(param_data_1, cmap='binary', aspect='equal')
            axes[i, 0].set_title(f'{title1}\n{labels[i]}', fontsize=9)
            fig.colorbar(im1, ax=axes[i, 0], orientation='vertical')

            # Create a colorplot for the parameter in column 2
            im2 = axes[i, 1].imshow(param_data_2, cmap='binary', aspect='equal')
            axes[i, 1].set_title(f'{title2}\n{labels[i]}', fontsize=9)
            fig.colorbar(im2, ax=axes[i, 1], orientation='vertical')

        else:   
            # Calculate local minimum and maximum for each pair of plots
            local_min = min(np.min(param_data_1), np.min(param_data_2))
            local_max = max(np.max(param_data_1), np.max(param_data_2))

            # Create a colorplot for the parameter in column 1
            im1 = axes[i, 0].imshow(param_data_1, cmap='binary', aspect='equal', vmin=local_min, vmax=local_max)
            axes[i, 0].set_title(f'{title1}\n{labels[i]}', fontsize=9)
            fig.colorbar(im1, ax=axes[i, 0], orientation='vertical')

            # Create a colorplot for the parameter in column 2
            im2 = axes[i, 1].imshow(param_data_2, cmap='binary', aspect='equal', vmin=local_min, vmax=local_max)
            axes[i, 1].set_title(f'{title2}\n{labels[i]}', fontsize=9)
            fig.colorbar(im2, ax=axes[i, 1], orientation='vertical')

    # Adjust layout and show plots
    plt.tight_layout()
    if savename != None:
        plt.savefig(savename)
    plt.show()

    # Reset font size
    mpl.rcParams['font.size'] = 14



# Define function to invert maps using the trained normalizing flow
def invert_nflow(spectra, param_num, nflow, spectrum_scaler, params_scaler):
    '''
    Finds atmospheric parameters for a given dataset of spectra using a trained normalizing flow
    that has learned how to do Milne-Eddington inversions. Notice, that the variable spectra must have dimensions (n, m), 
    where n is the number of spectra and m is the number of wavelength points (concatenated Stokes values for different wavelengths).
    Returns array of dimensions (n, 9), where the second dimension is the number of different parameters, 
    which are calculated as mean values over the inverted values for the parameters and standard deviations for each inverted spectrum.

    spectra: Array of flattened Stokes vectors for each pixel of a map.
    param_num: Number of parameters of the model to invert.
    nflow: Name of trained normalizing flow.
    spectrum_scaler: Trained scaler for spectra.
    params_scaler: Trained scaler for output parameters.
    '''
    # Define parameters array and parameter error array
    parameters = np.zeros((spectra.shape[0], param_num))
    parameters_std = np.zeros((spectra.shape[0], param_num))

    # Define number of samples from a Gaussian to pass to the normalizing flow alongside with the spectrum to invert
    samples = 500

    # Define array for maximum density values of parameter distributions
    max_density_values = np.zeros([parameters.shape[1]])

    # Loop over all spectra in the spectra dataset
    for i in tqdm(range(spectra.shape[0])):

        # Scale spectrum with the passed spectrum scaler to pass to the normalizing flow
        i_spectrum_std = torch.tensor(spectrum_scaler.transform(torch.tensor(np.repeat(spectra[i, :][None, :], 1, axis=0))), dtype=torch.float32)

        # Scale obtained parameters to normal values using the passed parameter scaler
        i_parameters = params_scaler.inverse_transform(nflow.sample(samples, context=i_spectrum_std).detach().numpy()[0])

        # Write parameters and parameter errors to numpy arrays
        parameters[i, :] = find_max_dens(i_parameters, multidim=True)
        parameters_std[i, :] = np.std(i_parameters, axis=0)
            
    return parameters, parameters_std

# Define function to interpolate between wavelength points of observations
def interp(dataset, oldwav, newwav):
    """
    Interpolates between wavelength points of observations.
    
    dataset: Array of shape (550, 600, 4, 13) containing Stokes vector observations for a certain FOV.
    oldwav: Old wavelength points array.
    newwav: New wavelength points array.
    newpoints: Number of new wavelength points (13 --> newpoints).
    """
    # Define new dataset
    interpolated_dataset = np.empty((550, 600, 4, len(newwav)))
    reshaped_dataset = dataset.reshape((550, 600, 4*13))

    # Loop over all pixels
    for i in tqdm(range(550)):
        for j in range(600):
            for k in range(4):
                original_spectrum = reshaped_dataset[i, j, k * 13:(k + 1) * 13]
                #print(original_spectrum.shape)
                interpolation_function = interp1d(oldwav, original_spectrum, kind='cubic')
                interpolated_spectrum = interpolation_function(newwav)
                #print(interpolated_spectrum.shape)
                interpolated_dataset[i, j, k, :] = interpolated_spectrum

    # Return interpolated dataset
    return interpolated_dataset

# Define a class for bayesian inference using the Bayes theorem and MCMC sampling
class Bayesian_Inference(object):
    def __init__(self, true_x, min_x, max_x, noise=8e-3):
        """
        regions: Wavelength points.
        lines: Line identifiers.
        noise: Noise estimate.
        true_x: True parameter values pertaining to the observation for which an mcmc sampling is to be conducted.
        min_x: Minimum parameter values o training dataset.
        max_x: Maximum parameter values of training dataset.
        """
        self.regions = [[np.array([6302.2134, 6302.253 , 6302.293 , 6302.333 , 6302.373 , 
                            6302.413 , 6302.453 , 6302.493 , 6302.533 , 6302.573 , 
                            6302.6133, 6302.6533, 6302.6934]), None]] # Change to available data if necessary
        self.lines = [6302] # Change to available data if necessary
        self.noise = noise
        self.true_x = np.array(true_x)
        self.min_x = np.array(min_x)
        self.max_x = np.array(max_x)
        self.ME = MilneEddington.MilneEddington(self.regions, self.lines)
        self.wl_points = len(self.regions[0][0])

    def gen_obs(self):
        self.obs = self.synth(self.true_x).reshape((4*self.wl_points,))
        self.obs_noise = self.obs + self.noise*np.random.randn(4*self.wl_points,) 

    def synth(self, x):
        """
        x: Parameter array.
        """
        syn = self.ME.synthesize(x).reshape((4*self.wl_points,))
        return syn

    def log_prob(self, x):
        """
        x: Parameter array.
        """
        # Call gen_obs
        self.gen_obs()
        
        # Return zero probability for impossible values of parameters
        for i in range(9):
            if (x[i] < self.min_x[i] or x[i] > self.max_x[i]):
                return -np.inf	

        synt = self.synth(x)

        chi2 = np.sum((synt - self.obs_noise)**2/self.noise**2)

        return -0.5 * chi2

    def sample(self):
        ndim = 9
        nwalkers = 1000
        p0 = self.true_x[None, :] + 1e-2*np.random.randn(nwalkers, ndim)
        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_prob)

        self.sampler.run_mcmc(p0, 1000, progress=True)

        self.samples = self.sampler.get_chain(discard=100, flat=True)

        return self.samples

# Define function to extract and average rectangles of maps
def extract_and_average_rectangles(params_list, r, c, row_start, col_start):
    """
    Extracts a rectangle of data from a map and averages over all columns per line.

    params_list: List of parameter arrays.
    r: Number of rows (height of rectangle).
    c: Number of columns (width of rectangle).
    row_start: y-coordinate of upper left edge of rectangle.
    col_start: x-coordinate of upper left edge of rectangle.
    """
    # Initialize an empty array to store the results
    result_array = np.zeros((r, len(params_list), params_list[0].shape[2]))

    # Iterate over each array in params_list
    for i, array in enumerate(params_list):
        # Iterate over each dimension (third axis)
        for dim in range(array.shape[2]):
            # Extract the rectangle and calculate the column-wise average
            extracted_rect = array[row_start:row_start + r, col_start:col_start + c, dim]
            column_avg = np.mean(extracted_rect, axis=1)

            # Store the result in the result_array
            result_array[:, i, dim] = column_avg

    return result_array

# Define function to extract vertical slices from maps
def extract_vertical_slices(original_arrays, center_index):
    """
    Extract 1-pixel wide vertical slices from each dimension of the input arrays.
    
    original_arrays: List of input arrays, each with shape (550, 600, 9).
    center_index: Index of the center pixel for the vertical slice (default is 300).
    new_arrays: NumPy array of shape (n, 550, 9) containing the extracted slices.
    """
    n = len(original_arrays)
    new_arrays = np.zeros((n, 550, 9), dtype=original_arrays[0].dtype)

    for i, original_array in enumerate(original_arrays):
        for j in range(9):
            vertical_slice = original_array[:, center_index, j]
            new_arrays[i, :, j] = vertical_slice

    return new_arrays.transpose(1, 0, 2)

# Plot parameter evolution plot
def plot_color_plots(data_array, additional_array, hl_pix_vert=None, hl_max_hor=None, hl_min_hor=None, labels=None, units=None, title=None, savename=None):
    """
    data_array: Actual evolution data.
    additional_array: Additional array containing 9 images (maps) for every of the nine parameters which show the initial situation.
    """
    n, _, _ = data_array.shape

    # Create a subplot for each dimension
    fig, axes = plt.subplots(9, 2, figsize=(12, 40))  # Change the figsize as needed
    if title is not None:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle('Colorplots', fontsize=16)

    # Plot the color maps for the first array (data_array)
    for dim in range(9):
        # Plot the color map for the current dimension
        ax = axes[dim, 1]
        im = ax.imshow(data_array[:, :, dim], aspect='auto', cmap='binary')

        # Set labels and title
        ax.set_xlabel('Map number \\footnotesize (measure of time)')
        ax.set_ylabel('Vertical pixel')

        # Customize the x-axis ticks
        x_ticks = np.arange(data_array.shape[1])
        x_ticks_labels = x_ticks + 1  # Add +1 to every tick
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks_labels.astype(int))

        # Add color bar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(units[dim])

        if labels is not None:
            ax.set_title('Evolution of ' + labels[dim])
        else:
            ax.set_title(f'Dimension {dim + 1}')

    # Highlight the specified lines in the second column
    if hl_max_hor is not None and hl_min_hor is not None:
        for dim in range(9):
            ax = axes[dim, 1]
            ax.axhline(y=hl_max_hor, color='red', linestyle='-', linewidth=1)
            ax.axhline(y=hl_min_hor, color='red', linestyle='-', linewidth=1)

    # Highlight the specified pixel in the first column
    if hl_pix_vert is not None and hl_max_hor is not None and hl_min_hor is not None:
        for dim in range(9):
            ax = axes[dim, 0]
            ax.axvline(x=hl_pix_vert, color='red', linestyle='-', linewidth=1)
            ax.axhline(y=hl_max_hor, color='red', linestyle='-', linewidth=1)
            ax.axhline(y=hl_min_hor, color='red', linestyle='-', linewidth=1)

    # Plot the color maps for the second array (additional_array)
    for dim in range(9):
        # Plot the color map for the current dimension from the additional array
        ax = axes[dim, 0]
        im = ax.imshow(additional_array[:, :, dim], aspect='equal', cmap='binary')

        # Set labels and title
        ax.set_xlabel('Horizontal pixel')
        ax.set_ylabel('Vertical pixel')

        if labels is not None:
            ax.set_title(labels[dim])
        else:
            ax.set_title(f'Dimension {dim + 1} (Additional)')

        # Add color bar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(units[dim])

    # Adjust layout and show plots
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename)
    plt.show()

# Plot parameter evolution plot
def plot_color_plots_2(params_array, errors_array, initial_array, r=None, c=None, row_start=None, col_start=None, labels=None, units=None, title=None, savename=None):
    """
    params_array: Actual evolution data of red rectangle.
    errors_array: Actual errors for evolution data of red rectangle.
    initial_array: Additional array containing 9 images (maps) for every of the nine parameters which show the initial situation.
    """
    # Extract params_array shape
    n, _, _ = params_array.shape

    # Define spacing between suptitles and plots
    spacing = 1.05

    # Set font size
    mpl.rcParams['font.size'] = 18
    mpl.rcParams['xtick.labelsize'] = 12  # Set x-axis tick label size
    mpl.rcParams['ytick.labelsize'] = 12  # Set y-axis tick label size

    # Create a subplot for each dimension
    fig, axes = plt.subplots(9, 3, figsize=(18, 40))  # Change the figsize as needed
    if title is not None:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle('Colorplots', fontsize=16)

    # Plot second column with parameter evolution for red rectangle
    for dim in range(9):
        # Plot the color map for the current dimension
        ax = axes[dim, 1]
        im = ax.imshow(params_array[:, :, dim], aspect='auto', cmap='binary')

        # Set labels and title
        ax.set_xlabel('Map number \\Large (measure of time)')
        ax.set_ylabel('Vertical pixel')

        # Customize the x-axis ticks
        x_ticks = np.arange(params_array.shape[1])
        x_ticks_labels = x_ticks + 1  # Add +1 to every tick
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks_labels.astype(int))

        # Customize the y-axis ticks
        y_ticks_step = params_array.shape[0] // 6  # Adjust the step as needed
        y_ticks = np.arange(0, params_array.shape[0], y_ticks_step)
        y_ticks_labels = np.arange(params_array.shape[0]) + row_start
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks_labels[y_ticks].astype(int))

        # Add a rectangle to the existing axes
        if r is not None and c is not None and row_start is not None and col_start is not None:
            rectangle = Rectangle((col_start, row_start), c, r, linewidth=1, edgecolor='red', facecolor='none')
            axes[dim,0].add_patch(rectangle)
        
        # Add color bar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(units[dim])

        if labels is not None:
            ax.set_title('Evolution of ' + labels[dim] + '\n\\Large(average over all columns per row in red rectangle)', y=spacing)
        else:
            ax.set_title(f'Dimension {dim + 1}', y=spacing)

    # Plot third column with errors for red rectangle
    for dim in range(9):
        # Plot the color map for the current dimension
        ax = axes[dim, 2]
        im = ax.imshow(errors_array[:, :, dim], aspect='auto', cmap='binary')

        # Set labels and title
        ax.set_xlabel('Map number \\Large (measure of time)')
        ax.set_ylabel('Vertical pixel')

        # Customize the x-axis ticks
        x_ticks = np.arange(errors_array.shape[1])
        x_ticks_labels = x_ticks + 1  # Add +1 to every tick
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks_labels.astype(int))

        # Customize the y-axis ticks
        y_ticks_step = errors_array.shape[0] // 6  # Adjust the step as needed
        y_ticks = np.arange(0, errors_array.shape[0], y_ticks_step)
        y_ticks_labels = np.arange(errors_array.shape[0]) + row_start
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks_labels[y_ticks].astype(int))
        
        # Add color bar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(units[dim])

        if labels is not None:
            ax.set_title('Error evolution of ' + labels[dim] + '\n\\Large(average over all columns per row in red rectangle)', y=spacing)
        else:
            ax.set_title(f'Dimension {dim + 1}', y=1.02)

    # Plot map for initial parameter value situation
    for dim in range(9):
        # Plot the color map for the current dimension from the additional array
        ax = axes[dim, 0]
        im = ax.imshow(initial_array[:, :, dim], aspect='equal', cmap='binary')

        # Set labels and title
        ax.set_xlabel('Horizontal pixel')
        ax.set_ylabel('Vertical pixel')

        if labels is not None:
            ax.set_title(labels[dim], y=1.02)
        else:
            ax.set_title(f'Dimension {dim + 1} (Additional)', y=1.02)

        # Add color bar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(units[dim])

    # Adjust layout and show plots
    plt.tight_layout()  # Adjust rect parameter to leave space for suptitle
    if savename is not None:
        plt.savefig(savename)
    plt.show()

    # Set font size back to normal
    mpl.rcParams['font.size'] = 14

# Define function to plot Milne-Eddington-inverted data against normalizing-flow-inverted data
def plot_colorplots_3(params1, params2, params3, title1, title2, title3, labels, savename=None, samescale=True):
    '''
    Create colorplots for a variable with dimensions (n, m, 9), where n and m are integer numbers. Params3 are errors of Params2.
    '''
    # Set font size
    mpl.rcParams['font.size'] = 7

    # Create 9 subplots in a 9x3 grid with shared x and y axes
    fig, axes = plt.subplots(9, 3, figsize=(7.5, 18), sharex=True, sharey=True)

    for i in range(0, 9):
        # Extract plotting data
        param_data_1 = params1[:, :, i]
        param_data_2 = params2[:, :, i]
        param_data_3 = params3[:, :, i]

        if samescale == False:
            # Create a colorplot for the parameter in column 1
            im1 = axes[i, 0].imshow(param_data_1, cmap='binary', aspect='equal')
            axes[i, 0].set_title(f'{title1}\n{labels[i]}', fontsize=9)
            fig.colorbar(im1, ax=axes[i, 0], orientation='vertical')

            # Create a colorplot for the parameter in column 2
            im2 = axes[i, 1].imshow(param_data_2, cmap='binary', aspect='equal')
            axes[i, 1].set_title(f'{title2}\n{labels[i]}', fontsize=9)
            fig.colorbar(im2, ax=axes[i, 1], orientation='vertical')

            # Create a colorplot for the parameter in column 3
            im3 = axes[i, 2].imshow(param_data_3, cmap='binary', aspect='equal')
            axes[i, 2].set_title(f'{title3}\n{labels[i]}', fontsize=9)
            fig.colorbar(im3, ax=axes[i, 2], orientation='vertical')

        else:
            # Calculate local minimum and maximum for each set of plots
            local_min = min(np.min(param_data_1), np.min(param_data_2), np.min(param_data_3))
            local_max = max(np.max(param_data_1), np.max(param_data_2), np.max(param_data_3))

            # Create a colorplot for the parameter in column 1
            im1 = axes[i, 0].imshow(param_data_1, cmap='binary', aspect='equal', vmin=local_min, vmax=local_max)
            axes[i, 0].set_title(f'{title1}\n{labels[i]}', fontsize=9)
            fig.colorbar(im1, ax=axes[i, 0], orientation='vertical')

            # Create a colorplot for the parameter in column 2
            im2 = axes[i, 1].imshow(param_data_2, cmap='binary', aspect='equal', vmin=local_min, vmax=local_max)
            axes[i, 1].set_title(f'{title2}\n{labels[i]}', fontsize=9)
            fig.colorbar(im2, ax=axes[i, 1], orientation='vertical')

            # Create a colorplot for the parameter in column 3
            im3 = axes[i, 2].imshow(param_data_3, cmap='binary', aspect='equal', vmin=local_min, vmax=local_max)
            axes[i, 2].set_title(f'{title3}\n{labels[i]}', fontsize=9)
            fig.colorbar(im3, ax=axes[i, 2], orientation='vertical')

    # Adjust layout and show plots
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename)
    plt.show()

    # Reset font size
    mpl.rcParams['font.size'] = 14

def plot_inversion_results(params1, params2, params3, title1, title2, title3, labels, colorbar_labels, savename=None, samescale=True, cmap='binary'):
    '''
    Create colorplots for three arrays of dimension (n, m, 9), where n and m are integer numbers. Visualizes results for inversion
    done by Milne-Eddington model and the normalizing flow model.

    params1: Array of shape (n, m, 9), denotes results of Milne-Eddington model.
    params2: Array of shape (n, m, 9), denotes results of normalizing flow model.
    params3: Errors of params2 in order to explain performance of the normalizing flow model.
    title1: Title for params1.
    title2: Title for params2.
    title3: Title for params3.
    labels: Labels for plots.
    colorbar_labels: Labels for colorbars.
    savename: Name for saving, default is None.
    samescale: If true, the plots for Milne-Eddington and normalizing flow model results are plotted on the same scale.
    cmap: Colorscheme for plots, default is binary.
    '''
    # Set font size
    mpl.rcParams['font.size'] = 7

    # Create 9 subplots in a 9x4 grid with shared x and y axes
    fig, axes = plt.subplots(9, 3, figsize=(10, 18), sharex=True, sharey=True, gridspec_kw={'height_ratios': [1] * 9, 'width_ratios': [1, 1, 1]})

    for i in range(0, 9):
        # Extract plotting data
        param_data_1 = params1[:, :, i]
        param_data_2 = params2[:, :, i]
        param_data_3 = params3[:, :, i]

        if samescale == False:
            # Create a colorplot for the parameter in column 1
            im1 = axes[i, 0].imshow(param_data_1, cmap=cmap, aspect='equal')
            if i == 0:
                axes[i, 0].set_title(f'\\large {title1}\n\n{labels[i]}', fontsize=9)
            else:
                axes[i, 0].set_title(f'{labels[i]}', fontsize=9)
            cbar1 = fig.colorbar(im1, ax=axes[i, 0], orientation='vertical', label=colorbar_labels[i])

            # Create a colorplot for the parameter in column 2
            im2 = axes[i, 1].imshow(param_data_2, cmap=cmap, aspect='equal')
            if i == 0:
                axes[i, 1].set_title(f'\\large {title2}\n\n{labels[i]}', fontsize=9)
            else:
                axes[i, 1].set_title(f'{labels[i]}', fontsize=9)
            cbar2 = fig.colorbar(im2, ax=axes[i, 1], orientation='vertical', label=colorbar_labels[i])

            # Create a colorplot for the parameter in column 3
            im3 = axes[i, 2].imshow(param_data_3, cmap=cmap, aspect='equal')
            if i == 0:
                axes[i, 2].set_title(f'\\large {title3}\n\n{labels[i]}', fontsize=9)
            else:
                axes[i, 2].set_title(f'{labels[i]}', fontsize=9)
            cbar3 = fig.colorbar(im3, ax=axes[i, 2], orientation='vertical', label=colorbar_labels[i])

        else:
            # Calculate local minimum and maximum for each set of plots
            local_min = min(np.min(param_data_1), np.min(param_data_2))
            local_max = max(np.max(param_data_1), np.max(param_data_2))

            # Create a colorplot for the parameter in column 1
            im1 = axes[i, 0].imshow(param_data_1, cmap=cmap, aspect='equal', vmin=local_min, vmax=local_max)
            if i == 0:
                axes[i, 0].set_title(f'\\large {title1}\n\n{labels[i]}', fontsize=9)
            else:
                axes[i, 0].set_title(f'{labels[i]}', fontsize=9)
            cbar1 = fig.colorbar(im1, ax=axes[i, 0], orientation='vertical', label=colorbar_labels[i])

            # Create a colorplot for the parameter in column 2
            im2 = axes[i, 1].imshow(param_data_2, cmap=cmap, aspect='equal', vmin=local_min, vmax=local_max)
            if i == 0:
                axes[i, 1].set_title(f'\\large {title2}\n\n{labels[i]}', fontsize=9)
            else:
                axes[i, 1].set_title(f'{labels[i]}', fontsize=9)
            cbar2 = fig.colorbar(im2, ax=axes[i, 1], orientation='vertical', label=colorbar_labels[i])

            # Create a colorplot for the parameter in column 3
            im3 = axes[i, 2].imshow(param_data_3, cmap=cmap, aspect='equal')
            if i == 0:
                axes[i, 2].set_title(f'\\large {title3}\n\n{labels[i]}', fontsize=9)
            else:
                axes[i, 2].set_title(f'{labels[i]}', fontsize=9)
            cbar3 = fig.colorbar(im3, ax=axes[i, 2], orientation='vertical', label=colorbar_labels[i])

    # Adjust layout and show plots
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename)
    plt.show()

    # Reset font size
    mpl.rcParams['font.size'] = 14

def plot_inversion_results_seq(params1, params2, params3, params4, title1, title2, title3, title4, labels, colorbar_labels, savename=None, samescale=True, cmap='binary'):
    '''
    Create colorplots for four arrays of dimension (n, m, 9), where n and m are integer numbers. Visualizes results for inversion
    done by Milne-Eddington model and the normalizing flow model.

    params1: Array of shape (n, m, 9), denotes results of Milne-Eddington model.
    params2: Array of shape (n, m, 9), denotes results 1 of normalizing flow model.
    params3: Array of shape (n, m, 9), denotes results 2 of normalizing flow model.
    params4: Array of shape (n, m, 9), denotes results 3 of normalizing flow model.
    title1: Title for params1.
    title2: Title for params2.
    title3: Title for params3.
    title4: Title for params4.
    labels: Labels for plots.
    colorbar_labels: Labels for colorbars.
    savename: Name for saving, default is None.
    samescale: If true, all plots are plotted on the same scale, default is True.
    cmap: Colorscheme for plots, default is binary.
    '''
    # Set font size
    mpl.rcParams['font.size'] = 7

    # Create 9 subplots in a 9x4 grid with shared x and y axes
    fig, axes = plt.subplots(9, 4, figsize=(12, 18), sharex=True, sharey=True, gridspec_kw={'height_ratios': [1] * 9, 'width_ratios': [1, 1, 1, 1]})

    for i in range(0, 9):
        # Extract plotting data
        param_data_1 = params1[:, :, i]
        param_data_2 = params2[:, :, i]
        param_data_3 = params3[:, :, i]
        param_data_4 = params4[:, :, i]

        if samescale == False:
            # Create a colorplot for the parameter in column 1
            im1 = axes[i, 0].imshow(param_data_1, cmap=cmap, aspect='equal')
            if i == 0:
                axes[i, 0].set_title(f'\\large {title1}\n\n{labels[i]}', fontsize=9)
            else:
                axes[i, 0].set_title(f'{labels[i]}', fontsize=9)
            cbar1 = fig.colorbar(im1, ax=axes[i, 0], orientation='vertical', label=colorbar_labels[i])

            # Create a colorplot for the parameter in column 2
            im2 = axes[i, 1].imshow(param_data_2, cmap=cmap, aspect='equal')
            if i == 0:
                axes[i, 1].set_title(f'\\large {title2}\n\n{labels[i]}', fontsize=9)
            else:
                axes[i, 1].set_title(f'{labels[i]}', fontsize=9)
            cbar2 = fig.colorbar(im2, ax=axes[i, 1], orientation='vertical', label=colorbar_labels[i])

            # Create a colorplot for the parameter in column 3
            im3 = axes[i, 2].imshow(param_data_3, cmap=cmap, aspect='equal')
            if i == 0:
                axes[i, 2].set_title(f'\\large {title3}\n\n{labels[i]}', fontsize=9)
            else:
                axes[i, 2].set_title(f'{labels[i]}', fontsize=9)
            cbar3 = fig.colorbar(im3, ax=axes[i, 2], orientation='vertical', label=colorbar_labels[i])

            # Create a colorplot for the parameter in column 4
            im3 = axes[i, 3].imshow(param_data_4, cmap=cmap, aspect='equal')
            if i == 0:
                axes[i, 3].set_title(f'\\large {title4}\n\n{labels[i]}', fontsize=9)
            else:
                axes[i, 3].set_title(f'{labels[i]}', fontsize=9)
            cbar3 = fig.colorbar(im4, ax=axes[i, 3], orientation='vertical', label=colorbar_labels[i])

        else:
            # Calculate local minimum and maximum for each set of plots
            local_min = min(np.min(param_data_1), np.min(param_data_2), np.min(param_data_3), np.min(param_data_4))
            local_max = max(np.max(param_data_1), np.max(param_data_2), np.min(param_data_3), np.min(param_data_4))

            # Create a colorplot for the parameter in column 1
            im1 = axes[i, 0].imshow(param_data_1, cmap=cmap, aspect='equal', vmin=local_min, vmax=local_max)
            if i == 0:
                axes[i, 0].set_title(f'\\large {title1}\n\n{labels[i]}', fontsize=9)
            else:
                axes[i, 0].set_title(f'{labels[i]}', fontsize=9)
            cbar1 = fig.colorbar(im1, ax=axes[i, 0], orientation='vertical', label=colorbar_labels[i])

            # Create a colorplot for the parameter in column 2
            im2 = axes[i, 1].imshow(param_data_2, cmap=cmap, aspect='equal', vmin=local_min, vmax=local_max)
            if i == 0:
                axes[i, 1].set_title(f'\\large {title2}\n\n{labels[i]}', fontsize=9)
            else:
                axes[i, 1].set_title(f'{labels[i]}', fontsize=9)
            cbar2 = fig.colorbar(im2, ax=axes[i, 1], orientation='vertical', label=colorbar_labels[i])

            # Create a colorplot for the parameter in column 3
            im3 = axes[i, 2].imshow(param_data_3, cmap=cmap, aspect='equal', vmin=local_min, vmax=local_max)
            if i == 0:
                axes[i, 2].set_title(f'\\large {title3}\n\n{labels[i]}', fontsize=9)
            else:
                axes[i, 2].set_title(f'{labels[i]}', fontsize=9)
            cbar3 = fig.colorbar(im3, ax=axes[i, 2], orientation='vertical', label=colorbar_labels[i])

            # Create a colorplot for the parameter in column 4
            im4 = axes[i, 3].imshow(param_data_4, cmap=cmap, aspect='equal', vmin=local_min, vmax=local_max)
            if i == 0:
                axes[i, 3].set_title(f'\\large {title4}\n\n{labels[i]}', fontsize=9)
            else:
                axes[i, 3].set_title(f'{labels[i]}', fontsize=9)
            cbar4 = fig.colorbar(im4, ax=axes[i, 3], orientation='vertical', label=colorbar_labels[i])

    # Adjust layout and show plots
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename)
    plt.show()

    # Reset font size
    mpl.rcParams['font.size'] = 14

# Define function to plot Milne-Eddington-inverted data against normalizing-flow-inverted data
def plot_colorplots_seq(params1, params2, params3, params4, title1, title2, title3, title4, labels, savename=None, samescale=True):
    '''
    Create colorplots for a variable with dimensions (n, m, 9), where n and m are integer numbers; params4 is exact solution.
    '''
    # Set font size
    mpl.rcParams['font.size'] = 6

    # Create 9 subplots in a 9x4 grid with shared x and y axes
    fig, axes = plt.subplots(9, 4, figsize=(8, 20), sharex=True, sharey=True)

    for i in range(0, 9):
        # Extract plotting data
        param_data_1 = params1[:, :, i]
        param_data_2 = params2[:, :, i]
        param_data_3 = params3[:, :, i]
        param_data_4 = params4[:, :, i]

        if samescale == False:
            # Create a colorplot for the parameter in column 1
            im1 = axes[i, 0].imshow(param_data_1, cmap='binary', aspect='equal')
            axes[i, 0].set_title(f'{title1}\n{labels[i]}', fontsize=8)
            fig.colorbar(im1, ax=axes[i, 0], orientation='vertical')

            # Create a colorplot for the parameter in column 2
            im2 = axes[i, 1].imshow(param_data_2, cmap='binary', aspect='equal')
            axes[i, 1].set_title(f'{title2}\n{labels[i]}', fontsize=8)
            fig.colorbar(im2, ax=axes[i, 1], orientation='vertical')

            # Create a colorplot for the parameter in column 3
            im3 = axes[i, 2].imshow(param_data_3, cmap='binary', aspect='equal')
            axes[i, 2].set_title(f'{title3}\n{labels[i]}', fontsize=8)
            fig.colorbar(im3, ax=axes[i, 2], orientation='vertical')

            # Create a colorplot for the parameter in column 4
            im4 = axes[i, 3].imshow(param_data_4, cmap='binary', aspect='equal')
            axes[i, 3].set_title(f'{title4}\n{labels[i]}', fontsize=8)
            fig.colorbar(im4, ax=axes[i, 3], orientation='vertical')

        else:
            # Calculate local minimum and maximum for each set of plots
            local_min = min(np.min(param_data_1), np.min(param_data_2), np.min(param_data_3), np.min(param_data_4))
            local_max = max(np.max(param_data_1), np.max(param_data_2), np.max(param_data_3), np.max(param_data_4))

            # Create a colorplot for the parameter in column 1
            im1 = axes[i, 0].imshow(param_data_1, cmap='binary', aspect='equal', vmin=local_min, vmax=local_max)
            axes[i, 0].set_title(f'{title1}\n{labels[i]}', fontsize=8)
            fig.colorbar(im1, ax=axes[i, 0], orientation='vertical')

            # Create a colorplot for the parameter in column 2
            im2 = axes[i, 1].imshow(param_data_2, cmap='binary', aspect='equal', vmin=local_min, vmax=local_max)
            axes[i, 1].set_title(f'{title2}\n{labels[i]}', fontsize=8)
            fig.colorbar(im2, ax=axes[i, 1], orientation='vertical')

            # Create a colorplot for the parameter in column 3
            im3 = axes[i, 2].imshow(param_data_3, cmap='binary', aspect='equal', vmin=local_min, vmax=local_max)
            axes[i, 2].set_title(f'{title3}\n{labels[i]}', fontsize=8)
            fig.colorbar(im3, ax=axes[i, 2], orientation='vertical')

            # Create a colorplot for the parameter in column 4
            im4 = axes[i, 3].imshow(param_data_4, cmap='binary', aspect='equal', vmin=local_min, vmax=local_max)
            axes[i, 3].set_title(f'{title4}\n{labels[i]}', fontsize=8)
            fig.colorbar(im4, ax=axes[i, 3], orientation='vertical')

    # Adjust layout and show plots
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename)
    plt.show()

    # Reset font size
    mpl.rcParams['font.size'] = 14

# Define function to plot Milne-Eddington-inverted data against normalizing-flow-inverted data with residuals
def plot_colorplots_res(params1, params2, params3, title1, title2, title3, labels, savename=None, samescale=True):
    '''
    Create colorplots for a variable with dimensions (n, m, 9), where n and m are integer numbers.
    '''
    # Set font size
    mpl.rcParams['font.size'] = 7

    # Create 9 subplots in a 9x2 grid with shared x and y axes
    fig, axes = plt.subplots(9, 3, figsize=(5, 18), sharex=True, sharey=True)

    for i in range(0,9):
        # Extract plotting data
        param_data_1 = params1[:, :, i]
        param_data_2 = params2[:, :, i]
        param_data_3 = params3[:, :, i]

        if samescale == False:
            # Create a colorplot for the parameter in column 1
            im1 = axes[i, 0].imshow(param_data_1, cmap='binary', aspect='equal')
            axes[i, 0].set_title(f'{title1}\n{labels[i]}', fontsize=9)
            fig.colorbar(im1, ax=axes[i, 0], orientation='vertical')

            # Create a colorplot for the parameter in column 2
            im2 = axes[i, 1].imshow(param_data_2, cmap='binary', aspect='equal')
            axes[i, 1].set_title(f'{title2}\n{labels[i]}', fontsize=9)
            fig.colorbar(im2, ax=axes[i, 1], orientation='vertical')

            # Create a colorplot for the parameter in column 3
            im3 = axes[i, 2].imshow(param_data_3, cmap='binary', aspect='equal')
            axes[i, 2].set_title(f'{title3}\n{labels[i]}', fontsize=9)
            fig.colorbar(im3, ax=axes[i, 2], orientation='vertical')

        else:   
            # Calculate local minimum and maximum for each pair of plots
            local_min = min(np.min(param_data_1), np.min(param_data_2))
            local_max = max(np.max(param_data_1), np.max(param_data_2))

            # Create a colorplot for the parameter in column 1
            im1 = axes[i, 0].imshow(param_data_1, cmap='binary', aspect='equal', vmin=local_min, vmax=local_max)
            axes[i, 0].set_title(f'{title1}\n{labels[i]}', fontsize=9)
            fig.colorbar(im1, ax=axes[i, 0], orientation='vertical')

            # Create a colorplot for the parameter in column 2
            im2 = axes[i, 1].imshow(param_data_2, cmap='binary', aspect='equal', vmin=local_min, vmax=local_max)
            axes[i, 1].set_title(f'{title2}\n{labels[i]}', fontsize=9)
            fig.colorbar(im2, ax=axes[i, 1], orientation='vertical')

            # Create a colorplot for the parameter in column 3
            im3 = axes[i, 2].imshow(param_data_3, cmap='binary', aspect='equal')
            axes[i, 2].set_title(f'{title3}\n{labels[i]}', fontsize=9)
            fig.colorbar(im3, ax=axes[i, 2], orientation='vertical')

    # Adjust layout and show plots
    plt.tight_layout()
    if savename != None:
        plt.savefig(savename)
    plt.show()

    # Reset font size
    mpl.rcParams['font.size'] = 14

# Data generation using the Milne-Eddington model (uniform sampling at random)
def genData(num_data, min_par, max_par):
    """
    num_data: Number of datapoints to generate.
    min_par: Minimum parameter values to generate spectra for.
    max_par: Maximum parameter values to generate spectra for.
    """
    # Initialize Milne-Eddington object for wavelength range of penumbra formation dataset
    regions = [[np.linspace(6302.2134, 6302.6934, num=51, endpoint=True), None]]
    lines = [6302]
    ME = MilneEddington.MilneEddington(regions, lines)

    # Initialize empty arrays to store data in
    parameters = []
    spectra = []

    # Sample parameter sets from defined ranges [|B|, theta, varphi, v_los, v_dop, eta_0, a, S_0, S_1]
    for k in range(num_data):
        par_in = np.float64([
            np.random.uniform(min_par[0], max_par[0]),   # Randomly sample magnetic field strength within specified range in [G]
            np.random.uniform(min_par[1], max_par[1]),   # Randomly sample inclination within specified range in [rad]
            np.random.uniform(min_par[2], max_par[2]),   # Randomly sample azimuth within specified range in [rad]
            np.random.uniform(min_par[3], max_par[3]),   # Randomly sample line of sight velocity within specified range in [km/s]
            np.random.uniform(min_par[4], max_par[4]),   # Randomly sample doppler shift within specified range in [Angstrom]
            np.random.uniform(min_par[5], max_par[5]),   # Randomly sample line strength parameter within specified range
            np.random.uniform(min_par[6], max_par[6]),   # Randomly sample damping parameter within specified range
            np.random.uniform(min_par[7], max_par[7]),   # Randomly sample  within specified range
            np.random.uniform(min_par[8], max_par[8])    # Randomly sample S1 within specified range
        ])

        obs_out = ME.synthesize(par_in)

        # Write parameters and associated spectra to dataset
        parameters.append(par_in)
        spectra.append(obs_out)

    # Convert datasets to numpy-arrays
    parameters, spectra = np.array(parameters), np.array(spectra)
    regions = np.array(regions[0][0])
    return parameters, spectra, regions

# Data generation using the Milne-Eddington model (uniform sampling with grid)
def genData1(num_data, min_par, max_par):
    """
    vals_per_par: Number of datapoints to generate.
    min_par: Minimum parameter values to generate spectra for.
    max_par: Maximum parameter values to generate spectra for.
    """
    # Initialize Milne-Eddington object for wavelength range of penumbra formation dataset
    regions = [[np.linspace(6302.2134, 6302.6934, num=51, endpoint=True), None]]
    lines = [6302]
    ME = MilneEddington.MilneEddington(regions, lines)

    # Initialize empty arrays to store data in
    parameters = []
    spectra = []

    # Calculate values per parameter
    vals_per_par = int(num_data**(1/9))
    
    # Generate parameter grid
    B_vals = np.linspace(min_par[0], max_par[0], vals_per_par, endpoint=True)
    theta_vals = np.linspace(min_par[1], max_par[1], vals_per_par, endpoint=True)
    phi_vals = np.linspace(min_par[2], max_par[2], vals_per_par, endpoint=True)
    v_los_vals = np.linspace(min_par[3], max_par[3], vals_per_par, endpoint=True)
    v_dop_vals = np.linspace(min_par[4], max_par[4], vals_per_par, endpoint=True)
    eta_0_vals = np.linspace(min_par[5], max_par[5], vals_per_par, endpoint=True)
    a_vals = np.linspace(min_par[6], max_par[6], vals_per_par, endpoint=True)
    S_0_vals = np.linspace(min_par[7], max_par[7], vals_per_par, endpoint=True)
    S_1_vals = np.linspace(min_par[8], max_par[8], vals_per_par, endpoint=True)

    # Create a grid of all combinations
    grid = np.meshgrid(B_vals, theta_vals, phi_vals, v_los_vals, v_dop_vals, eta_0_vals, a_vals, S_0_vals, S_1_vals)

    # Flatten the grids and iterate over all combinations
    for B, theta, phi, v_los, v_dop, eta_0, a, S_0, S_1 in zip(*[g.flatten() for g in grid]):
        par_in = np.float64([B, theta, phi, v_los, v_dop, eta_0, a, S_0, S_1])

        obs_out = ME.synthesize(par_in)

        # Write parameters and associated spectra to dataset
        parameters.append(par_in)
        spectra.append(obs_out)

    # Convert datasets to numpy-arrays
    parameters, spectra = np.array(parameters), np.array(spectra)
    regions = np.array(regions[0][0])
    return parameters, spectra, regions

# Data generation using the Milne-Eddington model (Gaussian sampling around mean and standard deviation of given data)
def genData2(num_data, mean_par, std_par):
    """
    num_data: Number of datapoints to generate.
    min_par: Minimum parameter values to generate spectra for.
    max_par: Maximum parameter values to generate spectra for.
    """
    # Initialize Milne-Eddington object for wavelength range of penumbra formation dataset
    regions = [[np.linspace(6302.2134, 6302.6934, num=51, endpoint=True), None]]
    lines = [6302]
    ME = MilneEddington.MilneEddington(regions, lines)

    # Initialize empty arrays to store data in
    parameters = []
    spectra = []

    # Sample parameter sets from defined ranges [|B|, theta, varphi, v_los, v_dop, eta_0, a, S_0, S_1]
    for k in range(num_data):
        par_in = np.float64([
            np.random.normal(loc=mean_par[0], scale=2*std_par[0]),   # Randomly sample magnetic field strength within specified range in [G]
            np.random.normal(loc=mean_par[1], scale=2*std_par[1]),   # Randomly sample inclination within specified range in [rad]
            np.random.normal(loc=mean_par[2], scale=2*std_par[2]),   # Randomly sample azimuth within specified range in [rad]
            np.random.normal(loc=mean_par[3], scale=2*std_par[3]),   # Randomly sample line of sight velocity within specified range in [km/s]
            np.random.normal(loc=mean_par[4], scale=2*std_par[4]),   # Randomly sample doppler shift within specified range in [Angstrom]
            np.random.normal(loc=mean_par[5], scale=2*std_par[5]),   # Randomly sample line strength parameter within specified range
            np.random.normal(loc=mean_par[6], scale=2*std_par[6]),   # Randomly sample damping parameter within specified range
            np.random.normal(loc=mean_par[7], scale=2*std_par[7]),   # Randomly sample  within specified range
            np.random.normal(loc=mean_par[8], scale=2*std_par[8])    # Randomly sample S1 within specified range
        ])

        obs_out = ME.synthesize(par_in)

        # Write parameters and associated spectra to dataset
        parameters.append(par_in)
        spectra.append(obs_out)

    # Convert datasets to numpy-arrays
    parameters, spectra = np.array(parameters), np.array(spectra)
    regions = np.array(regions[0][0])
    return parameters, spectra, regions