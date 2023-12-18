import numpy as np
import pandas as pd
from copy import deepcopy
# import utils

from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
import matplotlib.pyplot as plt

line_parameters = {}

def NCDFs(nprof):
    '''
    Transforms spectra into normalized cumulative distribution functions
    independent of line type (sensitive to noise?)
    Input: 
    ------
    nprof --> numpy array; (i, wavelength)
    
    Output:
    -------
    ncdfs --> numpy array; (i, wavelength) of normalized cumulative distribution functions
    '''
    nprof_adj = deepcopy(nprof)
    cdfs = np.cumsum(nprof_adj, axis=1)
    ncdfs = cdfs/np.max(cdfs, axis=1).reshape(cdfs.shape[0], 1)
    return ncdfs

def quartiles(ncdfs):
    '''
    Calculates the position of the quartiles for an numpy array
    independent of line type (sensitive to noise?)
    Input:
    ------
    ncdfs --> numpy array; (i, wavelength) of normalized cumulative distribution functions
    Output:
    -------
    qs --> numpy array; (i, 3 quartile values)
    '''
    q1 = np.abs(ncdfs - .25).argmin(axis=1).reshape(ncdfs.shape[0],1)
    q2 = np.abs(ncdfs - .50).argmin(axis=1).reshape(ncdfs.shape[0],1)
    q3 = np.abs(ncdfs - .75).argmin(axis=1).reshape(ncdfs.shape[0],1)
    qs = np.concatenate((q1,q2,q3), axis=1)
    return qs


#### Mg II k&h line ####################################################################################################

#---------------Line parameters------------------

n_bins = 240 
lambda_min = 2794
lambda_max = 2806
window = 60
core_1 = 2796.34
core_2 = 2803.52
xax = np.linspace( lambda_min, lambda_max, n_bins )
k = int(np.argmin(abs(xax-core_1)) + 1) # the + 1 seems to align the centre with k-core
h = int(np.argmin(abs(xax-core_2)))
kl = int(k - window/2)
kr = int(k + window/2)
hl =  int(h - window/2)
hr = int(h + window/2)

line_parameters['Mg II k'] = {'n_bins' : n_bins, 
                              'lambda_min' : lambda_min,
                              'lambda_max' : lambda_max,
                              'xax' : xax,
                              'window' : window,
                              'core_1' : core_1,
                              'core_2' : core_2,
                              'k' : k, # the + 1 seems to align the centre with k-core
                              'h' : h,
                              'kl' : kl,
                              'kr' : kr,
                              'hl' : hl,
                              'hr' : hr,
                              'k2l' : np.abs( (core_1-1) - xax ).argmin(), 
                              'k2r' : np.abs( (core_1+1) - xax ).argmin(),
                              'sm' : np.abs( 2798.77  - xax ).argmin(),
                              'wing' : np.abs( 2799.32 - xax ).argmin()
                             }


#---------------Functions-------------------------

def peak_locs_MgIIk(prof):
    '''
    Find k2v, k2r, and k3 locations and produces three features
    Input:
    ------
    prof --> numpy array; single spectrum (wavelength,) 
    Output:
    -------
    k3_h --> hight of the central dip of a normalized profile
    peak_ratios --> ratios of the k2v and k2r peaks
                    peak_ratios > 1 --> k2v higher than k2r
                    peak_ratios < 1 --> k2r higher than k2v
                    peak_ratios = 1 --> k2r same as k2v
    '''
    lambda_min = line_parameters['Mg II k']['lambda_min']
    lambda_max = line_parameters['Mg II k']['lambda_max']
    n_bins = line_parameters['Mg II k']['n_bins']
    
    k = line_parameters['Mg II k']['k']
    kl = line_parameters['Mg II k']['kl']
    kr = line_parameters['Mg II k']['kr']

    k2l = line_parameters['Mg II k']['k2l']
    k2r = line_parameters['Mg II k']['k2r']

    h = line_parameters['Mg II k']['h']
    hl = line_parameters['Mg II k']['hl']
    hr = line_parameters['Mg II k']['hr']

    sm = line_parameters['Mg II k']['sm']
    wing = line_parameters['Mg II k']['wing']

    # location in data units of k2v
    p1 = k2l + (prof[k2l:k2r]).argmax()
    if p1 < k:
        grads = np.gradient(prof[p1:k2r])
        xs = np.arange(p1,k2r)
        try: # try is for single peaks, grad keeps going down so this sets them ontop of eachother
            p2 = np.where(np.diff(np.sign(grads)))[0][1] + p1
            k3 = np.where(np.diff(np.sign(grads)))[0][0] + p1
        except:
            p2 = p1
            k3 = p2
    if p1 > k:        
        grads = np.gradient(prof[k2l:p1])
        grads = -np.flip(grads)
        xs = np.arange(k2l,p1)
        try: # try is for single peaks, grad keeps going down so this sets them ontop of eachother
            p2 = p1 - np.where(np.diff(np.sign(grads)))[0][1] -2
            k3 = p1 - np.where(np.diff(np.sign(grads)))[0][0] -2
        except:
            p2 = p1
            k3 = p2
    # for single peaks, stops p2 from sliding off the edge because there is no second change in the derivative
    if p1 == k:
        xs = k
        grads = 0
        p2 = p1
        k3 = p2
    # for single peaks, stops p2 from sliding off the edge because there is no second change in the derivative
    if prof[k3] < .15*np.max(prof[kl:kr]):
        xs = k
        grads = 0
        p2 = p1
        k3 = p2
    k2vio = np.min( [p1, p2] )
    k2red = np.max( [p1, p2] )
    k2vio_h = prof[k2vio]
    k2red_h = prof[k2red]
    k3_h = prof[k3]
    peak_ratios = k2vio_h/k2red_h
    peak_separation = abs( k2red - k2vio )
    peak_separation = (lambda_max-lambda_min)/(n_bins-1)*peak_separation
#     plt.plot(prof)
#     plt.plot(xs,grads)
#     plt.scatter(p1,prof[p1])
#     plt.scatter(k,prof[k])
#     plt.scatter(p2,prof[p2])
#     plt.scatter(k3,prof[k3], c='blue')
#     plt.axvline(k3, c='k', linestyle='dotted')
#     plt.scatter(p1,prof[p1])
#     plt.scatter(k2l,prof[k2l])
#     plt.scatter(k2r,prof[k2r])
#     plt.scatter(k2vio,prof[k2vio], c='blue')
#     plt.axvline(k2vio, c='k', linestyle='dotted')
#     plt.scatter(k2red,prof[k2red], c='blue')
#     plt.axvline(k2red, c='k', linestyle='dotted')
#     plt.axhline(0, c='k', linestyle='dotted')
#     plt.xlim([kl,kr])
#     plt.show()

    return k3_h, peak_ratios, peak_separation#, k2vio, k2red, k3

def doppler_weight_MgIIk(nprof, qrt):
    '''
    
    Input:
    ------
    nprof --> numpy array; (i, wavelength)
    Output:
    -------
    doppler --> differance bwtween integrated left and right of k-core
                doppler > 0 --> redshift (downflow)
                doppler < 0 --> blue shift (upflow)
                doppler = 0 --> symmetric, atleased to first moment
    '''
    
    lambda_min = line_parameters['Mg II k']['lambda_min']
    lambda_max = line_parameters['Mg II k']['lambda_max']
    n_bins = line_parameters['Mg II k']['n_bins']

    lambda_units = np.linspace( lambda_min, lambda_max, num=n_bins )
    
#     K = []
#     for prof in nprof:
#         k3 = peak_locs_MgIIk(prof)
#         K.append(lambda_units[k3])
#     K = np.vstack(K)    
#     doppler = (K - line_parameters['Mg II k']['core_1'])/line_parameters['Mg II k']['core_1']*3E+5
    
    
    doppler = -(lambda_max-lambda_min)/(n_bins-1)*((int((kr-kl)/2)-qrt[:,1]))/line_parameters['Mg II k']['core_1']*3E+5
    
    return doppler # in km/s

def peak_stats_MgIIk(nprof):
    '''
    Use the peak locations to calculate k3_h, peak_ratios, peak_separation over an entire aray
    Input:
    ------
    nprof --> numpy array; (i, wavelength)
    Output:
    -------
    k3_h, peak_ratios, peak_separation as above but for an entire array
    '''
    
    k3_h_list = []
    peak_ratios_list= []
    peak_separation_list = []
#     k2vio_list = []
#     k2red_list = []
#     k3_list = []
    # , k2vio, k2red, k3
    for prof in nprof:
        k3_h, peak_ratios, peak_separation = peak_locs_MgIIk(prof)
        k3_h_list.append(k3_h)
        peak_ratios_list.append(peak_ratios)
        peak_separation_list.append(peak_separation)
#         k2vio_list.append(k2vio)
#         k2red_list.append(k2red)
#         k3_list.append(k3)
        
    k3_h = np.asarray(k3_h_list)
    peak_ratios = np.asarray(peak_ratios_list)
    peak_separation = np.asarray(peak_separation_list)
#     k2vio = np.asarray(k2vio_list)
#     k2red = np.asarray(k2red_list)
#     k3 = np.asarray(k3_list)
    
    return k3_h, peak_ratios, peak_separation#, k2vio, k2red, k3

def extract_features_MgIIk(nprof, save_path='/data1/userspace/bpanos/XAI/data/df_features.csv'):
    '''
    Mian feature extraction function. Extracts 10 features
    Input:
    ------
    nprof --> numpy array; (i, wavelength)
    save_path --> location to save the features in a .csv file
    Output:
    -------
    df_features --> Pandas DataFrame; Features include:
                    center_mass, line_width, line_asymmetry, doppler, trip_emiss,
                    kh_ratio, k3_h, peak_ratios, peak_separation, total_continuum
    '''
    lambda_min = line_parameters['Mg II k']['lambda_min']
    lambda_max = line_parameters['Mg II k']['lambda_max']
    n_bins = line_parameters['Mg II k']['n_bins']
    
    lambda_units = np.linspace( lambda_min, lambda_max, num=n_bins )
    
    k = line_parameters['Mg II k']['k']
    kl = line_parameters['Mg II k']['kl']
    kr = line_parameters['Mg II k']['kr']
    
    h = line_parameters['Mg II k']['h']
    hl = line_parameters['Mg II k']['hl']
    hr = line_parameters['Mg II k']['hr']
    
    sm = line_parameters['Mg II k']['sm']
    wing = line_parameters['Mg II k']['wing']
    
    #-------------quartiles---------------
    #(Only for the k-core)
    ncdfs = NCDFs(nprof[:,kl:kr])
    qrt = quartiles(ncdfs)
    center_mass = -(int((kr-kl)/2)-qrt[:,1])
    center_mass = lambda_units[center_mass+k]
    line_width = (lambda_max-lambda_min)/(n_bins-1)*(qrt[:,2] - qrt[:,0])
    line_asymmetry = (lambda_max-lambda_min)/(n_bins-1)*( (qrt[:,2]-qrt[:,1])-(qrt[:,1]-qrt[:,0]) ) 
    #------------doppler asymmetry---------
    #(Only for the k-core)
    doppler = doppler_weight_MgIIk(nprof, qrt)
    #------------triplet emission---------
    #(taken with respect to the continiumm according to Pereira, T. M. D. et al. 2015, ApJ, 806, 14)
    trip_emiss = np.log(nprof[:, sm]/nprof[:,wing])
    #---------------k/h ratio------------
    #(Larger values are opticaly thinner)
    kh_ratio = np.sum( nprof[:, kl:kr],axis=1)/np.sum(nprof[:, hl:hr], axis=1)
    #------------k2 peak ratios----------
    #(peak_ratio > 1 if 2kv > 2kr, peak_ratio < 1 if 2kv < 2kr)
    k3_h, peak_ratios, peak_separation = peak_stats_MgIIk(nprof)
    #, k2vio, k2red, k3
    #-----------total continuum-----------
    total_continuum = np.sum( nprof[:, wing:hl],axis=1)
    intensity = np.max( nprof, axis=1 )
    #Turn feature into a pandas DataFrame and save to a CSV file
    feature_names = ['center_mass', 'line_width', 'line_asymmetry', 'doppler', 'trip_emiss', 'kh_ratio', 'k3_h', 'peak_ratios', 'peak_separation', 'total_continuum']
#     ['center_mass', 'line_width', 'line_asymmetry', 'doppler', 'k3_h', 'peak_ratios', 'peak_separation', 'intensity']
    feature_list  = [center_mass, line_width, line_asymmetry, doppler, trip_emiss, kh_ratio, k3_h, peak_ratios, peak_separation, total_continuum]
    feature_mat = np.vstack(feature_list).T
    feature_mat = np.round(feature_mat,3)
    df_features = pd.DataFrame(feature_mat, columns=feature_names)
    if save_path:
        df_features.to_csv(save_path, sep='\t')
    return df_features#, k2vio, k2red, k3



