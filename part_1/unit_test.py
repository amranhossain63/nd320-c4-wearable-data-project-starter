#!/usr/bin/env python
# coding: utf-8

# # Test Your Algorithm
# 
# ## Instructions
# 1. From the **Pulse Rate Algorithm** Notebook you can do one of the following:
#    - Copy over all the **Code** section to the following Code block.
#    - Download as a Python (`.py`) and copy the code to the following Code block.
# 2. In the bottom right, click the <span style="color:blue">Test Run</span> button. 
# 
# ### Didn't Pass
# If your code didn't pass the test, go back to the previous Concept or to your local setup and continue iterating on your algorithm and try to bring your training error down before testing again.
# 
# ### Pass
# If your code passes the test, complete the following! You **must** include a screenshot of your code and the Test being **Passed**. Here is what the starter filler code looks like when the test is run and should be similar. A passed test will include in the notebook a green outline plus a box with **Test passed:** and in the Results bar at the bottom the progress bar will be at 100% plus a checkmark with **All cells passed**.
# ![Example](example.png)
# 
# 1. Take a screenshot of your code passing the test, make sure it is in the format `.png`. If not a `.png` image, you will have to edit the Markdown render the image after Step 3. Here is an example of what the `passed.png` would look like 
# 2. Upload the screenshot to the same folder or directory as this jupyter notebook.
# 3. Rename the screenshot to `passed.png` and it should show up below.
# ![Passed](passed.png)
# 4. Download this jupyter notebook as a `.pdf` file. 
# 5. Continue to Part 2 of the Project. 

# In[2]:


import glob

import numpy as np
import scipy as sp
import scipy.io
import scipy.signal


def LoadTroikaDataset():
    """
    Retrieve the .mat filenames for the troika dataset.

    Review the README in ./datasets/troika/ to understand the organization of the .mat files.

    Returns:
        data_fls: Names of the .mat files that contain signal data
        ref_fls: Names of the .mat files that contain reference data
        <data_fls> and <ref_fls> are ordered correspondingly, so that ref_fls[5] is the 
            reference data for data_fls[5], etc...
    """
    data_dir = "./datasets/troika/training_data"
    data_fls = sorted(glob.glob(data_dir + "/DATA_*.mat"))
    ref_fls = sorted(glob.glob(data_dir + "/REF_*.mat"))
    return data_fls, ref_fls

def LoadTroikaDataFile(data_fl):
    """
    Loads and extracts signals from a troika data file.

    Usage:
        data_fls, ref_fls = LoadTroikaDataset()
        ppg, accx, accy, accz = LoadTroikaDataFile(data_fls[0])

    Args:
        data_fl: (str) filepath to a troika .mat file.

    Returns:
        numpy arrays for ppg, accx, accy, accz signals.
    """
    data = sp.io.loadmat(data_fl)['sig']
    return data[2:]

def AggregateErrorMetric(pr_errors, confidence_est):
    """
    Computes an aggregate error metric based on confidence estimates.

    Computes the MAE at 90% availability. 

    Args:
        pr_errors: a numpy array of errors between pulse rate estimates and corresponding 
            reference heart rates.
        confidence_est: a numpy array of confidence estimates for each pulse rate
            error.

    Returns:
        the MAE at 90% availability
    """
    # Higher confidence means a better estimate. The best 90% of the estimates
    #    are above the 10th percentile confidence.
    percentile90_confidence = np.percentile(confidence_est, 10)

    # Find the errors of the best pulse rate estimates
    best_estimtes = pr_errors[confidence_est >= percentile90_confidence]

    # Return the mean absolute error
    return np.mean(np.abs(best_estimtes))

fs = 125
min_BPM = 40
max_BPM = 240
window_length = 8 * fs
window_shift = 2 * fs 


def bandpass_filter(signal, fs):
    """filter the signal between 40 and 240 BPM

    Args:
        signal ([np_array]): input signal
        fs ([int]): Hz of input signal

    Returns:
        [np_array]: filtered signal
    """
    pass_band = (min_BPM/60, max_BPM/60)
    b, a = scipy.signal.butter(3, pass_band, btype='bandpass', fs=fs)
    return scipy.signal.filtfilt(b, a, signal)


def fourier_transform(signal, fs):
    """compute and return the one-dimensional fourier transform
    and the fourier transformed frequencies

    Args:
        signal (np_array): input signal
        fs (int): Hz of input signal

    Returns:
        fft (np_array): one-dimensional fourier transform
        freqs (np_array): fourier transformed frequencies
    """
    fft = np.abs(np.fft.rfft(signal, 2*len(signal)))
    freqs = np.fft.rfftfreq(2*len(signal), 1/fs)
    return fft, freqs


def calculate_confidence(freqs, fft_f, bpm_max):
    """calculates the confidence value for a signal window

    Args:
        freqs (np_array): list of frequenqies
        fft_f (np_array): fourier transformed signal
        bpm_max (float): max frequency

    Returns:
        confidence value (float64)
    """
    fundamental_freq_window = (
        freqs > bpm_max - min_BPM/60) & (freqs < bpm_max + min_BPM/60)
    return np.sum(fft_f[fundamental_freq_window]) / np.sum(fft_f)



def Evaluate():
    """
    Top-level function evaluation function.

    Runs the pulse rate algorithm on the Troika dataset and returns an aggregate error metric.

    Returns:
        Pulse rate error on the Troika dataset. See AggregateErrorMetric.
    """
    # Retrieve dataset files
    data_fls, ref_fls = LoadTroikaDataset()
    errs = []
    confs = []
    for data_fl, ref_fl in zip(data_fls, ref_fls):
        # Run the pulse rate algorithm on each trial in the dataset
        errors, confidence = RunPulseRateAlgorithm(data_fl, ref_fl)
        errs.append(errors)
        confs.append(confidence)
        # Compute aggregate error metric
    errs = np.hstack(errs)
    confs = np.hstack(confs)
    return AggregateErrorMetric(errs, confs)

def RunPulseRateAlgorithm(data_fl, ref_fl):
    # Load data using LoadTroikaDataFile
    
    
    ppg, accx, accy, accz = LoadTroikaDataFile(data_fl)
    
    # Compute pulse rate estimates and estimation confidence.
    
    ground_truth = sp.io.loadmat(ref_fl)['BPM0']
    
    
    ppg = bandpass_filter(ppg, fs)
    accx = bandpass_filter(accx, fs)
    accy = bandpass_filter(accy, fs)
    accz = bandpass_filter(accz, fs)
    
    bpm_pred = []
    
    confidence = []
    
    for i in range(0, len(ppg) - window_length, window_shift):
        ppg_window = ppg[i:i+window_length]

        # aggregate accelerometer data into single signal to get the acc window
        acc_window = np.sqrt(accx**2 + accy**2 + accz**2)
        acc_window = acc_window[i:i+window_length]

        # fft the ppg and acc signals
        fft_ppg, ppg_freqs = fourier_transform(ppg_window, fs)
        fft_acc, acc_freqs = fourier_transform(acc_window, fs)

        # filter the signals
        fft_ppg[ppg_freqs <= (min_BPM)/60.0] = 0.0
        fft_ppg[ppg_freqs >= (max_BPM)/60.0] = 0.0

        fft_acc[acc_freqs <= (min_BPM)/60.0] = 0.0
        fft_acc[acc_freqs >= (max_BPM)/60.0] = 0.0

        
        # get the maximum value of the ppg and acc signal
        ppg_max = ppg_freqs[np.argsort(fft_ppg, axis=0)[-1]]
        acc_max = acc_freqs[np.argsort(fft_acc, axis=0)[-1]]
        

        n = 3
        for i in range(1, n+1):
            ppg_max_tmp = ppg_freqs[np.argsort(fft_ppg, axis=0)[-i]]
            acc_max_tmp = acc_freqs[np.argsort(fft_acc, axis=0)[-i]]

            if ppg_max_tmp < ppg_max:
                ppg_max = ppg_max_tmp

            if acc_max_tmp < acc_max:
                acc_max = acc_max_tmp

        max_sig = ppg_max
        if acc_max > ppg_max:
            max_sig = acc_max

        conf_val = calculate_confidence(ppg_freqs, fft_ppg, ppg_max)
        bpm_pred.append(ppg_max*60)
        confidence.append(conf_val)
        
        
    
    

    # Return per-estimate mean absolute error and confidence as a 2-tuple of numpy arrays.
    errors = np.abs(np.diag(np.subtract(ground_truth, bpm_pred)))
    return errors, confidence


# In[ ]:




