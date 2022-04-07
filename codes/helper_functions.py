#import numpy as np
import time
from scipy.io import loadmat
from scipy.signal import butter, lfilter
from wfdb import processing
from functools import partial
from tqdm import tqdm 
from matplotlib import pyplot as plt
import numpy as np 
from models import sig2sig_unet, sig2sig_cnn
from random import uniform
from wfdb.processing import normalize_bound
import tensorflow as tf
import time
import os
#________________________________________________________
# Function to load patient ecg data and its annotations.
#________________________________________________________
def load_patient(base_path, pat_num):
    
    print('______________________________________________')
    print('Loading Data for Patient : {}'.format(pat_num))
    print('______________________________________________')

    # Loading ecg signal for given patient. 
    f_ecg = base_path + 'data/A' +  str(pat_num).zfill(2) + '.mat'
    x = loadmat(f_ecg)
    ecg = np.asarray(x['ecg'], dtype=np.float64)

    # Load R peak annotations stored as Sample number.
    f_R_ann = base_path + 'data/RPN_' +  str(pat_num).zfill(2) + '.mat'
    #f_R_ann = base_path + 'Data/RP' +  str(pat_num).zfill(2) + '.npy'
    y = loadmat(f_R_ann)
    #y = np.load(f_R_ann) 
    y = np.asarray(y['R'])
    R_ann = np.asarray(y)

    # Load S peak annotations
    f_S_ann = base_path + 'data/S_ref' +  str(pat_num).zfill(2) + '.mat'
    ref = loadmat(f_S_ann)
    S_ann = np.asarray(ref['S_ref'])


    f_V_ann = base_path + 'data/V_ref' +  str(pat_num).zfill(2) + '.mat'
    ref = loadmat(f_V_ann)
    V_ann = np.asarray(ref['V_ref'])

    print('Total Beats : ',str(len(R_ann)))
    print('S beats : ', str(len(S_ann)))
    print('V beats : ', str(len(V_ann)))

    return ecg, R_ann, S_ann, V_ann


#________________________________________________________
# Function to prepare training data.
#________________________________________________________
def extract_training_windows(ecg, R_ann, S_ann, V_ann, win_size, beat_type = 'all'):
    
    print('______________________________________________')
    print('Preparing Training Data for {} beats....'.format(beat_type))
    print('______________________________________________')

    # Total windows in ecg. (Number of training examples)
    tot_wins = int(len(ecg)/win_size)

    X_train = np.zeros((tot_wins,win_size), dtype=np.float64)
    y_train = np.zeros((tot_wins,win_size))

    # Annotations for each window. (R,S,V)
    R_w = []
    S_w = []
    V_w = []
    
    normalize = partial(processing.normalize_bound, lb=-1, ub=1)
    
    for i in tqdm(range(tot_wins)):
    
        # Start of window in whole ecg stream.
        st = i*win_size
        # End of window
        end = st + win_size

        # R peaks in the current window. 
        rIndx = np.where((R_ann >= st) & (R_ann < end))[0]

        # S peaks in the current window. 
        sIndx = np.where((S_ann >= st) & (S_ann < end))[0]

        # V peaks in the current window. 
        vIndx = np.where((V_ann >= st) & (V_ann < end))[0]

        R_w.append(R_ann[rIndx]-st)
        S_w.append(S_ann[sIndx]-st)
        V_w.append(V_ann[vIndx]-st)

        if beat_type == 'all':
                  
            for j in R_ann[rIndx]:
                r = int(j)-st
                y_train[i,r-2:r+3] = 1

        elif beat_type == 'S':
            for j in S_ann[sIndx]:
                r = int(j)-st
                y_train[i,r-2:r+3] = 1

        elif beat_type == 'V':
            for j in V_ann[vIndx]:
                r = int(j)-st
                y_train[i,r-3:r+4] = 1


        # If ecg window is non zero. Normalize it. 
        if ecg[st:end].any():
            X_train[i,:] = np.squeeze(np.apply_along_axis(normalize, 0, ecg[st:end]))
        # All zero ecg window. (Dont normalize)
        else:
            X_train[i,:] = ecg[st:end].T


    # Reshaping (n_examples,win_size,1)   
    #X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_train = np.expand_dims(X_train, axis=2)
    
    #y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1)).astype(int)
    y_train = np.expand_dims(y_train, axis=2)
    
    return X_train, y_train, R_w, S_w, V_w


#________________________________________________________
# Function to train model for each patient
#________________________________________________________
def train_for_patient(model_name, pat_num, epochs , run, train_path = '../', input_size = 8000, train_all = False):

    all_patients = np.array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]) 

    if train_all : 
        train_patients = all_patients
    else:
       
        train_patients = np.delete(all_patients, np.where(all_patients == pat_num))

    print('______________________________________________')
    print('Loading Train Data')
    print('______________________________________________')

    for i, p in enumerate(train_patients):
    
        f_X = train_path+ 'X_train_P' + str(pat_num).zfill(2) + '.npy'
        f_y = train_path+ 'y_train_P' + str(pat_num).zfill(2) + '.npy'

        if i == 0:
            X_train = np.load(f_X)
            y_train = np.load(f_y)
        else:
            X_train = np.concatenate((X_train, np.load(f_X)))
            y_train = np.concatenate((y_train, np.load(f_y)))

    # Some noisy and corrupted data examples. (Can be skipped)
    #X_train_Noise =  np.load('../data/X_Train_Noise.npy')
    #y_train_Noise =  np.load('../data/y_Train_Noise.npy')

    #X_train = np.concatenate((X_train, X_train_Noise))
    #y_train = np.concatenate((y_train, y_train_Noise))

    print('Training over {} samples'.format(len(X_train)))
    print('______________________________________________')
    print('Training model for Patient {}....'.format(pat_num))
    print('______________________________________________')
    
    start = time.process_time()

    model = sig2sig_unet(input_size)

    model_path =  '../models/'+ model_name + 'P' + str(pat_num).zfill(2) + '_r' + str(run) + '.h5'

    if not os.path.exists('../models/'):
        os.makedirs('../models/')

    model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.MeanIoU(num_classes = 2)])

    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6)

    history = model.fit(X_train,y_train,epochs = epochs, batch_size = 16,
                         callbacks=[checkpoint, callback], shuffle=True)

    print(time.process_time() - start)
 

#________________________________________________________
# Function to test model for each patient
#________________________________________________________
def test_for_patient(model_name, pat_num, epochs, run, base_path = '../', input_size = 8000, threshold = 0.5):

    stats_R = []
    stats_S = []
    stats_V = []
    win_size = 8000
    stride = 6000
    
    model_path =  '../models/'+ model_name + 'P' + str(pat_num).zfill(2) + '_r' + str(run) + '.h5'
    #model_path = base_path + 'models/UNET_Aug_forP' + str(pat_num).zfill(2) + '_r' + str(run) + '.h5'
    # Load test patient ecg and annotations
    ecg, R_ann, S_ann, V_ann = load_patient(base_path, pat_num)

    #ecg = ecg[325000:384000]

    print('______________________________________________')
    print('Predicting Test Patient....')
    print('______________________________________________')

    model = sig2sig_unet(input_size)

    model.load_weights(model_path)
    padded_indices, data_windows = extract_test_windows(ecg,win_size,stride)
    predictions = model.predict(data_windows, verbose=0)
    predictions = mean_preds(win_idx=padded_indices,
                                                     preds=predictions,
                                                     orig_len=ecg.shape[0],win_size=win_size,
                                                       stride= stride)
    filtered_peaks, filtered_proba = filter_predictions(
                                          signal=ecg,
                                          preds=predictions,
                                          threshold = threshold
                                          )


    if not os.path.exists( base_path + 'predictions/'):
        os.makedirs( base_path + 'predictions/')
    
    f = base_path + 'predictions/'+ model_name +'_R_samples_P' + str(pat_num).zfill(2) + '_r' + str(2) + '.npy'
    np.save(f,filtered_peaks)
    f = base_path + 'predictions/'+ model_name+ '_R_probs_P' + str(pat_num).zfill(2) + '_r' + str(2) + '.npy'
    np.save(f,filtered_proba)

    R_peaks_ver, _ = verifier(ecg, filtered_peaks, filtered_proba, ver_wind = 80)

    print('______________________________________________')
    print('All Beats')
    stats_R = calculate_stats(r_ref = R_ann, r_ans = R_peaks_ver, thr_ = 0.15, fs = 400)

    if (S_ann.any()):
        print('______________________________________________')
        print('S Beats')
        stats_S = calculate_stats(r_ref = S_ann, r_ans = R_peaks_ver, thr_ = 0.15, fs = 400)
    
    if (V_ann.any()):
        print('______________________________________________')
        print('V Beats')
        stats_V = calculate_stats(r_ref = V_ann, r_ans = R_peaks_ver, thr_ = 0.15, fs = 400)



    return stats_R, stats_S, stats_V




def extract_test_windows(signal,win_size,stride):
    
    normalize = partial(processing.normalize_bound, lb=-1, ub=1)

    signal = np.squeeze(signal)
    
    pad_sig = np.pad(signal,
                     (win_size - stride, win_size),
                     mode='edge')
    # Lists of data windows and corresponding indices
    data_windows = []
    win_idx = []

    # Indices for padded signal
    pad_id = np.arange(pad_sig.shape[0])


    # Split into windows and save corresponding padded indices
    for win_id in range(0, len(pad_sig), stride):
        if win_id + win_size < len(pad_sig):
            
            window = pad_sig[win_id:win_id+win_size]
            if window.any():
                window = np.squeeze(np.apply_along_axis(normalize, 0, window))

            data_windows.append(window)
            win_idx.append(pad_id[win_id:win_id+win_size])


    data_windows = np.asarray(data_windows)
    data_windows = data_windows.reshape(data_windows.shape[0],
                                        data_windows.shape[1], 1)
    win_idx = np.asarray(win_idx)
    win_idx = win_idx.reshape(win_idx.shape[0]*win_idx.shape[1])
    
    return win_idx, data_windows


#________________________________________________________
# Calculate mean of predictions during test
#________________________________________________________
def calculate_means(indices, values):
    """
    Calculate means of the values that have same index.
    Function calculates average from the values that have same
    index in the indices array.
    Parameters
    ----------
    indices : array
        Array of indices.
    values : array
        Value for every indice in the indices array.
    Returns
    -------
    mean_values : array
        Contains averages for the values that have the duplicate
        indices while rest of the values are unchanged.
    """
    assert(indices.shape == values.shape)

    # Combine indices with predictions
    comb = np.column_stack((indices, values))

    # Sort based on window indices and split when indice changes
    comb = comb[comb[:, 0].argsort()]
    split_on = np.where(np.diff(comb[:, 0]) != 0)[0]+1

    # Take mean from the values that have same index
    startTime = time.time()
    mean_values = [arr[:, 1].mean() for arr in np.split(comb, split_on)]
    executionTime = (time.time() - startTime)
    #print('Execution time in seconds: ' + str(executionTime))
    mean_values = np.array(mean_values)

    return mean_values
#________________________________________________________________________
def mean_preds(win_idx, preds, orig_len, win_size, stride):
        """
        Calculate mean of overlapping predictions.
        Function takes window indices and corresponding predictions as
        input and then calculates mean for predictions. One mean value
        is calculated for every index of the original padded signal. At
        the end padding is removed so that just the predictions for
        every sample of the original signal remain.
        Parameters
        ----------
        win_idx : array
            Array of padded signal indices before splitting.
        preds : array
            Array that contain predictions for every data window.
        orig_len : int
            Lenght of the signal that was used to extract data windows.
        Returns
        -------
        pred_mean : int
            Predictions for every point for the original signal. Average
            prediction is calculated from overlapping predictions.
        """
        # flatten predictions from different windows into one vector
        preds = preds.reshape(preds.shape[0]*preds.shape[1])
        assert(preds.shape == win_idx.shape)

        pred_mean = calculate_means(indices=win_idx, values=preds)

        # Remove paddig
        pred_mean = pred_mean[int(win_size-stride):
                              (win_size-stride)+orig_len]

        return pred_mean
#________________________________________________________________________
def filter_predictions(signal, preds, threshold):
    """
    Filter model predictions.
    Function filters model predictions by using following steps:
    1. selects only the predictions that are above the given
    probability threshold.
    2. Correct these predictions upwards with respect the given ECG
    3. Check if at least five points are corrected into the same
    location.
    4. If step 3 is true, then location is classified as an R-peak
    5. Calculate probability of location being an R-peak by taking
    mean of the probabilities from predictions in the same location.
    Aforementioned steps can be thought as an noise reducing measure as
    in original training data every R-peak was labeled with 5 points.
    Parameters
    ----------
    signal : array
        Same signal that was used with extract_windows function. It is
        used in correct_peaks function.
    preds : array
        Predictions for the sample points of the signal.
    Returns
    -------
    filtered_peaks : array
        locations of the filtered peaks.
    filtered_probs : array
        probability that filtered peak is an R-peak.
    """
    
    signal = np.squeeze(signal)
    
    assert(signal.shape == preds.shape)

    # Select points probabilities and indices that are above
    # self.threshold
    above_thresh = preds[preds > threshold]
    above_threshold_idx = np.where(preds > threshold)[0]

    # Keep only points above self.threshold and correct them upwards
    correct_up = processing.correct_peaks(sig=signal,
                                          peak_inds=above_threshold_idx,
                                          search_radius=30,
                                          smooth_window_size=30,
                                          peak_dir='up')

    filtered_peaks = []
    filtered_probs = []

    for peak_id in tqdm(np.unique(correct_up)):
        # Select indices and take probabilities from the locations
        # that contain at leas 5 points
        points_in_peak = np.where(correct_up == peak_id)[0]
        if points_in_peak.shape[0] >= 5:
            filtered_probs.append(above_thresh[points_in_peak].mean())
            filtered_peaks.append(peak_id)
    
    print(len(filtered_peaks))
    filtered_peaks = np.asarray(filtered_peaks)
    filtered_probs = np.asarray(filtered_probs)

    return filtered_peaks, filtered_probs
#________________________________________________________________________
def calculate_stats(r_ref, r_ans, thr_ , fs):
    # Threshold region to consider correct detection. in samples
    #thr_ = 0.15 #(150/4 ms)

    print('______________________________________________')
    print('_________Calculating Stats____________________')
    print('______________________________________________')
    FP_index_array = []
    FN_index_array = []

    FP = 0
    TP = 0
    FN = 0
    for j in range(len(r_ref)):
        loc = np.where(np.abs(r_ans - r_ref[j]) <= thr_*fs)[0]
        if j == 0:
            err = np.where((r_ans >= 0.5*fs + thr_*fs) & (r_ans <= r_ref[j] - thr_*fs))[0]
        elif j == len(r_ref)-1:
            err = np.where((r_ans >= r_ref[j]+thr_*fs) & (r_ans <= 9.5*fs - thr_*fs))[0]
        else:
            err = np.where((r_ans >= r_ref[j]+thr_*fs) & (r_ans <= r_ref[j+1]-thr_*fs))[0]
    
        FP = FP + len(err)
        
        if err.any():
            #print(err)
            #print(len(err))
            for er in err:
                #print(r_ans[er])
                FP_index_array.append(r_ans[er])
            
        if len(loc) >= 1:
            TP += 1
            FP = FP + len(loc) - 1
        elif len(loc) == 0:
            FN += 1
            FN_index_array.append(r_ref[j])
            

            
    all_FP = FP
    all_FN = FN
    all_TP = TP

    Recall = float(str(round((all_TP / (all_FN + all_TP))*100,2)))
    Precision = float(str(round((all_TP / (all_FP + all_TP))*100,2)))

    if Recall + Precision == 0:
        F1_score = 0
    else:
        F1_score = float(str(round((2 * Recall * Precision / (Recall + Precision)),2)))
    print("TP's:{} FN's:{} FP's:{}".format(all_TP,all_FN,all_FP))
    print("Recall:{}, Precision(FNR):{}, F1-Score:{}".format(Recall,Precision,F1_score))
    print("Total {}".format(len(r_ref)))
    
    return [len(r_ref), TP, FN, FP, Recall, Precision, F1_score, FN_index_array, FP_index_array]
#________________________________________________________________________
def get_noise(ma, bw, win_size):
    """
    Create noise that is typical in ambulatory ECG recordings.
    Creates win_size of noise by using muscle artifact, baseline
    wander, and mains interefence (60 Hz sine wave) noise. Windows from
    both ma and bw are randomly selected to
    maximize different noise combinations. Selected noise windows from
    all of the sources are multiplied by different random numbers to
    give variation to noise strengths. Mains interefence is always added
    to signal, while addition of other two noise sources varies.
    Parameters
    ----------
    ma : array
        Muscle artifact signal
    bw : array
        Baseline wander signal
    win_size : int
        Wanted noise length
    Returns
    -------
    noise : array
        Noise signal of given window size
    """
    # Get the slice of data
    beg = np.random.randint(ma.shape[0]-win_size)
    end = beg + win_size
    beg2 = np.random.randint(ma.shape[0]-win_size)
    end2 = beg2 + win_size

    # Get mains_frequency US 60 Hz (alter strenght by multiplying)
    #mains = create_sine(400, int(win_size/400), 60)*uniform(0, 0.05)

    # Choose what noise to add
    mode = np.random.randint(3)

    # Add noise with different strengths
    ma_multip = uniform(0, 2)
    bw_multip = uniform(0, 4)

    # Add noise
    if mode == 0:
        noise = ma[beg:end]*ma_multip
    elif mode == 1:
        noise = bw[beg:end]*bw_multip
    else:
        noise = (ma[beg:end]*ma_multip)+(bw[beg2:end2]*bw_multip)

    return noise
#________________________________________________________________________
def create_sine(sampling_frequency, time_s, sine_frequency):
    """
    Create sine wave.
    Function creates sine wave of wanted frequency and duration on a
    given sampling frequency.
    Parameters
    ----------
    sampling_frequency : float
        Sampling frequency used to sample the sine wave
    time_s : float
        Lenght of sine wave in seconds
    sine_frequency : float
        Frequency of sine wave
    Returns
    -------
    sine : array
        Sine wave
    """
    samples = np.arange(time_s * sampling_frequency) / sampling_frequency
    sine = np.sin(2 * np.pi * sine_frequency * samples)

    return sine

#________________________________________________________________________
def verifier(ecg, R_peaks, R_probs, ver_wind = 60):
    
    del_indx = []
    wind = 30
    check_ind = np.squeeze(np.array(np.where(R_probs < 0.95)))

    if R_peaks[check_ind[-1]] == R_peaks[-1]:
        check_ind = check_ind[:-1]

    for ind in check_ind:

        two_ind = [ind, ind+1]


        diff = R_peaks[two_ind[1]] - R_peaks[two_ind[0]]

        # 60 for chinese data 
        if diff < ver_wind:

            try:

                two_probs = [ R_probs[two_ind[0]] , R_probs[two_ind[1]] ] 
                two_peaks = [ R_peaks[two_ind[0]] , R_peaks[two_ind[1]] ] 

                beat1 = ecg[ two_peaks[0] - wind : two_peaks[0] + wind ] 
                beat2 = ecg[ two_peaks[1] - wind : two_peaks[1] + wind ]

                for i in range(two_ind[0]-1,two_ind[0]-1-30,-1):
                    for thr_p in [0.6,0.5,0.4,0.3,0.2,0.1]:
                        if(R_probs[i] > thr_p):
                            #print(i)
                            prv_beat = ecg[ R_peaks[i] - wind : R_peaks[i] + wind ]
                            break

                for i in range(two_ind[1]+1,two_ind[1]+1+30):
                    
                    for thr_p in [0.6,0.5,0.4,0.3,0.2,0.1]:
                        if i == len(R_probs):
                            nxt_beat = ecg[ R_peaks[i-1] - wind : R_peaks[i-1] + wind]
                            break
                        
                        
                        if(R_probs[i] > thr_p):
                            #print(i)
                            nxt_beat = ecg[ R_peaks[i] - wind : R_peaks[i] + wind]
                            break

                    else:
                        # Continue if the inner loop wasn't broken.
                        continue
                    # Inner loop was broken, break the outer.
                    break

                if len(nxt_beat) != 60:
                    nxt_beat = prv_beat
                if len(prv_beat) != 60:
                    prv_beat = nxt_beat
                if len(beat1) != 60:
                    beat1 = beat2
                if len(beat2) != 60:
                    beat2 = beat1

                X1 = np.corrcoef(np.squeeze(beat1),np.squeeze(prv_beat))[0,1]
                X2 = np.corrcoef(np.squeeze(beat1),np.squeeze(nxt_beat))[0,1]

                Y1 = np.corrcoef(np.squeeze(beat2),np.squeeze(prv_beat))[0,1]
                Y2 = np.corrcoef(np.squeeze(beat2),np.squeeze(nxt_beat))[0,1]

                si = np.argmin([X1*X2, Y1*Y2])
                del_indx.append(two_ind[si])
            except:
                pass
    
    R_peaks_ver = np.delete(R_peaks, del_indx)
    
    R_probs_ver = np.delete(R_probs, del_indx)
    
    return R_peaks_ver, R_probs_ver





    
  