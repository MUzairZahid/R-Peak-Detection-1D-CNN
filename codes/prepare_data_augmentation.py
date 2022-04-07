from helper_functions import load_patient, extract_training_windows, get_noise,normalize_bound
import numpy as np
import os


#_______________________________________________
# Initializations
#_______________________________________________

# Window/Segment length 
l = 20 #seconds
# window stride for testing. 
s = 10 #seconds

# Base folder path
base_path = '../'

training_data_path = base_path + 'training_data_aug/'

#_______________________________________________
# Dont change These Values
#_______________________________________________
# Sapmling frequency of ecg signal is 400 hz. (CPSC2020 Data)
fs = 400
# Window/Segment length in samples. 
win_size = l*fs
# Stride for test window in samples. 
stride = s*fs
#_______________________________________________

if not os.path.exists(training_data_path):
    os.makedirs(training_data_path)



ma = np.load(base_path +'data/ma.npy')
bw = np.load(base_path +'data/bw.npy')


aug_factor = 1 # How many times to augment.



for pat_num in range(1,11):


    #_______________________________________________
    # Training Data Preparation
    #_______________________________________________
    # Load 1 patient ecg and annotations
    ecg, R_ann, S_ann, V_ann = load_patient(base_path, pat_num)
    X_train, y_train, R_w, S_w, V_w = extract_training_windows(ecg, R_ann, S_ann, V_ann, win_size)

    print('Total Windows : ', len(X_train))

    # Indexes of windows where S beats are present.
    s_w_idx = []
    for idx,ann in enumerate(S_w):    
        if ann.any():
            s_w_idx.append(idx)
    s_w_idx = np.asarray(s_w_idx, dtype=np.int32)


    # Indexes of windows where V beats are present.
    v_w_idx = []
    for idx,ann in enumerate(V_w):    
        if ann.any():
            v_w_idx.append(idx)
    v_w_idx = np.asarray(v_w_idx, dtype=np.int32)

    # Indexes of windows for V beats and S beats combined.
    sv_idx = np.unique(np.concatenate((s_w_idx,v_w_idx)))
    # All indexes of training windows.
    idx = np.arange(len(X_train))
    # indexes other than S and V beats. (Normal R peaks)
    rem_idx = np.delete(idx, sv_idx, 0)
    
    #_____________________________________________________________________
    # Choose 50% of remaining. 
    #norm_idx = np.random.choice(rem_idx, size=int(len(rem_idx)*(1/2)), replace=False)
    
    norm_idx = rem_idx[::2]
    
    # Placeholder for augmented beats
    X_aug = np.zeros((int(len(sv_idx)*aug_factor), win_size))
    y_aug = np.zeros((int(len(sv_idx)*aug_factor), win_size))
    
    count = 0
    for i in tqdm(range(len(sv_idx))):

        current_window = X_train[sv_idx[i]]
        current_window = np.squeeze(current_window)

        for j in range(aug_factor):

            noise = get_noise(ma, bw, win_size)
            aug_window = current_window + noise

            X_aug[count] = normalize_bound(aug_window, lb=-1, ub=1)

            y_aug[count] = np.squeeze(y_train[sv_idx[i]])

            count += 1
            
    X_aug = np.expand_dims(X_aug, axis=2)
    y_aug = np.expand_dims(y_aug, axis=2)
    
    
    #X_train = np.concatenate((X_train,X_aug))
    #y_train = np.concatenate((y_train,y_aug))
    
    X_train = np.concatenate((X_train[norm_idx],X_train[sv_idx], X_aug))
    y_train = np.concatenate((y_train[norm_idx],y_train[sv_idx], y_aug))


    assert len(X_train) == len(y_train)

    print('Normal Beat Windows : ', len(rem_idx))
    print('Abnormal Beat Windows : ', len(sv_idx))
    print('Augmented abnormal Beat Windows : ', len(X_aug))

    print('Saving Data')
    f_X = training_data_path+ 'X_train_P' + str(pat_num).zfill(2) + '.npy'
    f_y = training_data_path+ 'y_train_P' + str(pat_num).zfill(2) + '.npy'
    np.save(f_X, X_train)
    np.save(f_y, y_train)
    print('Done..')