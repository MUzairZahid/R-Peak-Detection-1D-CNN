from helper_functions import train_for_patient, test_for_patient
import os
import numpy as np
import pandas as pd

# Epochs
epochs = 30
# Base folder path
base_path = '../'
training_data_path = base_path + 'training_data_aug/'
model_name = 'sig2sig_unet' # or sig2sig_cnn

all_pat = [1,2,3,4,5,6,7,8,9,10]

results_all = np.zeros((10,5), dtype = np.int32)
perc_all = np.zeros((10,3), dtype = np.float32)
results_S = np.zeros((7,4), dtype = np.int32)
results_V = np.zeros((8,4), dtype = np.int32)
s_count = 0
v_count = 0

for run in [1]:
    
    print('______________________________________________')
    print('Run : ', str(run))
    print('______________________________________________')
    
    for pat_num in all_pat:

        train_for_patient(model_name, pat_num, epochs = epochs , run = run, input_size = 8000, train_path = training_data_path)
        stats_R, stats_S, stats_V = test_for_patient(model_name, pat_num, epochs = epochs , 
                                                        run = run, threshold = 0.1,input_size = 8000)

        #_______________________________________________
        # Saving stats
        #_______________________________________________
        if stats_S != []:
            results_S[s_count][0] = pat_num
            results_S[s_count][1:] = stats_S[:3]

            df_S = pd.DataFrame(results_S)
            df_S.columns = ['Patient No', 'Total Beats', 'Detected', 'Missed']
            f = base_path + 'Results/'+ model_name +'_S_r' + str(run) + '.csv'
            df_S.to_csv (r'{}'.format(f), index = False, header=True)

            s_count += 1
        
        if stats_V != []:
            results_V[v_count][0] = pat_num
            results_V[v_count][1:] = stats_V[:3]

            df_V = pd.DataFrame(results_V)
            df_V.columns = ['Patient No', 'Total Beats', 'Detected', 'Missed']
            f = base_path + 'Results/'+ model_name +'_V_r' + str(run) + '.csv'
            df_V.to_csv (r'{}'.format(f), index = False, header=True)

            v_count += 1
        
        results_all[pat_num-1][0] = pat_num
        results_all[pat_num-1][1:] = stats_R[:4]
        perc_all[pat_num-1] = stats_R[4:7]



        df_all = pd.DataFrame(results_all)
        df_all = pd.concat([df_all, pd.DataFrame(perc_all, dtype = np.float32)], axis=1)
        df_all.columns = ['Patient No', 'Total Beats', 'TP', 'FN', 'FP', 'Recall', 'Precision', 'F1']

        if not os.path.exists(base_path + 'results/'):
            os.makedirs(base_path + 'results/')

        f = base_path + 'results/'+ model_name +'_all_r' + str(run) + '.csv'

        df_all.to_csv (r'{}'.format(f), index = False, header=True)
