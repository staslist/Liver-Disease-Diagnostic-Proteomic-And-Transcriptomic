# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 05:55:33 2021

@author: Stanislav
"""

'''Ch3 of Stanislav's PhD thesis. Proteomic only models. PBMC 3-Way and LV 3-Way.'''

from SL_Thesis_ML_Codebase import *

PBMC_3Way_AH = ['...']
# Note that ... was removed after we discovered that they were invalid. 

PBMC_3Way_CT = ['...']

PBMC_3Way_AC = ['...']

PBMC_3Way = PBMC_3Way_AH + PBMC_3Way_CT + PBMC_3Way_AC
PBMC_3Way_Dict = {'q1': PBMC_3Way_AH, 'q2':PBMC_3Way_CT, 'q3':PBMC_3Way_AC}

# This is the unmatched list of AH PBMC samples for the ensembl setup. Moved nine AH samples from matched to unmatched for this.
PBMC_3Way_AH_Unmatched_Balanced = ['...']
# Note that ... was removed after we discovered that they were invalid. 

# Note, I added 7 matched CT samples into the "unmatched" set for purposes of classification.
PBMC_3Way_CT_Unmatched_Balanced = ['...']

# This is the unmatched list of AC PBMC samples for the ensembl setup. Moved seven AC samples from matched to unmatched for this.
PBMC_3Way_AC_Unmatched_Balanced = ['...']

PBMC_3Way_Unmatched_Balanced = PBMC_3Way_AH_Unmatched_Balanced + PBMC_3Way_CT_Unmatched_Balanced + PBMC_3Way_AC_Unmatched_Balanced
PBMC_3Way_Unmatched_Balanced_Dict = {'q1':PBMC_3Way_AH_Unmatched_Balanced, 'q2':PBMC_3Way_CT_Unmatched_Balanced,
                                     'q3':PBMC_3Way_AC_Unmatched_Balanced}

PBMC_3Way_AH_Matched = ['...']

PBMC_3Way_AH_Matched_Balanced = ['...']

PBMC_3Way_CT_Matched = ['...']

PBMC_3Way_CT_Matched_Balanced = ['...']

PBMC_3Way_AC_Matched = ['...']

PBMC_3Way_AC_Matched_Balanced = ['...']

PBMC_3Way_Matched = PBMC_3Way_AH_Matched + PBMC_3Way_CT_Matched + PBMC_3Way_AC_Matched
PBMC_3Way_Matched_Dict = {'q1': PBMC_3Way_AH_Matched, 'q2':PBMC_3Way_CT_Matched, 'q3':PBMC_3Way_AC_Matched}

PBMC_3Way_Matched_Balanced = PBMC_3Way_AH_Matched_Balanced + PBMC_3Way_CT_Matched_Balanced + PBMC_3Way_AC_Matched_Balanced
PBMC_3Way_Matched_Balanced_Dict = {'q1': PBMC_3Way_AH_Matched_Balanced, 'q2':PBMC_3Way_CT_Matched_Balanced,
                                   'q3':PBMC_3Way_AC_Matched_Balanced}

PBMC_3Way_Proteomic_to_RNAseq = {'...'}


PBMC_3Way_Proteomic_to_RNAseq_New = {'...'}

LV_3Way_AH_Old = ['...']
LV_3Way_AH = ['...']
# Removed '...' due to it being invalid.

LV_3Way_CT = ['...']
LV_3Way_AC = ['...']
LV_3Way = LV_3Way_AH + LV_3Way_CT + LV_3Way_AC
LV_2Way = LV_3Way_AH + LV_3Way_CT
LV_3Way_Old = LV_3Way_AH_Old + LV_3Way_CT + LV_3Way_AC
LV_3Way_Dict = {'q1': LV_3Way_AH, 'q2':LV_3Way_CT, 'q3':LV_3Way_AC}
LV_2Way_Dict = {'q1': LV_3Way_AH, 'q2':LV_3Way_CT}
LV_3Way_Old_Dict = {'q1': LV_3Way_AH_Old, 'q2':LV_3Way_CT, 'q3':LV_3Way_AC}


LV_3Way_Matched_AH = ['...']

LV_3Way_Matched_AH_Balanced = ['...']

LV_3Way_Unmatched_AH = ['...']
# Removed '...' due to it being invalid.

# Added 5 matched AH samples to "unmatched" set for classification sake.
LV_3Way_Unmatched_AH_Balanced = ['...']
# Removed '...' due to it being invalid.

LV_3Way_Matched_CT = ['...']

LV_3Way_Matched_CT_Balanced = LV_3Way_Matched_CT

LV_3Way_Unmatched_CT = ['...']

LV_3Way_Unmatched_CT_Balanced = LV_3Way_Unmatched_CT

LV_3Way_Matched_AC = ['...']

LV_3Way_Matched_AC_Balanced = ['...']

LV_3Way_Unmatched_AC = ['...']

# Added 2 matched AC samples to "unmatched" set for classification sake.
LV_3Way_Unmatched_AC_Balanced = ['...']

LV_3Way_Matched = LV_3Way_Matched_AH + LV_3Way_Matched_CT + LV_3Way_Matched_AC
LV_3Way_Matched_Balanced = LV_3Way_Matched_AH_Balanced + LV_3Way_Matched_CT_Balanced + LV_3Way_Matched_AC_Balanced
LV_3Way_Unmatched = LV_3Way_Unmatched_AH + LV_3Way_Unmatched_CT + LV_3Way_Unmatched_AC
LV_3Way_Unmatched_Balanced = LV_3Way_Unmatched_AH_Balanced + LV_3Way_Unmatched_CT_Balanced + LV_3Way_Unmatched_AC_Balanced
LV_3Way_Matched_Dict = {'q1': LV_3Way_Matched_AH, 'q2':LV_3Way_Matched_CT, 'q3':LV_3Way_Matched_AC}
LV_3Way_Matched_Dict_Balanced = {'q1': LV_3Way_Matched_AH_Balanced, 'q2':LV_3Way_Matched_CT_Balanced, 'q3':LV_3Way_Matched_AC_Balanced}
LV_3Way_Unmatched_Dict = {'q1': LV_3Way_Unmatched_AH, 'q2':LV_3Way_Unmatched_CT, 'q3':LV_3Way_Unmatched_AC}
LV_3Way_Unmatched_Dict_Balanced = {'q1': LV_3Way_Unmatched_AH_Balanced, 'q2':LV_3Way_Unmatched_CT_Balanced, 'q3':LV_3Way_Unmatched_AC_Balanced}

LV_3Way_Proteomic_to_RNAseq = {'...'}

LV_3Way_Proteomic_to_RNAseq_New = {'...'}

validation = ['...']
validation_rev = ['...']
datasets = {'PBMC_3Way': PBMC_3Way, 'PBMC_3Way_Matched': PBMC_3Way_Matched,
            'PBMC_3Way_Matched_Balanced': PBMC_3Way_Matched_Balanced, 
            'PBMC_3Way_Unmatched_Balanced': PBMC_3Way_Unmatched_Balanced, 'LV_3Way_Old': LV_3Way_Old,
            'LV_3Way': LV_3Way, 'LV_3Way_Matched': LV_3Way_Matched, 'LV_3Way_Matched_Balanced': LV_3Way_Matched_Balanced,
            'LV_2Way': LV_2Way, 'LV_3Way_Unmatched': LV_3Way_Unmatched,
            'LV_3Way_Unmatched_Balanced': LV_3Way_Unmatched_Balanced,'Validation': validation}

# Tiny bit of code to mute warnings.
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def filter_inferno_by_protein_list(protein_list_file:str, inferno_file:str, out_dir:str,
                                   out_fname = 'inferno_filtered.csv'):
    to_write = []
    proteins = read_in_csv_file_one_column(protein_list_file, 0, ',')
    with open(inferno_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        i = 0
        for row in csv_reader:
            if(i == 0):
                to_write.append(row)
            if(i > 0):
                protein_name = row[0]
                try:
                    index = protein_name.index('HUMAN')
                except ValueError:
                    continue
                # Remove number at the end
                protein_name = protein_name[0:(index+5)]
                if(protein_name in proteins):
                    to_write.append(row)
            i += 1
            
    filename = out_dir + '/' + out_fname
    with open(filename, 'w') as writer:
        for line in to_write:
            for ele in line:
                writer.write(str(ele) + ',')
            writer.write('\n')

def generate_proteomic_sample_files_from_cuffdiff(root_dir:str, out_dir:str, tissue:str):
    
    assert tissue in ['LV', 'PB']
    RNAseq_to_Proteomic = {}
    sample_by_cond = {}
    if(tissue == 'LV'):
        RNAseq_to_Proteomic = {v: k for k, v in LV_3Way_Proteomic_to_RNAseq_New.items()}
        sample_by_cond = LV_3Way_Matched_Dict
    elif(tissue == 'PB'):
        RNAseq_to_Proteomic = {v: k for k, v in PB_3Way_Proteomic_to_RNAseq_New.items()}
        sample_by_cond = PBMC_3Way_Matched_Dict
    
    samples_all = []
    with open(root_dir + 'Cuffdiff_GEOM_POOL/read_groups.info') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        i = 0
        for row in csv_reader:
            if(i > 0):
                full_sample_name = row[0]
                sample_name_array = full_sample_name.split('.')
                sample_name_RNAseq = sample_name_array[0]
                sample_name_proteomic = RNAseq_to_Proteomic[sample_name_RNAseq]
                samples_all.append(sample_name_proteomic)
            i += 1
    
    with open(out_dir + 'Samples_ALL.csv', 'w') as writer:
        for sample in samples_all:
            writer.write(str(sample))
            if(sample in sample_by_cond['q1']):
                writer.write(',AH,q1')
            elif(sample in sample_by_cond['q2']):
                writer.write(',CT,q2')
            elif(sample in sample_by_cond['q3']):
                writer.write(',AC,q3')
            writer.write('\n')
    
    i = 1 
    while i <= 5:
        outer_train_set = []
        with open(root_dir + 'Cuffdiff_GEOM_POOL' + '_FOLD' + str(i) + '/read_groups.info') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            r = 0
            for row in csv_reader:
                if(r > 0):
                    full_sample_name = row[0]
                    sample_name_array = full_sample_name.split('.')
                    sample_name_RNAseq = sample_name_array[0]
                    sample_name_proteomic = RNAseq_to_Proteomic[sample_name_RNAseq]
                    outer_train_set.append(sample_name_proteomic)
                r += 1
                
        fname = 'Sample_Train' + str(i) + '.csv'
        with open(out_dir + fname, 'w') as writer:
            for sample in outer_train_set:
                writer.write(str(sample))
                if(sample in sample_by_cond['q1']):
                    writer.write(',AH,q1')
                elif(sample in sample_by_cond['q2']):
                    writer.write(',CT,q2')
                elif(sample in sample_by_cond['q3']):
                    writer.write(',AC,q3')
                writer.write('\n')
        
        
        j = 1
        while j <= 5:
            inner_train_set = []
            with open(root_dir + 'Cuffdiff_GEOM_POOL' + '_FOLD' + str(i) + '_' + str(j) + '/read_groups.info') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter='\t')
                r = 0
                for row in csv_reader:
                    if(r > 0):
                        full_sample_name = row[0]
                        sample_name_array = full_sample_name.split('.')
                        sample_name_RNAseq = sample_name_array[0]
                        sample_name_proteomic = RNAseq_to_Proteomic[sample_name_RNAseq]
                        inner_train_set.append(sample_name_proteomic)
                    r += 1
                    
            fname = 'Sample_Train' + str(i) + '_' + str(j) + '.csv'
            with open(out_dir + fname, 'w') as writer:
                for sample in inner_train_set:
                    writer.write(str(sample))
                    if(sample in sample_by_cond['q1']):
                        writer.write(',AH,q1')
                    elif(sample in sample_by_cond['q2']):
                        writer.write(',CT,q2')
                    elif(sample in sample_by_cond['q3']):
                        writer.write(',AC,q3')
                    writer.write('\n')
                    
            j += 1
                    
        i += 1
            
def generate_uniprot_proteins(fname:str):
    # Assume column 2 (start at 0) is review status, and column 3 is protein name.
    proteins = []
    with open(fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        i = 0
        for row in csv_reader:
            if(i == 0):
                i += 1
                continue
            if(row[2] == 'reviewed'):
                proteins.append(row[3])
            i += 1
            
    return proteins
        

def generate_proteomic_sample_files(core_sample_set:list, sample_by_cond:dict, main_dir:str):
    core_set_folds = generate_kfolds(core_sample_set, 5)
    core_set_train, core_set_valid = divide_folds_in_training_and_validation(core_set_folds)
    
    fname = 'Samples_ALL.csv'
    
    # print("core_sample_set: ", core_sample_set)
    
    with open(main_dir + fname, 'a') as writer:
        for sample in core_sample_set:
            writer.write(str(sample))
            if(sample in sample_by_cond['q1']):
                writer.write(',AH,q1')
            elif(sample in sample_by_cond['q2']):
                writer.write(',CT,q2')
            elif(sample in sample_by_cond['q3']):
                writer.write(',AC,q3')
            writer.write('\n')
        
    i = 1
    for sample_set in core_set_train:
        fname = 'Sample_Train' + str(i) + '.csv'
        with open(main_dir + fname, 'a') as writer:
            for sample in sample_set:
                writer.write(str(sample))
                if(sample in sample_by_cond['q1']):
                    writer.write(',AH,q1')
                elif(sample in sample_by_cond['q2']):
                    writer.write(',CT,q2')
                elif(sample in sample_by_cond['q3']):
                    writer.write(',AC,q3')
                writer.write('\n')
        i += 1
        
    core_set_folds_nested = generate_nested_kfolds(core_sample_set, 5, 5)
    i = 1
    for subset in core_set_folds_nested:
        subset_train, subset_validate = divide_folds_in_training_and_validation(subset)
        j = 1
        for sample_set in subset_train:
            fname = 'Sample_Train' + str(i) + '_' + str(j) + '.csv'
            with open(main_dir + fname, 'a') as writer:
                for sample in sample_set:
                    writer.write(str(sample))
                    if(sample in sample_by_cond['q1']):
                        writer.write(',AH,q1')
                    elif(sample in sample_by_cond['q2']):
                        writer.write(',CT,q2')
                    elif(sample in sample_by_cond['q3']):
                        writer.write(',AC,q3')
                    writer.write('\n')
            j += 1
              
        i += 1

def generate_proteomic_test_data(impute:str, scaling:str, impute_thresh:float = 0.2, log_transform = False):
    
    assert impute_thresh in [0, 0.05, 0.1, 0.2, 0.5, 1.0]
    assert impute in ['mean', 'median', 'kNN', 'zero']
    assert scaling in [None, 'standard', 'minmax', 'normal', 'robust']
    assert log_transform in [False, True]
    
    #fname = 'C:/...'
    fname = 'C:/...'
    
    proteomics_data = {}
    protein_names = []
    with open(fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        i = 0
        study_ids = []
        for row in csv_reader:
            if (i == 0):
                study_ids = row[1:]
                for study_id in study_ids:
                    proteomics_data[study_id] = []
            else:
                # Determine how many values are missing for this protein.
                miss_ratio = 0
                for value in row[1:]:
                    if(value == ''):
                        miss_ratio += 1
                miss_ratio = miss_ratio / len(row[2:])
                # If too many values are missing skip this protein completely.
                if(miss_ratio > impute_thresh):
                    continue
                
                j = 0 
                protein_names.append(row[0])
                protein_name = row[0]
                
                for value in row[1:]:
                    if(value == '' and impute != 'zero'):
                        proteomics_data[study_ids[j]].append('NaN')
                    elif(value == '' and impute == 'zero'):
                        proteomics_data[study_ids[j]].append(0)
                    else:
                        if(not log_transform):
                            proteomics_data[study_ids[j]].append(float(row[1 + j]))
                        elif(log_transform):
                            proteomics_data[study_ids[j]].append(math.log(1 + float(row[1 + j])))
                    j += 1
            i += 1
        
    prev_num_values = 0
    i = 0
    for k,v in proteomics_data.items():
        if(i == 0):
            prev_num_values = len(v)
        num_values = len(v)
        if(num_values != prev_num_values):
            raise ValueError("Not every sample has the same number of protein expression records.")
        prev_num_values = num_values
        i += 1
        
        
    study_ids1 = []
    for k in proteomics_data.keys():
        study_ids1.append(k)
    
        
    Y = []
    
    for study_id in study_ids1:
        cond = study_id[0:2]
        if(cond == 'AH'):
            Y.append(0)
        else:
            Y.append(1)

    Y = np.array(Y)
    
    X = []
    for k,v in proteomics_data.items():
        X.append(v)
    
    X = np.array(X)
    X = X.astype(float)
    Y = np.array(Y)
    
    if(impute != 'zero'):
        if(impute == 'kNN'):
            imputer = KNNImputer(n_neighbors=2, weights="uniform")
        elif(impute == 'mean'):
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        elif(impute == 'median'):
            imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        X_old = X
        X = imputer.fit_transform(X, Y)
        
    if(scaling == 'standard'):
        scaler = preprocessing.StandardScaler()
        scaled = scaler.fit_transform(X, Y)
        X = scaled
    elif(scaling == 'minmax'):
        scaler = preprocessing.MinMaxScaler()
        scaled = scaler.fit_transform(X, Y)
        X = scaled
    elif(scaling == 'normal'):
        scaler = preprocessing.Normalizer()
        scaled = scaler.fit_transform(X, Y)
        X = scaled
    elif(scaling == 'robust'):
        scaler = preprocessing.RobustScaler()
        scaled = scaler.fit_transform(X, Y)
        X = scaled
    
    assert(np.shape(X)[1] == len(protein_names))
        
    return X,Y,protein_names,study_ids1

def generate_proteomic_counts_means(fname:str, dataset:str):
    assert(dataset in ['LV_2Way', 'LV_3Way', 'LV_3Way_Old', 'PBMC_3Way', 'PBMC_3Way_Matched', 'LV_3Way_Matched',
                       'LV_3Way_Unmatched_Balanced', 'PBMC_3Way_Unmatched_Balanced', 'Validation'])
    
    sample_names = datasets[dataset]
    protein_names = []
    
    proteomics_data = {}
    with open(fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        i = 0
        study_ids = []
        for row in csv_reader:
            if (i == 0):
                study_ids = row[2:]
            else:
                protein_names.append(row[0])
                protein_name = row[0]
                proteomics_data[protein_name] = {'AH':[], 'CT':[], 'AC':[]}
                j = 0
                for value in row[2:]:
                    current_cond = ''
                    if(dataset == 'LV_2Way'):
                        if(study_ids[j] in LV_3Way_AH):
                            current_cond = 'AH'
                        elif(study_ids[j] in LV_3Way_CT):
                            current_cond = 'CT'
                        else:
                            j += 1
                            continue
                    if(dataset == 'LV_3Way'):
                        if(study_ids[j] in LV_3Way_AH):
                            current_cond = 'AH'
                        elif(study_ids[j] in LV_3Way_CT):
                            current_cond = 'CT'
                        elif(study_ids[j] in LV_3Way_AC):
                            current_cond = 'AC'
                        else:
                            j += 1
                            continue
                    if(dataset == 'LV_3Way_Old'):
                        if(study_ids[j] in LV_3Way_AH_Old):
                            current_cond = 'AH'
                        elif(study_ids[j] in LV_3Way_CT):
                            current_cond = 'CT'
                        elif(study_ids[j] in LV_3Way_AC):
                            current_cond = 'AC'
                        else:
                            j += 1
                            continue
                    elif(dataset == 'PBMC_3Way'):
                        if(study_ids[j] in PBMC_3Way_AH):
                            current_cond = 'AH'
                        elif(study_ids[j] in PBMC_3Way_CT):
                            current_cond = 'CT'
                        elif(study_ids[j] in PBMC_3Way_AC):
                            current_cond = 'AC'
                        else:
                            j += 1
                            continue
                    elif(dataset == 'PBMC_3Way_Matched'):
                        if(study_ids[j] in PBMC_3Way_AH_Matched):
                            current_cond = 'AH'
                        elif(study_ids[j] in PBMC_3Way_CT_Matched):
                            current_cond = 'CT'
                        elif(study_ids[j] in PBMC_3Way_AC_Matched):
                            current_cond = 'AC'
                        else:
                            j += 1
                            continue
                    elif(dataset == 'LV_3Way_Matched'):
                        if(study_ids[j] in LV_3Way_Matched_AH):
                            current_cond = 'AH'
                        elif(study_ids[j] in LV_3Way_Matched_CT):
                            current_cond = 'CT'
                        elif(study_ids[j] in LV_3Way_Matched_AC):
                            current_cond = 'AC'
                        else:
                            j += 1
                            continue
                    elif(dataset == 'LV_3Way_Unmatched_Balanced'):
                        if(study_ids[j] in LV_3Way_Unmatched_AH_Balanced):
                            current_cond = 'AH'
                        elif(study_ids[j] in LV_3Way_Unmatched_CT_Balanced):
                            current_cond = 'CT'
                        elif(study_ids[j] in LV_3Way_Unmatched_AC_Balanced):
                            current_cond = 'AC'
                        else:
                            j += 1
                            continue
                    elif(dataset == 'PBMC_3Way_Unmatched_Balanced'):
                        if(study_ids[j] in PBMC_3Way_AH_Unmatched_Balanced):
                            current_cond = 'AH'
                        elif(study_ids[j] in PBMC_3Way_CT_Unmatched_Balanced):
                            current_cond = 'CT'
                        elif(study_ids[j] in PBMC_3Way_AC_Unmatched_Balanced):
                            current_cond = 'AC'
                        else:
                            j += 1
                            continue
                    elif(dataset == 'Validation'):
                        if(study_ids[j][0:2] == 'AH'):
                            current_cond = 'AH'
                        else:
                            current_cond = 'CT'
                        
                    j += 1
                    if(value == ''):
                        #proteomics_data[protein_name][current_cond].append(0)
                        continue
                    else:
                        proteomics_data[protein_name][current_cond].append(float(value))
                        
                    #print(proteomics_data[protein_name])
                    
            i += 1
    
    proteomics_data2 = {}
    for k,v in proteomics_data.items():
        if(dataset in ['LV_3Way', 'LV_3Way_Old', 'PBMC_3Way', 'LV_3Way_Matched', 'PBMC_3Way_Matched',
                       'LV_3Way_Unmatched_Balanced', 'PBMC_3Way_Unmatched_Balanced']):
            mean1 = np.mean(np.array(proteomics_data[k]['AH']))
            mean2 = np.mean(np.array(proteomics_data[k]['CT']))
            mean3 = np.mean(np.array(proteomics_data[k]['AC']))
            proteomics_data2[k] = [mean1,mean2,mean3]
        elif(dataset == 'LV_2Way'):
            mean1 = np.mean(np.array(proteomics_data[k]['AH']))
            mean2 = np.mean(np.array(proteomics_data[k]['CT']))
            proteomics_data2[k] = [mean1,mean2]
        elif(dataset == 'Validation'):
            mean1 = np.mean(np.array(proteomics_data[k]['AH']))
            mean2 = np.mean(np.array(proteomics_data[k]['CT']))
            proteomics_data2[k] = [mean1,mean2]
        
    cond_order = []
    if(dataset in ['LV_3Way', 'LV_3Way_Old', 'PBMC_3Way', 'LV_3Way_Matched', 'PBMC_3Way_Matched',
                   'LV_3Way_Unmatched_Balanced', 'PBMC_3Way_Unmatched_Balanced']):
        cond_order = ['AH', 'CT', 'AC']
    elif(dataset == 'LV_2Way'):
        cond_order = ['AH', 'CT']
    elif(dataset == 'Validation'):
        cond_order = ['AH', 'CT']
    
    return proteomics_data2, cond_order
            
def generate_proteomic_counts_individual(fname:str, dataset:str):
    assert(dataset in ['LV_2Way', 'LV_3Way', 'LV_3Way_Matched', 'LV_3Way_Matched_Balanced', 'LV_3Way_Unmatched',
                       'LV_3Way_Unmatched_Balanced', 'PBMC_3Way', 'PBMC_3Way_Matched', 'PBMC_3Way_Matched_Balanced',
                       'PBMC_3Way_Unmatched', 'PBMC_3Way_Unmatched_Balanced', 'Validation'])
    
    sample_names = datasets[dataset]
    protein_names = []
    
    proteomics_data = {}
    with open(fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        i = 0
        study_ids = []
        for row in csv_reader:
            if (i == 0):
                study_ids = row[2:]
                for study_id in study_ids:
                    if(study_id in sample_names):
                        proteomics_data[study_id] = dict()
            else:                
                j = 0 
                protein_names.append(row[0])
                protein_name = row[0]
                
                for value in row[2:]:
                    if(study_ids[j] in sample_names):
                        if(value == ''):
                            proteomics_data[study_ids[j]][protein_name] = 0
                        else:
                            proteomics_data[study_ids[j]][protein_name] = float(value)
                    j += 1
            i += 1
            
    return proteomics_data, protein_names

def generate_proteomic_matrices(fname:str, dataset:str, impute:str, scaling:str, impute_thresh:float = 0.2, log_transform = False,
                                harmonization_protein_set:list = []):
    '''
    fname: name of the file containing proteomic counts.
    datasets: PBMC_3Way, PBMC_3Way_Matched, LV_3Way, LV_3Way_Matched.
    '''
    
    protein_names = []
    sample_names = datasets[dataset]
    
    assert dataset in ['PBMC_3Way', 'PBMC_3Way_Matched', 'PBMC_3Way_Matched_Balanced',
                       'PBMC_3Way_Unmatched', 'PBMC_3Way_Unmatched_Balanced',
                       'LV_3Way', 'LV_2Way', 'LV_3Way_Old', 'LV_3Way_Matched', 'LV_3Way_Matched_Balanced',
                       'LV_3Way_Unmatched', 'LV_3Way_Unmatched_Balanced']
    assert impute_thresh in [0, 0.05, 0.1, 0.2, 0.5, 1.0]
    assert impute in ['mean', 'median', 'kNN', 'zero']
    assert scaling in [None, 'standard', 'minmax', 'normal', 'robust']
    assert log_transform in [False, True]
    
    proteomics_data = {}
    with open(fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        i = 0
        study_ids = []
        for row in csv_reader:
            if (i == 0):
                study_ids = row[2:]
                for study_id in study_ids:
                    if(study_id in sample_names):
                        proteomics_data[study_id] = []
            else:
                # Determine how many values are missing for this protein.
                miss_ratio = 0
                for value in row[2:]:
                    if(value == ''):
                        miss_ratio += 1
                miss_ratio = miss_ratio / len(row[2:])
                #print("Miss Ratio: ", miss_ratio)
                #print("Impute Threshold: ", impute_thresh)
                # If too many values are missing skip this protein completely.
                if(miss_ratio > impute_thresh):
                    i += 1
                    continue
                
                if(len(harmonization_protein_set) > 0):
                    if row[0] not in harmonization_protein_set:
                        continue
                
                j = 0 
                protein_names.append(row[0])
                protein_name = row[0]
                
                for value in row[2:]:
                    if(study_ids[j] in sample_names):
                        #print("Value: ", value)
                        #print("Row: ", i)
                        if(value == '' and impute != 'zero'):
                            proteomics_data[study_ids[j]].append('NaN')
                        elif(value == '' and impute == 'zero'):
                            proteomics_data[study_ids[j]].append(0)
                        else:
                            if(not log_transform):
                                proteomics_data[study_ids[j]].append(float(row[2 + j]))
                            elif(log_transform):
                                proteomics_data[study_ids[j]].append(math.log(1 + float(row[2 + j])))
                    j += 1
            i += 1
        
    prev_num_values = 0
    i = 0
    for k,v in proteomics_data.items():
        if(i == 0):
            prev_num_values = len(v)
        num_values = len(v)
        if(num_values != prev_num_values):
            raise ValueError("Not every sample has the same number of protein expression records.")
        prev_num_values = num_values
        i += 1
    
    study_ids1 = []
    for k in proteomics_data.keys():
        study_ids1.append(k)
        
    #print("Protein Names: ", protein_names)
    Y = []
    if(dataset in ['PBMC_3Way', 'PBMC_3Way_Matched', 'PBMC_3Way_Matched_Balanced',
                   'PBMC_3Way_Unmatched', 'PBMC_3Way_Unmatched_Balanced']):
        # This file is used to map patient ids to patient conditions.
        cond_to_label = {'AH':0, 'CT':1, 'AC':2}
        for study_id in study_ids1:
            if(study_id in PBMC_3Way_AH):
                Y.append(cond_to_label['AH'])
            elif(study_id in PBMC_3Way_CT):
                Y.append(cond_to_label['CT'])
            elif(study_id in PBMC_3Way_AC):
                Y.append(cond_to_label['AC'])
            else:
                raise ValueError('Unrecognized sample name.')
    else: 
        if(dataset == 'LV_2Way'):
            cond_to_label = {'AH_LB':0, 'AT_LT_HH':1}
        else:
            cond_to_label = {'AH_LB':0, 'AT_LT_HH':1, 'AH_LT_UMN':2}
        
        for study_id in study_ids1:
            #print(study_id)
            if(study_id[3:5] == 'LB'):
                # AH Sample
                Y.append(0)
            elif(study_id[3:5] == 'LT'):
                if(study_id[6:8] == 'HH'):
                    # CT
                    Y.append(1)
                elif(study_id[6:8] == 'UM' and dataset != 'LV_2Way'):
                    # AC
                    Y.append(2)
                else:
                    raise ValueError("Unrecognized sample name.")
            else:
                raise ValueError("Unrecognized sample name.")
    
    Y = np.array(Y)
    
    X = []
    for k,v in proteomics_data.items():
        X.append(v)
    
    X = np.array(X)
    X = X.astype(float)
    Y = np.array(Y)
    
    if(impute != 'zero'):
        if(impute == 'kNN'):
            imputer = KNNImputer(n_neighbors=2, weights="uniform")
        elif(impute == 'mean'):
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        elif(impute == 'median'):
            imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        X_old = X
        X = imputer.fit_transform(X, Y)
        
    if(scaling == 'standard'):
        scaler = preprocessing.StandardScaler()
        scaled = scaler.fit_transform(X, Y)
        X = scaled
    elif(scaling == 'minmax'):
        scaler = preprocessing.MinMaxScaler()
        scaled = scaler.fit_transform(X, Y)
        X = scaled
    elif(scaling == 'normal'):
        scaler = preprocessing.Normalizer()
        scaled = scaler.fit_transform(X, Y)
        X = scaled
    elif(scaling == 'robust'):
        scaler = preprocessing.RobustScaler()
        scaled = scaler.fit_transform(X, Y)
        X = scaled
    
    assert(np.shape(X)[1] == len(protein_names))
        
    return X,Y,protein_names,study_ids1


def generate_CV_split_proteomics(cv_k:int, fnames:list, sample_order:list):
    index_outer = 0
    outer_cv_split = []
    while index_outer < cv_k:
        fname = fnames[index_outer]
        #print(fname)
        # Translate sample names within the corresponding training set into indices of sample_order. 
        # These indeces are the training indeces. The remaining indeces are validation indeces. 
        train_set_sample_names = read_in_csv_file_one_column(fname, 0, ',')
        #print(train_set_sample_names)
        train_set_indeces = []
        validation_set_indeces = []
        for train_set_sample in train_set_sample_names:
            if(train_set_sample not in sample_order):
                # print(fname)
                # print(train_set_sample)
                raise ValueError("Could not find a training sample within sample_order.")
            sample_index = sample_order.index(train_set_sample)
            train_set_indeces.append(sample_index)
            
        j = 0 
        while j < len(sample_order):
            if j not in train_set_indeces:
                validation_set_indeces.append(j)
            j += 1
        #print(validation_set_indeces)   
        outer_cv_split.append( (train_set_indeces, validation_set_indeces) )
        index_outer += 1
        
    return outer_cv_split

def generate_top_DE_proteins(fname_in:str, num_features:int, feature_names:list, fname_out:str, num_conditions:int,
                             impute_thresh:float, taboo_features:list = [], q_value:float = 0.05):
    '''
    DE Protein file organization: 
    0 => Protein Name
    1 => Count1
    2 => Cond1 # non-missing
    3 => Count2
    4 => Cond2 # non-missing
    5 => Fold Change
    6 => P-Value
    7 => Q-Value
    8 => Cond1_Label
    9 => Cond2_Label
    10 => Cond1 # Total
    11 => Cond2 # Total'''
    
    # The inferno appends a number to names of proteins for some pairwise comparisons. This has to be dealt with in order to correctly 
    # process the proteins. Ex: ALBU_HUMAN2 instead of ALBU_HUMAN.
    
    DE_dati = dict()
    with open(fname_in) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        i = 0
        for row in csv_reader:
            if (i > 0):
                try:
                    protein_name = row[0]
                    index = protein_name.index('HUMAN')
                    protein_name = protein_name[0:(index+5)]
                    
                    if((1-(int(row[2])/int(row[10]))<=impute_thresh) and (1-(int(row[4])/int(row[11]))<=impute_thresh) and
                       (float(row[7]) < q_value) and (protein_name in feature_names) and (protein_name not in taboo_features)):
                        #print()
                        cond_label1 = row[8]
                        cond_label2 = row[9]
                        try:
                            DE_dati[cond_label1 + '_' + cond_label2][protein_name] = abs(float(row[5]))
                        except KeyError:
                            DE_dati[cond_label1 + '_' + cond_label2] = dict()
                except ValueError:
                    # If there is missing data (p-value, q-value, etc, just skip this protein.)
                    continue
            i += 1
                    
    iters = []
    for v in DE_dati.values():
        s_DE_data = sorted(v.items(), key = lambda x: abs(x[1]), reverse = True)
        #print("s_DE_data:", s_DE_data)
        iters.append(iter(s_DE_data))
    
    i = 0
    iter_flags = []
    while i < len(iters):
        iter_flags.append(False)
        i += 1
    
    result = []
    j = 0
    while len(result) < num_features:
        #print('j:', j)
        #print('result:', result)
        #print('iter_flags:', iter_flags)
        if(all(iter_flags)):
            break
        
        if(not iter_flags[j]):
            try:
                gene_count = next(iters[j])
                if(gene_count[0] not in result):
                    result.append(gene_count)
            except StopIteration:
                iter_flags[j] = True
            
        j += 1
        j = j % (len(iter_flags))
    
    with open(fname_out, 'w') as writer:
        for gene_feature in result:
            writer.write(gene_feature[0] + '\t' + str(gene_feature[1]) + '\n')
            
    return result
    # DE Feature Ranking Over

def select_features_in_CV_proteomics(X, Y, sample_order:list, protein_names:list, work_dir:str, out_dir:str, cv_k:int, 
                                     feature_size:int, num_conditions:int, impute_thresh:float = 0.2, FS_Mode = 'IG',
                                     Filter:bool = True, Filter_Mode = None, std_treshold:float = 3.5):
    #print("Protein Names: ", protein_names)
    assert(Filter_Mode in ['Variance', None])
    assert(FS_Mode in ['DE', 'IG'])
    index = 0
    
    fnames = []
    z = 0
    while z < cv_k:
        fname_in = work_dir + 'Sample_Train' + str(z+1) + '.csv'
        fnames.append(fname_in)
        z += 1
    
    cv_split = generate_CV_split_proteomics(cv_k, fnames, sample_order)
    
    for tr_va_indeces in cv_split:
        
        X_train = X[tr_va_indeces[0], :]
        Y_train = Y[tr_va_indeces[0]]

        if(Filter):
            outlier_features = detect_outlier_features_by_std2(X_train, Y_train, protein_names, std_treshold, True)
        else:
            outlier_features = []

        filename = out_dir + 'top_proteomic_features_set' + str(index)
        filename += '.txt'
        if(FS_Mode == 'IG'):
            generate_top_IG_features(X_train, Y_train, protein_names, feature_size, filename, outlier_features)
        else:
            fname_in = work_dir + 'Sample_Train' + str(index+1) + '_DE.csv'
            generate_top_DE_proteins(fname_in, feature_size, protein_names, filename, num_conditions, impute_thresh, outlier_features)
        
        index += 1

def select_features_in_nested_CV_proteomics(X, Y, sample_order:list, protein_names:list, work_dir:str, out_dir:str, num_conditions:int, 
                                            impute_thresh = 0.2, cv_k_outer:int = 5, cv_k_inner:int = 5, num_DEPs:int = 500,
                                            FS_Mode:str = 'IG', Filter:bool = True, Filter_Mode = 'Variance', std_treshold:float = 3.5,
                                            DEG_Filter:bool = False, DEG_Filter_Dir:str = ''):
    # print("Protein Names: ", protein_names)
    assert(Filter_Mode in ['Variance', None])
    assert(FS_Mode in ['DE', 'IG'])
    
    index_outer = 0
    fnames = []
    while index_outer < cv_k_outer:
        fname = work_dir + 'Sample_Train' + str(index_outer+1) + '.csv'
        
        fnames.append(fname)
        index_outer += 1
        
    outer_cv_split = generate_CV_split_proteomics(cv_k_outer, fnames, sample_order)
    # print("Outer_CV_SPLIT: ", outer_cv_split)
    
    index_outer = 0
    for tr_va_te_indeces in outer_cv_split:
        
        X_train_validate = X[tr_va_te_indeces[0], :]
        Y_train_validate = Y[tr_va_te_indeces[0]]
        
        #Flag highly variant or mostly 0 features.
        if(Filter):
            outlier_features = detect_outlier_features_by_std2(X_train_validate, Y_train_validate, protein_names, std_treshold, True)
        else:
            outlier_features = []
        
        fname = out_dir + 'top_proteomic_features_set' + str(index_outer) + '.txt'
        fname_in = work_dir + 'Sample_Train' + str(index_outer+1) + '_DE.csv'
        if(FS_Mode == 'IG'):
            generate_top_IG_features(X_train_validate, Y_train_validate, protein_names, num_DEPs, fname, outlier_features)
        else:
            if(DEG_Filter):
                # Do an intersection of protein names and DEGs translated to DEP names. Use this as the superset instead of protein names.
                DEG_Proteins = generate_uniprot_proteins(DEG_Filter_Dir + 'FOLD' + str(index_outer + 1) + '.tsv')
                protein_names_filtered = (set(DEG_Proteins) & set(protein_names))
                # print("Proteins left post DEG pre-filtering: ", len(protein_names_filtered))
                generate_top_DE_proteins(fname_in, num_DEPs, protein_names_filtered, fname, num_conditions, impute_thresh, outlier_features)
            else:
                generate_top_DE_proteins(fname_in, num_DEPs, protein_names, fname, num_conditions, impute_thresh, outlier_features)

        index_inner = 0
        fnames = []
        while index_inner < cv_k_inner:
            test_accuracies = []
            fname = work_dir + 'Sample_Train' + str(index_outer+1) + '_' + str(index_inner + 1) + '.csv'
            fnames.append(fname)
            
            index_inner += 1
        
        outer_sample_order = list(np.array(sample_order)[tr_va_te_indeces[0]])
        
        inner_cv_split = generate_CV_split_proteomics(cv_k_inner, fnames, outer_sample_order)
        #print("INNER_CV_SPLIT: ", inner_cv_split)
            
        index_inner = 0
        for tr_va_indeces in inner_cv_split:
            
            X_train = X_train_validate[tr_va_indeces[0], :]
            Y_train = Y_train_validate[tr_va_indeces[0]]
            
            #Flag highly variant or mostly 0 features.
            if(Filter):
                outlier_features = detect_outlier_features_by_std2(X_train, Y_train, protein_names, std_treshold, True)
            else:
                outlier_features = []
            
            fname = out_dir + 'top_proteomic_features_set' + str(index_outer) + '_' + str(index_inner) + '.txt'
            fname_in = work_dir + 'Sample_Train' + str(index_outer+1) + '_' + str(index_inner+1) + '_DE.csv'
            if(FS_Mode == 'IG'):
                generate_top_IG_features(X_train, Y_train, protein_names, num_DEPs, fname, outlier_features)
            else:
                if(DEG_Filter):
                    # Do an intersection of protein names and DEGs translated to DEP names. Use this as the superset instead of protein names.
                    DEG_Proteins = generate_uniprot_proteins(DEG_Filter_Dir + 'FOLD' + str(index_outer + 1) + '_' + str(index_inner + 1) + '.tsv')
                    protein_names_filtered = (set(DEG_Proteins) & set(protein_names))
                    generate_top_DE_proteins(fname_in, num_DEPs, protein_names_filtered, fname, num_conditions, impute_thresh, outlier_features)
                else:
                    generate_top_DE_proteins(fname_in, num_DEPs, protein_names, fname, num_conditions, impute_thresh, outlier_features)
            index_inner += 1
            
        index_outer += 1
    
def perform_CV_proteomics(X, Y, sample_order:list, protein_names:list, root_dir:str, root_dir2:str, feature_size:int, cv_k:int, 
                          model_name:str, balanced_acc:bool = False):
    assert(model_name in ['LR', 'kNN', 'SVM'])
    
    p_grid = p_grid_dict[model_name]
    
    hyper_param_names = []
    hyper_param_values = []
    for k,v in p_grid.items():
        hyper_param_names.append(k)
        hyper_param_values.append(v)
    
    hyper_param_grid = list(itertools.product(*hyper_param_values))
    
    max_acc = 0
    best_hyper_param_dict = {}
    
    for hyper_param_tup in hyper_param_grid:
        hyper_param_dict = {}
        i = 0
        for hyper_param_name in hyper_param_names:
            hyper_param_dict[hyper_param_name] = hyper_param_tup[i]
            i += 1
           
        index = 0
        validate_accuracies = []
        
        fnames = []
        z = 0
        while z < cv_k:
            fname_in = root_dir + str(z+1) + '.csv'
            fnames.append(fname_in)
            z += 1
        
        #print("Sample order in CV: ", sample_order)
        cv_split = generate_CV_split_proteomics(cv_k, fnames, sample_order)
        
        for tr_va_indeces in cv_split:
            
            X_train = X[tr_va_indeces[0], :]
            Y_train = Y[tr_va_indeces[0]]
            X_validate = X[tr_va_indeces[1], :]
            Y_validate = Y[tr_va_indeces[1]]

            feature_file = root_dir2 + str(index) + '.txt'
            proteins_to_read = read_in_csv_file_one_column(feature_file, 0, '\t', 0, feature_size)
            
            # print("Proteins to read: ", proteins_to_read)
            # print("Protein Names: ", protein_names)
    
            X_train = select_features_from_matrix(X_train, protein_names, proteins_to_read)
            X_validate = select_features_from_matrix(X_validate, protein_names, proteins_to_read)
            
            if(model_name == 'SVM'):model = SVC(**hyper_param_dict)
            elif(model_name == 'kNN'):model = KNeighborsClassifier(**hyper_param_dict)
            elif(model_name == 'LR'):model = LogisticRegression(**hyper_param_dict)
                
            clf = model.fit(X_train, Y_train)
            Y_hat = clf.predict(X_validate)
            if(not balanced_acc):
                validate_accuracy = accuracy_score(Y_validate, Y_hat)
            else:
                validate_accuracy = balanced_accuracy_score(Y_validate, Y_hat)
            validate_accuracies.append(validate_accuracy)
            
            index += 1
                
        validate_accuracy = np.mean(np.array(validate_accuracies))
        if(validate_accuracy > max_acc):
            max_acc = validate_accuracy
            best_hyper_param_dict = hyper_param_dict
        
    # print("Max Accuracy: ", max_acc)
    # print("Hyper-parameter tuning is over.") 
    # print("Best hyper-parameters are: ", best_hyper_param_dict)
        
    return best_hyper_param_dict
    
def perform_nested_CV_proteomics(X, Y, sample_order:list, protein_names:list, work_dir:str, out_dir:str, feature_size:int,
                                 cv_k_outer:int, cv_k_inner:int, model_name:str, balanced_acc:bool = False):  
    assert(model_name in ['kNN', 'SVM', 'LR'])
    test_accuracies = []
    test_baccuracies = []
    conf_matrices = []
    
    index_outer = 0
    outer_cv_split = []
    fnames = []
    while index_outer < cv_k_outer:
        fname = work_dir + 'Sample_Train' + str(index_outer+1) + '.csv'
        fnames.append(fname)
        
        index_outer += 1
    
    outer_cv_split = generate_CV_split_proteomics(cv_k_outer, fnames, sample_order)
    print("Sample order in nested_CV: ", sample_order)
    
    misclass = {}
    index_outer = 0
    for tr_va_te_indeces in outer_cv_split:
        
        X_train_validate = X[tr_va_te_indeces[0], :]
        Y_train_validate = Y[tr_va_te_indeces[0]]
        X_test = X[tr_va_te_indeces[1], :]
        Y_test = Y[tr_va_te_indeces[1]]
        print("Train Indeces: ", tr_va_te_indeces[0])
        print("Test Indeces: ", tr_va_te_indeces[1])
        
        # Do inner loop of CV once to find best hyper-parameter configuration.
        outer_sample_order = list(np.array(sample_order)[tr_va_te_indeces[0]])
        
        print("Outer sample order: ", outer_sample_order)
        
        temp = out_dir + 'top_proteomic_features_set'
        root_dir_inner = work_dir + 'Sample_Train' + str(index_outer+1) + '_'
        root_dir_inner2 = temp + str(index_outer) + '_'
        best_hyper_param_dict = perform_CV_proteomics(X_train_validate, Y_train_validate, outer_sample_order, protein_names,
                                                      root_dir_inner, root_dir_inner2, feature_size, cv_k_inner, model_name, balanced_acc)
        
        print("NESTED CROSS VALIDATION BEST HYPER-PARAM DICT.")
        print(best_hyper_param_dict)
        
        feature_file = temp + str(index_outer) + '.txt'
        proteins_to_read = read_in_csv_file_one_column(feature_file, 0, '\t', 0, feature_size)
        
        print("Outer Set Index: ", str(index_outer))
        print("Top Proteins For This Training Set: ", proteins_to_read)
        
        X_train_validate = select_features_from_matrix(X_train_validate, protein_names, proteins_to_read)
        X_test = select_features_from_matrix(X_test, protein_names, proteins_to_read)
        
        if(model_name == 'SVM'):model = SVC(**best_hyper_param_dict)
        elif(model_name == 'kNN'):model = KNeighborsClassifier(**best_hyper_param_dict)
        elif(model_name == 'LR'):model = LogisticRegression(**best_hyper_param_dict)
        
        clf = model.fit(X_train_validate, Y_train_validate)
        Y_hat = clf.predict(X_test)
        print("Y_hat: ", Y_hat)
        print("Y_test: ", Y_test)
        
        test_samples = np.array(sample_order)[tr_va_te_indeces[1]]
        num_test_samples = Y_hat.shape[0]
        q = 0
        while q < num_test_samples:
            if(Y_hat[q] != Y_test[q]):
                # print("Misclassified: ", test_samples[q])
                try:
                    misclass[test_samples[q]] = misclass[test_samples[q]] + 1
                except KeyError:
                    misclass[test_samples[q]] = 1
            q += 1
        
        
        if(not balanced_acc):
            accuracy = accuracy_score(Y_test, Y_hat)
        else:
            accuracy = balanced_accuracy_score(Y_test, Y_hat)
        baccuracy = balanced_accuracy_score(Y_test, Y_hat)
        #print("Accuracy: ", accuracy)
        test_accuracies.append(accuracy)
        test_baccuracies.append(baccuracy)
        conf_matrices.append(confusion_matrix(Y_test, Y_hat))
        
        index_outer += 1
            
    test_accuracies = np.array(test_accuracies)
    mean_test_accuracy = np.mean(test_accuracies)
    mean_test_baccuracy = np.mean(test_baccuracies)
    print("**************************************")
    print("FEATURE SIZE:", feature_size)
    print("MEAN TEST ACCURACY: ", mean_test_accuracy)       
    i = 0
    mean_conf_matrix = 0
    while i < len(conf_matrices):
        if(i == 0):
            mean_conf_matrix = conf_matrices[0]
        else:
            mean_conf_matrix = mean_conf_matrix + conf_matrices[i]
        i += 1
    print("CONFUSION MATRIX: ", mean_conf_matrix)
    print("**************************************")
    
    # Hardcoded Instructions
    # dataset = 'LV_3Way'
    # conditions = ['AH', 'CT', 'AC']
    # gen_conf_matrix(work_dir, dataset, np.array(mean_conf_matrix), conditions, mean_test_accuracy, mean_test_baccuracy, model_name)
    # plot_misclass(work_dir, dataset, misclass, model_name)
    # plot_feature_importance(out_dir, dataset, 1, cv_k_outer, model_name, True, 27)
    
    return mean_test_accuracy, mean_conf_matrix, misclass

def verify_nested_cross_val_proteomics(root_dir:str, samples_expected:list, k_outer:int, k_inner:int):
    sample_names = []
    fname = root_dir + 'Samples_ALL.csv'
    sample_names = read_in_csv_file_one_column(fname, 0, ',')
    
    # Verify sample composition
    assert(len(set(samples_expected) - set(sample_names)) == 0)
    assert(len(set(sample_names) - set(samples_expected)) == 0)
    
    sample_count = {}
    samples_union = []
    for sample in samples_expected:
        sample_count[sample] = 0
    # Verify that each sample is in k-1 training sets (outer cv split)
    i = 1
    while i <= k_outer:
        samples = []
        fname = root_dir + 'Sample_Train' + str(i) + '.csv'
        with open(fname) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                sample_name = row[0]
                samples.append(sample_name)
                sample_count[sample_name] += 1
        i += 1
        samples_union += samples
    
    #print(sample_count)
    # This is only true if number of samples in each condition is > than k_outer.
    # samples_union = set(samples_union)
    # for sample,count in sample_count.items():
    #     #print("sample:", sample)
    #     #print(count)
    #     assert(count == (k_outer-1))
    
    # Also verify that the union of all outer train sets equals to the expected samples
    assert(len(set(samples_expected) - set(samples_union)) == 0)
    assert(len(set(samples_union) - set(samples_expected)) == 0)
    
    # Verify that each inner CV loops is valid
    # Each inner CV loop must only possess samples from the corresponding outer CV training set
    # Each sample should appear in k-1 training sets within inner CV loop
    
    z = 1
    outer_train_sets = []
    while z <= k_outer:
        
        outer_train_set = []
        fname = root_dir + 'Sample_Train' + str(z) + '.csv'
        with open(fname) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                sample_name = row[0]
                outer_train_set.append(sample_name)
        outer_train_set = set(outer_train_set)
        
        i = 1
        samples_inner = {}
        while i <= k_inner:
            fname = root_dir + 'Sample_Train' + str(z) + '_' + str(i) + '.csv'
            with open(fname) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    sample_name = row[0]
                    try:
                        samples_inner[sample_name] += 1
                    except KeyError:
                        samples_inner[sample_name] = 1
            i = i + 1
            
        # This is only true if number of samples in each condition is > than k_inner.
        # for sample,count in samples_inner.items():
        #     # print(count)
        #     assert(count == (k_inner-1))
            
        # The union of all inner samples should equal to the corresponding outer training set.
        samples_inner_union = []
        for sample in samples_inner.keys():
            samples_inner_union.append(sample)
        
        samples_inner_union = set(samples_inner_union)
        assert(len(outer_train_set - samples_inner_union) == 0)
        assert(len(samples_inner_union - outer_train_set) == 0)
        
        z += 1
    
    # Assume that all training sets are of similar length ((k-1)/k) * N. When combined with the above performed assertions, 
    # this should be sufficient to establish that the outer and inner splits were performed correctly. 

def biological_validation_proteomics(protein_set:list, analysis_type:str):
    '''Analysis type can be Pathway, Tissue, or Disease. '''
    assert(analysis_type in ['Pathway', 'Tissue', 'Disease'])
    
    url = r"https://agotool.org/api_orig"
    
    kbs = ""
    if(analysis_type == 'Pathway'):
        kbs = "-21;-52;-58"
    elif(analysis_type == 'Tissue'):
        kbs = "-25"
    elif(analysis_type == 'Disease'):
        kbs = "-26"
    
    fg = "%0d".join(protein_set)
    result = requests.post(url,
                       params={"output_format": "tsv",
                               "enrichment_method": "genome",
                               "taxid": 9606,
                               "filter_parents": True,
                               "filter_foreground_count_one": True,
                               "filter_PMID_top_n": 20,
                               "caller_identity": None,
                               "FDR_cutoff": 0.05,
                               "limit_2_entity_type":kbs,
                               "background": None,
                               "background_intensity": None,
                               "goslim": "basic",
                               "o_or_u_or_both": "both",
                               "num_bins": 100,
                               "p_value_cutoff": 0.01,
                               "multiple_testing_per_etype": True},
                       data={"foreground": fg})
    
    df = pd.read_csv(StringIO(result.text), sep='\t')
    
    disease_regex = 'hepa|liver|cirrhosis|NAFLD|liver fibrosis|NASH|steatohepatitis|HCV|'
    disease_regex += 'alcohol|sepsis|septic shock|hypercholesterolemia|'
    disease_regex += 'hyperlipidemia|obesity'
    
    tissue_regex = 'Blood|Macrophage|Erythro|Platelet|Basophil|Neutrophil|Eosinophil|Cytokine|Tumor Necrosis Factor|'
    tissue_regex += 'Monocyte|Lymphocyte|Granulocyte|Dendritic|Megakaryocyte|T Cell|B Cell|NK Cell|Toll-like receptor|'
    tissue_regex += 'Fc receptor|Liver|Hepatocyte|Stellate|Kupffer|Sinusoidal Endothelial Cells|'
    tissue_regex += 'CD34+|Natural Killer Cell|PBMC|Tcell|Bcell|lymphoblast|CD8+|CD19+|CD4+|CD71+|Omentum'
    
    pathway_regex = 'Interferon|Immun|Interleukin|Prolactin|Complement|Chemokine|Oncostatin M|Rejection|Inflamma|' 
    pathway_regex += 'IL-1|IL1|IL-|selenium|osteopontin|circulation|coagulation|clotting|biosynthesis|'
    pathway_regex += 'degradation|cholesterol|lipid|TNF|steroid|metal ion|heme|metallo|CXCR|LDL|'
    pathway_regex += 'Phagocytosis|metabolism|TYROBP|AP-1|' + disease_regex + '|' + tissue_regex
    
    selected_regex = ""
    if(analysis_type == "Pathway"):
        selected_regex = pathway_regex
    elif(analysis_type == "Tissue"):
        selected_regex = tissue_regex
    elif(analysis_type == "Disease"):
        selected_regex = disease_regex
    
    result = df
    result1 = result.loc[result['FDR'] < 0.05]
    result2 = result1.loc[result1['description'].str.contains(selected_regex, case = False)]
    
    return result1,result2


def classify_with_nested_CV_proteomics(root_dir:str, work_dir:str, out_dir:str, dataset:str, model_name:str, conditions:list, cv_k_outer:int,
                                       cv_k_inner:int, features_to_gen:int, feature_sizes:list, FS_Mode:str, Filter:bool, Filter_Mode:str,
                                       balanced_acc:bool, scaling:str, impute:str, log_transform:bool, impute_thresh:float = 0.2,
                                       tissue:str = 'LV', std_treshold:float = 3.5, DEG_Filter:bool = False, DEG_Filter_Dir:str = ''):
    # ---------------------------------------------------------------------------------------------------------------- 
    # ******************************************Verify Inner / Outer CV Splits****************************************
    # ---------------------------------------------------------------------------------------------------------------- 
    print("Nested cross-validation within our project data.")
    num_conditions = len(conditions)
    assert(FS_Mode in ['DE', 'IG'])
    assert(Filter_Mode in ['Variance', None])
    assert(model_name in ['LR', 'SVM', 'kNN'])
    assert(tissue in ['PB', 'LV'])
    assert(scaling in [None, 'standard', 'minmax', 'normal', 'robust'])
    assert(impute in ['mean', 'median', 'kNN', 'zero'])
    assert(impute_thresh in [0, 0.05, 0.1, 0.2, 0.5, 1.0])
    assert(dataset in ['PBMC_3Way', 'PBMC_3Way_Matched', 'PBMC_3Way_Unmatched_Balanced',
                       'LV_2Way', 'LV_3Way', 'LV_3Way_Matched', 'LV_3Way_Unmatched_Balanced'])
    if(tissue == 'LV'):
        # fname = root_dir + 'AH_TMT_global_biopsy_protein_ratio_cleaned_BATCH_CORRECTED.csv'
        fname = root_dir + 'AH_TMT_global_biopsy_protein_ratio.csv'
        if(num_conditions == 3):
            Expected_Samples = LV_3Way
        if(dataset == 'LV_3Way_Matched'):
            Expected_Samples = LV_3Way_Matched
        if(dataset == 'LV_3Way_Unmatched_Balanced'):
            Expected_Samples = LV_3Way_Unmatched_Balanced
        if(dataset == 'LV_2Way'):
            Expected_Samples = LV_2Way
    elif(tissue == 'PB'):
        fname = root_dir + 'AH_PBMC_masterproteinratiofirst.csv'
        if(num_conditions == 3):
            Expected_Samples = PBMC_3Way
        if(dataset == 'PBMC_3Way_Matched'):
            Expected_Samples = PBMC_3Way_Matched
        elif(dataset == 'PBMC_3Way_Unmatched_Balanced'):
            Expected_Samples = PBMC_3Way_Unmatched_Balanced
    verify_nested_cross_val_proteomics(work_dir, Expected_Samples, cv_k_outer, cv_k_inner)
    
    # ---------------------------------------------------------------------------------------------------------------- 
    # *****************************************************FS + Nested CV*********************************************
    # ---------------------------------------------------------------------------------------------------------------- 
    print("Data File: ", fname)
    # Remove all proteins that are not present in independent test dataset for liver tissue.
    if(tissue == 'LV'):
        X_test,Y_test,protein_names2,sample_order2 = generate_proteomic_test_data(impute, scaling, impute_thresh, log_transform)
    else:
        protein_names2 = []
    X,Y,protein_names,sample_order = generate_proteomic_matrices(fname, dataset, impute, scaling, impute_thresh, log_transform, protein_names2)
    
    # FEATURE SELECTION SECTION
    select_features_in_nested_CV_proteomics(X, Y, sample_order, protein_names, work_dir, out_dir, num_conditions, impute_thresh,
                                            cv_k_outer, cv_k_inner, features_to_gen, FS_Mode, Filter, Filter_Mode, std_treshold,
                                            DEG_Filter, DEG_Filter_Dir)
    
    # ML PORTION
    mean_test_accuracies = []
    for feature_size in feature_sizes:
        fs = feature_size
        mean_acc, conf, misclass = perform_nested_CV_proteomics(X, Y, sample_order, protein_names, work_dir, out_dir, feature_size, cv_k_outer,
                                                                cv_k_inner, model_name, balanced_acc)
        mean_test_accuracies.append(mean_acc)
        
    return mean_test_accuracies

def validate_in_test_data(root_dir:str, work_dir:str, out_dir:str, feature_size:int, cv_k:int, model_name:str, conditions:list,
                          dataset:str, impute:str, scaling:str, impute_thresh:float, FS_Test:int, log_transform:bool, balanced_acc:bool = False,
                          custom:bool = False):
    # Generate X, Y
    # Generate X_test, Y_test
    # Tune hyper-parameters over X, Y using CV
    # Generate feature set over X, Y using nested CV
    # Test in X_test, Y_test a model trained on X, Y with hyperameters and features found above. 
    assert(cv_k in [5])
    assert(FS_Test in [3,4,5])
    assert(scaling in [None, 'standard', 'minmax', 'normal', 'robust'])
    assert(impute in ['mean', 'median', 'kNN', 'zero'])
    assert(impute_thresh in [0, 0.05, 0.1, 0.2, 0.5, 1.0])
    assert(model_name in ['LR', 'SVM', 'kNN'])
    assert(dataset in ['LV_2Way', 'LV_3Way', 'LV_3Way_Matched', 'LV_3Way_Unmatched_Balanced'])
    
    fname = ''
    if(dataset in ['LV_2Way', 'LV_3Way', 'LV_3Way_Matched', 'LV_3Way_Unmatched_Balanced']):
        # fname = root_dir + 'AH_TMT_global_biopsy_protein_ratio_cleaned_BATCH_CORRECTED.csv'
        fname = root_dir + 'AH_TMT_global_biopsy_protein_ratio.csv'
    else:
        raise ValueError("Not yet defined!")
    
    X_test,Y_test,protein_names2,sample_order2 = generate_proteomic_test_data(impute, scaling, impute_thresh, log_transform)
    X,Y,protein_names,sample_order = generate_proteomic_matrices(fname, dataset, impute, scaling, impute_thresh, log_transform, protein_names2)
    
    root_dir1 = work_dir + 'Sample_Train'
    root_dir2 = out_dir + 'top_proteomic_features_set'
    best_hyper_param_dict = perform_CV_proteomics(X, Y, sample_order, protein_names, root_dir1, root_dir2, feature_size, cv_k, model_name)

    # Note, this section assumes that nested cv has already been ran and the top proteomics features files have been generated.
    if(not custom):
        proteins_to_read_2d = []
        ii = 0
        while ii < cv_k:
            feature_file = out_dir + 'top_proteomic_features_set' + str(ii) + '.txt'
            proteins_to_read = read_in_csv_file_one_column(feature_file, 0, '\t', 0, feature_size)
            proteins_to_read_2d.append(set(proteins_to_read))
            ii += 1
        proteins_to_read_2d_eval = count_elements_in_2dlist(proteins_to_read_2d)
        proteins_to_read = []
        for k,v in proteins_to_read_2d_eval.items():
            if v >= FS_Test:
                proteins_to_read.append(k)
        print("Features selected for independent dataset testing: ", proteins_to_read)
        print("Number of features selected: ", len(proteins_to_read))
    else:
        proteins_to_read = []
    
    # Some proteins present in our data are not present in test data. Remove these proteins.
    proteins_to_read_cleaned = []
    for protein_name in proteins_to_read:
        if protein_name in protein_names2:
            proteins_to_read_cleaned.append(protein_name)
            
    print("Features selected for independent dataset testing post cleaning: ", proteins_to_read_cleaned)
    print("Number of features selected post cleaning: ", len(proteins_to_read_cleaned))
            
    X_top = select_features_from_matrix(X, protein_names, proteins_to_read_cleaned)
    X_top2 = select_features_from_matrix(X_test, protein_names2, proteins_to_read_cleaned)
    
    if(model_name == 'SVM'):model = SVC(**best_hyper_param_dict)
    elif(model_name == 'kNN'):model = KNeighborsClassifier(**best_hyper_param_dict)
    elif(model_name == 'LR'):model = LogisticRegression(**best_hyper_param_dict)
    clf = model.fit(X_top, Y)
    #print(clf.coef_)
    
    # Write out model coeffecients to files in out directory if model is logistic regression. 
    # Note this could also work for SVM if it is a linear kernel.
    print(best_hyper_param_dict)
    if(model_name == 'LR' or (model_name == 'SVM' and best_hyper_param_dict['kernel'] == 'linear')):
        coefficients = clf.coef_
        print(coefficients)
        out_fname = out_dir + 'top_model_features.txt'
        with open(out_fname, 'w') as writer:
            iii = 0
            for feature in proteins_to_read_cleaned:
                avg_coef = 0
                jjj = 0
                if(len(conditions) > 2):
                    while jjj < len(conditions):
                        avg_coef += abs(coefficients[jjj][iii])
                        jjj += 1
                else:
                    avg_coef = coefficients[0][iii]
                avg_coef = avg_coef / len(conditions)
                writer.write(feature + '\t' + str(avg_coef) + '\n')
                iii += 1
    
    print("Sample Order in Test: ", sample_order2)
    Y_hat = clf.predict(X_top2)
    print("Y_hat: ", Y_hat)
    print("Y_ind_test: ", Y_test)
    acc = accuracy_score(Y_test, Y_hat)
    bacc = balanced_accuracy_score(Y_test, Y_hat)
    conf_matrix = confusion_matrix(Y_test, Y_hat)
    
    print("***********************************")
    print("Feature Size: ", feature_size)
    if(not balanced_acc):
        print("Independent Dataset Accuracy: ", acc)
    else:
        print("Independent Dataset Balanced Accuracy: ", bacc)
    print("Independent Dataset Confusion Matrix: ", conf_matrix)
    print("***********************************")
    
    misclass = {}
    test_samples = sample_order2
    num_test_samples = len(sample_order2)
    q = 0
    while q < num_test_samples:
        if(Y_hat[q] != Y_test[q]):
            # print("Misclassified: ", test_samples[q])
            try:
                misclass[test_samples[q]] = misclass[test_samples[q]] + 1
            except KeyError:
                misclass[test_samples[q]] = 1
        q += 1
    
    gen_conf_matrix(work_dir, dataset, np.array(conf_matrix), conditions, acc, bacc, model_name)
    plot_misclass(work_dir, dataset, misclass, model_name)
    #plot_feature_importance(out_dir, dataset, 1, cv_k, model_name, True, 20)
    
def get_trained_proteomics_classifier(root_dir:str, work_dir:str, out_dir:str, feature_size:int, cv_k:int, model_name:str, conditions:list,
                                      dataset:str, impute:str, scaling:str, impute_thresh:float, FS_Test:int, log_transform:bool, balanced_acc:bool = False,
                                      custom:bool = False):
    # Generate X, Y
    # Generate X_test, Y_test
    # Tune hyper-parameters over X, Y using CV
    # Generate feature set over X, Y using nested CV
    assert(cv_k in [5])
    assert(FS_Test in [3,4,5])
    assert(scaling in [None, 'standard', 'minmax', 'normal', 'robust'])
    assert(impute in ['mean', 'median', 'kNN', 'zero'])
    assert(impute_thresh in [0, 0.05, 0.1, 0.2])
    assert(model_name in ['LR', 'SVM', 'kNN'])
    assert(dataset in ['LV_3Way', 'LV_3Way_Matched', 'LV_3Way_Unmatched_Balanced',
                       'PBMC_3Way', 'PBMC_3Way_Matched', 'PBMC_3Way_Unmatched_Balanced'])
    
    fname = ''
    if(dataset in ['LV_3Way', 'LV_3Way_Matched', 'LV_3Way_Unmatched_Balanced']):
        # fname = root_dir + 'AH_TMT_global_biopsy_protein_ratio_cleaned_BATCH_CORRECTED.csv'
        fname = root_dir + 'AH_TMT_global_biopsy_protein_ratio.csv'
    elif(dataset in ['PBMC_3Way', 'PBMC_3Way_Matched', 'PBMC_3Way_Unmatched_Balanced']):
        fname = root_dir + 'AH_PBMC_masterproteinratiofirst.csv'
    else:
        raise ValueError("Not yet defined!")
    
    X,Y,protein_names,sample_order = generate_proteomic_matrices(fname, dataset, impute, scaling, impute_thresh, log_transform)
    
    #print("Sample Order in get_trained_proteomics_classifier: ", sample_order)
    
    root_dir1 = work_dir + 'Sample_Train'
    root_dir2 = out_dir + 'top_proteomic_features_set'
    best_hyper_param_dict = perform_CV_proteomics(X, Y, sample_order, protein_names, root_dir1, root_dir2, feature_size, cv_k, model_name)

    print("Proteomic model, best configuration:", best_hyper_param_dict)
    # Note, this section assumes that nested cv has already been ran and the top proteomics features files have been generated.
    if(not custom):
        proteins_to_read_2d = []
        ii = 0
        while ii < cv_k:
            feature_file = out_dir + 'top_proteomic_features_set' + str(ii) + '.txt'
            proteins_to_read = read_in_csv_file_one_column(feature_file, 0, '\t', 0, feature_size)
            proteins_to_read_2d.append(set(proteins_to_read))
            ii += 1
        proteins_to_read_2d_eval = count_elements_in_2dlist(proteins_to_read_2d)
        proteins_to_read = []
        for k,v in proteins_to_read_2d_eval.items():
            if v >= FS_Test:
                proteins_to_read.append(k)
    else:
        proteins_to_read = []
    
    #print("Features selected for independent dataset testing post cleaning: ", proteins_to_read)
            
    X_top = select_features_from_matrix(X, protein_names, proteins_to_read)
    
    if(model_name == 'SVM'):model = SVC(**best_hyper_param_dict)
    elif(model_name == 'kNN'):model = KNeighborsClassifier(**best_hyper_param_dict)
    elif(model_name == 'LR'):model = LogisticRegression(**best_hyper_param_dict)
    clf = model.fit(X_top, Y)
    #print(clf.coef_)
    
    # Write out model coeffecients to files in out directory if model is logistic regression. 
    if(model_name == 'LR'):
        coefficients = clf.coef_
        out_fname = out_dir + 'top_model_features.txt'
        with open(out_fname, 'w') as writer:
            iii = 0
            for feature in proteins_to_read:
                avg_coef = 0
                jjj = 0
                while jjj < len(conditions):
                    avg_coef += abs(coefficients[jjj][iii])
                    jjj += 1
                avg_coef = avg_coef / len(conditions)
                writer.write(feature + '\t' + str(avg_coef) + '\n')
                iii += 1
                
    plot_feature_importance(out_dir, dataset, 1, cv_k, model_name, True, 20)
    
    return clf, proteins_to_read
   
def parse_execution_log_proteomics(fdir:str):
    result = []
    
    with open(fdir + 'execution_log.txt') as f:
        lines = f.readlines()
        for line in lines:
            if(line.startswith("MEAN TEST ACCURACY")):
                newline = line.index('\n')
                accuracy = round(float(line[21:newline]), 2)
                result.append(accuracy)
                
    return result

def parse_execution_logs_proteomics():
    # Parsing execution logs to organize accuracies.
    classifiers = ['LR', 'kNN', 'SVM']
    filters = ['', 'Variance_2.5_', 'Variance_3.0_']
    imputations = ['median_0', 'median_0.05', 'median_0.1', 'zero_0', 'zero_0.05', 'zero_0.1']
    
    for classifier in classifiers:
        for filter_ in filters:
            table = []
    
            for imputation in imputations:
                
                fdir = 'C:/.../'
                fdir += 'LV_AH_CT_Excluded/LV_2Way_' + classifier + '_DE_' + filter_ + imputation + '/'
                accuracies =  parse_execution_log_proteomics(fdir)
                table.append(accuracies)
                
            filename = 'C:/.../'
            filename += 'LV_AH_CT_Excluded/LV_2Way_' + classifier + '_DE_' + filter_ + 'Summary.txt'
            with open(filename, 'w') as writer:
                i = 0
                while i < 11:
                    j = 0
                    to_write = ''
                    for column in table:
                        print('Row: ', str(i), '; Column: ', str(j))
                        if(j < 5):
                            to_write += (str(column[i]) + ',')
                        else:
                            to_write += (str(column[i]) + '\n')
                        j += 1
                    writer.write(to_write)
                    i += 1
                
def parse_agotool_log(fname:str):
    result = []
    result2 = []
    
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            if(len(line) < 10 and len(line) > 4):
                newline = line.index('\n')
                relevance = line[0:newline]
                result.append(relevance)
            elif(line.startswith('Number of features:')):
                newline = line.index('\n')
                fsize = line[20:newline]
                result2.append(fsize)
                
    return result,result2
    
def parse_agotool_logs():
    impute_tresholds = ['median_0', 'median_0.05', 'median_0.1']
    ftests = ['agotool_validation_3.txt', 'agotool_validation_4.txt', 'agotool_validation_5.txt']
    filters = ['', 'Variance_2.5_', 'Variance_3.0_']
    
    for ftest in ftests:
        for filter_ in filters:
    
            table = ['15', '25', '35', '50', '60', '70', '80', '90', '100', '150', '200']
            results = []
            results2 = []
            
            for impute_tresh in impute_tresholds:
            
                fname = 'C:/.../'
                fname += 'LV_AH_CT_Excluded/LV_2Way_SVM_DE_' + filter_ + impute_tresh + '/' + ftest
                
                result,result2 = parse_agotool_log(fname)
                
                results.append(result)
                results2.append(result2)
                
            i = 0
            while i < 11:
                table[i] += ' - ' + results2[0][i] + '/' + results2[1][i] + '/' + results2[2][i] + ',' 
                table[i] += results[0][i] + ',' + results[1][i] + ',' + results[2][i]
                i += 1 
            
            filename = 'C:/.../'
            filename += 'LV_AH_CT_Excluded/LV_2Way_SVM_DE_' + filter_ + ftest
            
            with open(filename, 'w') as writer:
                i = 0
                while i < 11:
                    writer.write(table[i])
                    writer.write('\n')
                    i += 1

class TestProteomicCodeBase(unittest.TestCase):   
    
    def test_generate_top_DE_proteins(self):
        fname = 'C:/.../AH_TMT_global_biopsy_protein_ratio.csv'
        X,Y,protein_names,study_ids1 = generate_proteomic_matrices(fname, 'LV_3Way', 'zero', None, 1.0)
        
        fname = 'C:/.../Sample_Train1_1_DE.csv'
        fname_out = 'C:/.../LV_3Way_Proteomics_DE_.txt'
        num_features = 5
        result = generate_top_DE_proteins(fname, num_features, protein_names, fname_out, 3, 0)
        
        self.assertEqual(result[0], ('GSTA1_HUMAN', 12.559916274202))
        self.assertEqual(result[1], ('K2C80_HUMAN', 6.53438596491228))
        self.assertEqual(result[2], ('ALBU_HUMAN', 7.59139784946237))
        self.assertEqual(result[3], ('GSTM4_HUMAN', 9.92846270928463))
        self.assertEqual(result[4], ('DR4L2_HUMAN', 5.42304526748971))
        
    
    def test_generate_CV_split_proteomics(self):
        cv_k = 5
        work_dir = 'C:/.../'
        
        sample_order = ['...'] #53
        
        fnames = []
        z = 0
        while z < cv_k:
            fname_in = work_dir + 'Sample_Train' + str(z+1) + '.csv'
            fnames.append(fname_in)
            z += 1
        
        cv_split = generate_CV_split_proteomics(cv_k, fnames, sample_order)
        
        cv_split_expected = ([1,9,14,22,30,38,47,15,42,25,
                             44,2,10,18,23,31,39,48,16,50,26,52,
                             3,11,19,27,32,40,49,24,51,34,53,4,
                             12,20,28,36,45,6,33,8,35],[0,5,7,13,17,21,29,37,41,43,46])
        
        self.assertEqual(cv_split[0][0], cv_split_expected[0])
        self.assertEqual(cv_split[0][1], cv_split_expected[1])
    
    def test_generate_proteomic_sample_files_from_cuffdiff(self):
        root_dir = 'C:/.../'
        out_dir = 'C:/.../'
        generate_proteomic_sample_files_from_cuffdiff(root_dir, out_dir, 'LV')
        
        samples_expected_all = ['...']
        samples_expected_fold1 = ['...']
        samples_expected_fold1_5 = ['...']
        samples_expected_fold2 = ['...']
        
        samples_all = read_in_csv_file_one_column(out_dir + 'Samples_ALL.csv', 0, ',')
        samples_fold1 = read_in_csv_file_one_column(out_dir + 'Sample_Train1.csv', 0, ',')
        samples_fold1_5 = read_in_csv_file_one_column(out_dir + 'Sample_Train1_5.csv', 0, ',')
        samples_fold2 = read_in_csv_file_one_column(out_dir + 'Sample_Train2.csv', 0, ',')
        
        self.assertEqual(samples_all, samples_expected_all)
        self.assertEqual(samples_fold1, samples_expected_fold1)
        self.assertEqual(samples_fold1_5, samples_expected_fold1_5)
        self.assertEqual(samples_fold2, samples_expected_fold2)
        
    def test_generate_proteomic_counts_means(self):
        fname = 'C:/.../AH_TMT_global_biopsy_protein_ratio.csv'
        fname3 = 'C:/.../AH_TMT_global_biopsy_protein_ratio_cleaned_BATCH_CORRECTED.csv'
        fname2 = 'C:/.../Validation_Data.csv'
        fname4 = 'C:/.../Validation_Data_No_Batch_Correction.csv'
        fname5 = 'C:/.../AH_PBMC_masterproteinratiofirst.csv'
        counts, _ = generate_proteomic_counts_means(fname, 'LV_3Way_Old')
        self.assertEqual(counts['1433B_HUMAN'], [0.9102941176470588, 1.0510000000000002, 1.299])
        self.assertEqual(counts['1433E_HUMAN'], [0.8591176470588234, 1.4760000000000002, 1.341])
        counts, _ = generate_proteomic_counts_means(fname5, 'PBMC_3Way_Matched')
        self.assertEqual(counts['1433F_HUMAN'], [0.7621784397222222, 1.0677605042105263, 0.8396090766923078])
        
    def test_generate_proteomic_matrices(self):
        fname = 'C:/.../AH_TMT_global_biopsy_protein_ratio.csv'
        fname3 = 'C:/.../AH_TMT_global_biopsy_protein_ratio_cleaned_BATCH_CORRECTED.csv'
        fname2 = 'C:/.../Validation_Data.csv'
        fname4 = 'C:/.../Validation_Data_No_Batch_Correction.csv'
        fname5 = 'C:/.../AH_PBMC_masterproteinratiofirst.csv'
        X,Y,protein_names,study_ids1 = generate_proteomic_matrices(fname, 'LV_3Way_Old', 'zero', None, 0)
        
        expected_protein_names = ['...']
        expected_study_ids = ['...']
        X_expected = np.array([[0.41, 0.46],[1.55, 0.74],[0.2,  0.25],[0.93, 0.89],[0.76, 0.72],[0.87, 1.  ],[1.4,  1.52],
                               [0.85, 1.29],[1.48, 1.36],[1.17, 1.17],[1.13, 1.18],[0.45, 0.47],[1.13, 1.34],[1.12, 1.06],
                               [1.13, 1.15],[0.87, 1.1 ],[0.98, 1.33],[1.6,  1.36],[0.97, 0.8 ],[0.82, 0.57],[0.89, 0.69],
                               [0.99, 0.85],[0.89, 0.98],[0.98, 0.86],[0.89, 1.43],[1.1,  1.17],[1.15, 1.24],[0.9,  0.8 ],
                               [0.38, 0.41],[0.78, 0.69],[0.66, 0.72],[0.98, 0.96],[1.01, 1.1 ],[1.08, 1.55],[1.07, 1.12],
                               [1.14, 1.19],[1.2,  0.98],[1.45, 1.09],[0.67, 0.91],[0.73, 0.84],[0.85, 0.85],[1.03, 1.5 ],
                               [1.12, 1.67],[1.36, 1.61],[1.53, 1.53],[0.89, 0.37],[1.09, 0.66],[0.95, 1.09],[1.12, 1.33],
                               [0.9 , 1.23],[1.25, 1.87],[1.04, 1.5 ],[1.27, 1.38],[1.29, 1.45]])
        
        X_expected = np.round(X_expected, 5)
        X = np.round(X, 5)
        
        self.assertEqual(protein_names[0:10], expected_protein_names)
        self.assertEqual(study_ids1, expected_study_ids)
        X_top = X[:,0:2]
        i = 0
        for row in X_top:
            j = 0
            for ele in row:
                self.assertEqual(ele, X_expected[i,j])
                j += 1
                
            i += 1
            
        X,Y,protein_names,study_ids1 = generate_proteomic_matrices(fname, 'LV_3Way_Matched', 'median', None, 1.0)
        X_expected = np.array([[0.54, 0.33],[0.75, 0.79],[0.31, 0.23],[0.67, 0.77],[0.51, 1.06],[1.21, 1.21],[2.17, 1.49],[1.2,  2.18],
                               [0.63, 1.12],[0.75, 0.88],[1.21, 0.77],[0.69, 1.56],[1.04, 0.96],[1.1,  0.87],[0.78, 1.12],[0.72, 0.93],
                               [0.84, 1.07],[3.11, 1.06],[0.87, 0.94],[0.87, 0.4 ],[0.87, 0.62],[0.87, 0.81],[0.87, 1.05],[0.87, 1.1 ],
                               [0.98, 0.91],[2,   1.28],[0.95, 0.58],[0.87, 0.54],[0.92, 1.69],[0.95, 1.64],[0.87, 0.45],[0.53, 0.6 ],
                               [0.67, 1.08],[2.74, 1.09],[0.84, 1.  ],[0.7,  1.52],[1.78, 1.59]])
        
        X_expected = np.round(X_expected, 5)
        X = np.round(X, 5)
        
        X_top = X[:,4:6]
        i = 0
        for row in X_top:
            j = 0
            for ele in row:
                self.assertEqual(ele, X_expected[i,j])
                j += 1
                
            i += 1
        
        
    def test_generate_proteomic_counts_individual(self):
        fname = 'C:/.../AH_TMT_global_biopsy_protein_ratio.csv'
        fname3 = 'C:/.../AH_TMT_global_biopsy_protein_ratio_cleaned_BATCH_CORRECTED.csv'
        fname2 = 'C:/.../Validation_Data.csv'
        fname4 = 'C:/.../Validation_Data_No_Batch_Correction.csv'
        fname5 = 'C:/.../AH_PBMC_masterproteinratiofirst.csv'
        
        counts, _ = generate_proteomic_counts_individual(fname, 'LV_3Way')
        self.assertEqual(counts['...']['1433B_HUMAN'], 0.89)
        self.assertEqual(counts['...']['1433E_HUMAN'], 0.37)
    
    def test_generate_proteomic_test_data(self):
        X,Y,protein_names,study_ids1 = generate_proteomic_test_data('zero', None, 0)
        
        expected_protein_names = ['1433B_HUMAN', '1433E_HUMAN', '1433F_HUMAN', '1433G_HUMAN', '1433S_HUMAN', '1433T_HUMAN',
                                  '1433Z_HUMAN', '1A01_HUMAN', '1A02_HUMAN', '1A23_HUMAN']
        expected_study_ids = ['...']
        expected_X = np.array([[1.25809577, 1.20448867],[1.46206364, 1.06615073],[1.55297681, 1.12237227],[1.20760171, 1.02655792],
                               [1.34800166, 1.09892448],[1.61056651, 0.95999903],[0.85833222, 0.9395273 ],[1.01768196, 1.23902804],
                               [0.97309748, 1.21108108],[0.95467545, 1.20948314],[0.89391728, 1.24320878],[1.00846945, 1.0558559 ],
                               [0.95005932, 1.03747459],[0.7688476,  1.00700831],[0.72447275, 0.90147464],[0.63851227, 0.92283016],
                               [0.74794181, 0.57738404],[0.90678879, 1.04125706]])
        
        expected_X = np.round(expected_X, 5)
        X = np.round(X, 5)
        
        self.assertEqual(protein_names[0:10], expected_protein_names)
        self.assertEqual(study_ids1, expected_study_ids)
        X_top = X[:,0:2]
        i = 0
        for row in X_top:
            j = 0
            for ele in row:
                self.assertEqual(ele, expected_X[i,j])
                j += 1
                
            i += 1
            
        X,Y,protein_names,study_ids1 = generate_proteomic_test_data('zero', None, 1.0)
        
        expected_X = np.array([[0,         0.81714753],[0,         1.0654919 ],[0,         0.86255224],[1.1694139,  1.82676895],
                               [1.30890445, 1.18681333],[1.16430573, 1.4413042 ],[0,         0.8751522 ],[0,         1.31037075],
                               [0,         1.03969069],[0.90619089, 0.98832019],[1.6411084,  1.16925307],[0,         0.67407248],
                               [0,         0.77160057],[0,         1.14186984],[0.88310317, 0.82462391],[0.61801879, 0.70579816],
                               [0.78183152, 0.83127226],[2.12708484, 0.99286658]])
        
        expected_X = np.round(expected_X, 5)
        X = np.round(X, 5)
        
        X_top = X[:, 11:13]
        i = 0
        for row in X_top:
            j = 0
            for ele in row:
                self.assertEqual(ele, expected_X[i,j])
                j += 1
                
            i += 1
            
        X,Y,protein_names,study_ids1 = generate_proteomic_test_data('median', None, 1.0)
        
        expected_X = np.array([[1.16430573, 0.81714753],[1.16430573, 1.0654919 ],[1.16430573, 0.86255224],[1.1694139,  1.82676895],
                               [1.30890445, 1.18681333],[1.16430573, 1.4413042 ],[1.16430573, 0.8751522 ],[1.16430573, 1.31037075],
                               [1.16430573, 1.03969069],[0.90619089, 0.98832019],[1.6411084 , 1.16925307],[1.16430573, 0.67407248],
                               [1.16430573, 0.77160057],[1.16430573, 1.14186984],[0.88310317, 0.82462391],[0.61801879, 0.70579816],
                               [0.78183152, 0.83127226],[2.12708484, 0.99286658]])
        
        expected_X = np.round(expected_X, 5)
        X = np.round(X, 5)
        
        X_top = X[:, 11:13]
        i = 0
        for row in X_top:
            j = 0
            for ele in row:
                self.assertEqual(ele, expected_X[i,j])
                j += 1
                
            i += 1
        
        
def test_suite():
    # Unit Tests
    unit_test_suite = unittest.TestSuite()
    unit_test_suite.addTest(TestProteomicCodeBase('test_generate_proteomic_sample_files_from_cuffdiff'))
    unit_test_suite.addTest(TestProteomicCodeBase('test_generate_proteomic_test_data'))
    unit_test_suite.addTest(TestProteomicCodeBase('test_generate_proteomic_matrices'))
    unit_test_suite.addTest(TestProteomicCodeBase('test_generate_proteomic_counts_means'))
    unit_test_suite.addTest(TestProteomicCodeBase('test_generate_proteomic_counts_individual'))
    unit_test_suite.addTest(TestProteomicCodeBase('test_generate_CV_split_proteomics'))
    unit_test_suite.addTest(TestProteomicCodeBase('test_generate_top_DE_proteins'))    
    
    # Performed a trace of perform_nested_CV_proteomics to verify correctness.
    
    runner = unittest.TextTestRunner()
    runner.run(unit_test_suite)

if __name__ == "__main__":
    # MUST BE PYTHON 3.7+ since, DICTIONARIES are ASSUMED to be ORDERED throughout the codebase.
    assert sys.version_info.major == 3
    assert sys.version_info.minor >= 7  
    
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600
    
    #parse_execution_logs_proteomics()
    #parse_agotool_logs()
    
    # fname = 'C:/.../'
    # fname += 'results_mapped.tsv'
    
    # fname2 = 'C:/...'
    # fname2 += '/.../result.txt'
    
    # DEGs = read_in_csv_file_one_column(fname2, 0, '\t', 2)
    # DEPs_as_genes = read_in_csv_file_one_column(fname, 1, '\t', 1)
    
    # print(len(DEPs_as_genes))
    # print(len(DEGs))
    # print(len(set(DEGs)&set(DEPs_as_genes)))
    
    # i = 0
    # prob = 0
    # g = 16
    # p = 33
    # DEPs = 876
    # DEGs = 971
    # ov = 88
    # n = 1
    # # math.comb(p,i)
    # while i <= g:
    #     prob += ((DEGs-ov)/DEGs)**(g-i) * (ov/DEGs)**i * math.comb(g,i) * ((DEPs-i)/DEPs)**(p-n) * (i/DEPs)**n * math.comb(p, n)
    #     i += 1
    
    # print(prob)
    
    # test_suite()
    
    '''
    in_dir = 'C:/.../'
    fname1 = 'liver_proteins_genes.tsv'
    fname2 = 'pbmc_proteins_genes.tsv'
    
    fname3 = 'C:/.../'
    fname3 += '.../gene_exp.diff'
    
    fname4 = 'C:/.../'
    fname4 += '.../gene_exp.diff'
    
    fname5 = 'C:/../'
    fname5 += '.../Liver_Top_Protein_Genes_Set0.tsv'
    
    fname6 = 'C:/.../'
    fname6 += '.../'
    fname6 += 'top_rnaseq_features_set0.txt'
    
    liver_protein_genes_sig = read_in_csv_file_one_column(fname5, 1, '\t', 1)
    liver_genes_sig = read_in_csv_file_one_column(fname6, 0, '\t', 1)
    
    print(len(liver_protein_genes_sig))
    print(len(liver_genes_sig))
    
    print(len(set(liver_protein_genes_sig)&set(liver_genes_sig)))
    
    liver_protein_genes = read_in_csv_file_one_column(in_dir + fname1, 1, '\t', 1)
    pbmc_protein_genes = read_in_csv_file_one_column(in_dir + fname2, 1, '\t', 1)

    liver_protein_genes = set(liver_protein_genes)
    pbmc_protein_genes = set(pbmc_protein_genes)
    
    print(len(liver_protein_genes))
    print(len(pbmc_protein_genes))
    
    genes = read_in_csv_file_one_column(fname3, 0, '\t', 1)

    genes = set(genes)
    
    print(len(genes))
    
    liver_inter = genes & liver_protein_genes
    pbmc_inter = genes & pbmc_protein_genes
    
    print(len(liver_inter))
    print(len(pbmc_inter))
    
    a = list(genes - liver_protein_genes)
    a.sort()
    print(a)
    '''
    
    
    
    
    '''
    # Setting selection using IG as feature selection.
    fname = 'C:/.../AH_TMT_global_biopsy_protein_ratio.csv'
    dataset = 'LV_3Way'
    impute = 'median'
    impute_thresh = 1.0
    scaling = None
    log_transform = False
    
    out_dir = 'C:/.../'
    
    X,Y,protein_names,sample_order = generate_proteomic_matrices(fname, dataset, impute, scaling, impute_thresh, log_transform)
    
    num_features = 50
    num_runs = 3
    FS_Filter = None
    Filter_Threshold = 3.0
    
    perform_nested_cv(X, Y, protein_names, sample_order, 'LR', ['AH','CT','AC'], 5, 5, num_features, num_runs, 'IG', out_dir, 'LV_3Way',
                      FS_Filter, Filter_Threshold, False)
    '''

    '''
    dataset = 'LV_2Way'
    tissue = 'LV'
    k_inner = 5
    k_outer = 5
    log_transform = False
    scaling = None
    bacc = False
    DEG_Filter = False
    #DEG_Filter_Dir = 'C:/.../'
    #DEG_Filter_Dir += '.../DEG_to_DEP/'
    DEG_Filter_Dir = ''
    conditions = ['AH', 'CT']
    feature_sizes = [15, 25, 35, 50, 60, 70, 80, 90, 100, 150, 200]
    #feature_sizes = [35]
    filter_threshold = 3.0
    root_dir = 'C:/.../'
    work_dir = 'C:/.../'
    for FS_Method in ['DE']:
        for FS_Filter in [None]:
            for impute in ['median']:
                for impute_thresh in [0]:
                    for classifier in ['LR']:
                        #for filter_threshold in [2.5, 3.0]:
                            
                        dir_name = dataset + '_' + classifier + '_' + FS_Method# + '_' + str(FS_Filter) + '_' + str(filter_threshold)
                        dir_name += '_' + impute + '_' + str(impute_thresh)
                        # dir_name += '_FS_Test' + str(FS_Test)
                        dir_name += '/'
                            
                        # out_dir = work_dir + dir_name
                        out_dir = "C:/.../" + dir_name
                        try:
                            os.makedirs(out_dir)
                        except FileExistsError:
                            pass
                        
                        log_file = out_dir + 'execution_log.txt'
                        console = sys.stdout
                        sys.stdout = open(log_file, 'w')
                        
                        print("root_dir: ", root_dir)
                        print("out_dir: ", out_dir)
                        print("Tissue: ", tissue)
                        print("dataset: ", dataset)
                        print("classifier: ", classifier)
                        print("conditions: ", conditions)
                        print("k_outer: ", k_outer)
                        print("k_inner: ", k_inner)
                        print("feature sizes: ", feature_sizes)
                        print("Feature Selection Method: ", FS_Method)
                        print("Feature Selection Filter: ", FS_Filter)
                        print("Balanced Accuracy: ", bacc)
                        print("Scaling: ", scaling)
                        print("Imputation Method: ", impute)
                        print("Imputation Treshold: ", impute_thresh)
                        print("Log Transform: ", log_transform)
                        print('\n')
                        
                        classify_with_nested_CV_proteomics(root_dir, work_dir, out_dir, dataset, classifier, conditions, k_outer,
                                                           k_inner, 500, feature_sizes, FS_Method, bool(FS_Filter), FS_Filter, bacc,
                                                           scaling, impute, log_transform, impute_thresh, tissue, filter_threshold,
                                                           DEG_Filter, DEG_Filter_Dir)
                            
                        sys.stdout.close()
                        sys.stdout = console
    '''

    '''
    # Proteomics Biological Validation
    filter_threshold = '3.0'
    # Note, this section assumes that nested cv has already been ran and the top proteomics features files have been generated.
    out_dir = 'C:/.../LV_2Way_'
    for classifier in ['SVM']:
        for FS_Method in ['DE']:
            for FS_Filter in [None]:
                for Impute in  ['median']:
                    for Impute_Thresh in ['0']:
                        work_dir = out_dir + classifier + '_' + FS_Method + '_' + Impute + '_' + Impute_Thresh + '/'
                        # work_dir = out_dir + classifier + '_' + FS_Method + '_' + str(FS_Filter) + '_' + filter_threshold
                        # work_dir += '_' + Impute + '_' + Impute_Thresh + '/'
                        cv_k = 5
                        feature_sizes = [35]
                        FS_Tests = [4]
                        #feature_sizes = [15, 25, 35, 50, 60, 70, 80, 90, 100, 150, 200]
                        #FS_Tests = [3,4,5]
                        
                        for FS_Test in FS_Tests:
                            for feature_size in feature_sizes:
                            
                                proteins_to_read_2d = []
                                ii = 0
                                while ii < cv_k:
                                    feature_file = work_dir + 'top_proteomic_features_set' + str(ii) + '.txt'
                                    proteins_to_read = read_in_csv_file_one_column(feature_file, 0, '\t', 0, feature_size)
                                    proteins_to_read_2d.append(set(proteins_to_read))
                                    ii += 1
                                proteins_to_read_2d_eval = count_elements_in_2dlist(proteins_to_read_2d)
                                proteins_to_read = []
                                for k,v in proteins_to_read_2d_eval.items():
                                    if v >= FS_Test:
                                        proteins_to_read.append(k)
                                print("Features selected for in-silico biological validation: ", proteins_to_read)
                                print("Number of features selected: ", len(proteins_to_read))
                                     
                                r1,r2 = biological_validation_proteomics(proteins_to_read, 'Pathway')
                                r3,r4 = biological_validation_proteomics(proteins_to_read, 'Tissue')
                                r5,r6 = biological_validation_proteomics(proteins_to_read, 'Disease')
                                
                                # with open(work_dir + 'agotool_validation_' + str(FS_Test) + '.txt', 'a') as writer:
                                #     writer.write('Feature Size: ' + str(feature_size) + '\n')
                                #     writer.write('Number of features: ' + str(len(proteins_to_read)) + '\n')
                                #     writer.write('Number of hits Pathway/Tissue/Disease:\n')
                                #     writer.write(str(r2.shape[0]) + '/' + str(r4.shape[0]) + '/' + str(r6.shape[0]) + '\n')
    '''
    
    
    # root_dir = 'C:/.../'
    # work_dir = 'C:/.../'
    # out_dir = 'C:/.../'
    # validate_in_test_data(root_dir, work_dir, out_dir, 35, 5, 'kNN', ['AH','CT'], 'LV_2Way', 'median', None, 0, 4, False)
    
    # Manual Biological Validation
    protein_list = ['...']
    r1, r2 = biological_validation_proteomics(protein_list, 'Pathway')
    
    # Code to generate proteomic count heatmaps.
    # PER CONDITION HEATMAP
    '''
    fname = 'C:/.../AH_TMT_global_biopsy_protein_ratio.csv'
    fname3 = 'C:/.../AH_TMT_global_biopsy_protein_ratio_cleaned_BATCH_CORRECTED.csv'
    fname2 = 'C:/.../Validation_Data.csv'
    fname4 = 'C:/.../Validation_Data_No_Batch_Correction.csv'
    fname5 = 'C:/.../AH_PBMC_masterproteinratiofirst.csv'
    out_dir = 'C:/.../Datasets_3Way/'
    
    counts, cond_order = generate_proteomic_counts_means(fname4, 'Validation')
    
    counts2 = {}
    proteins = ['...']
    
    # Re-order counts to be in same order as the protein list.
    for protein in proteins:
        counts2[protein] = counts[protein]
            
    plot_per_condition_counts_heatmap(counts2, proteins, cond_order, out_dir, 'Validation', 28)
    
    # PER SAMPLE HEATMAP
    counts3, protein_names = generate_proteomic_counts_individual(fname4, 'Validation')
    counts4 = {}
    for protein in proteins:
        for k,v in counts3.items():
            try:
                counts4[k].append(v[protein])
            except KeyError:
                counts4[k] = [v[protein]]
            
    # Re-arrange samples to be grouped by condition.
    counts5 = {}
    #for sample in (PBMC_3Way_AC_Unmatched_Balanced + PBMC_3Way_CT_Unmatched_Balanced + PBMC_3Way_AH_Unmatched_Balanced):
    #for sample in (LV_3Way_Unmatched_AC_Balanced + LV_3Way_Unmatched_CT_Balanced + LV_3Way_Unmatched_AH_Balanced):
    #for sample in (LV_3Way_CT + LV_3Way_AH):
    for sample in validation_rev:
        counts5[sample] = counts4[sample]
        
    plot_per_sample_counts_heatmap(counts5, proteins, out_dir, 'Validation', 28)
    '''
    
    # Generate custom confusion matrix
    # dataset = 'PBMC 3-Way Matched Balanced Integrated'
    # conditions = ['AH', 'CT', 'AC']
    # conf_matrix = [[9, 0, 0],[0, 11, 1],[0, 4, 2]]
    # accuracy = 0.81
    # baccuracy = 0.75
    # model_name = 'LR'
    # work_dir = 'C:/.../'
    # gen_conf_matrix(work_dir, dataset, np.array(conf_matrix), conditions, accuracy, baccuracy, model_name)
    
    # Code to generate proteomic sample to training set distribution.
    # main_dir = 'C:/.../'
    # generate_proteomic_sample_files(LV_3Way_AH, LV_3Way_Dict, main_dir)
    # generate_proteomic_sample_files(LV_3Way_CT, LV_3Way_Dict, main_dir)
    # generate_proteomic_sample_files(LV_3Way_Unmatched_AC_Balanced, LV_3Way_Unmatched_Dict_Balanced, main_dir)
    
    # Code to generate proteomic samples to training set distribution from existing Cuffdiff distribution.
    # PB_3Way_Root_Dir = 'C:/.../'
    # LV_3Way_Root_Dir = 'C:/.../'
    # LV_3Way_Out_Dir = 'C:/.../'
    # PB_3Way_Out_Dir = 'C:/.../'
    # generate_proteomic_sample_files_from_cuffdiff(PB_3Way_Root_Dir, PB_3Way_Out_Dir, 'PB')