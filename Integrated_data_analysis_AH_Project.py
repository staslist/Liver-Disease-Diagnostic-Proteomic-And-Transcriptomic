# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 10:44:11 2022

@author: staslist
"""

from sklearn.inspection import permutation_importance

# RNA-seq IDs

samples_AH_PB = ['...']
        
samples_CT_PB = ['...']

samples_DA_PB = ['...']

samples_AA_PB = ['...']

samples_PBMC_3Way = samples_AH_PB + samples_CT_PB + samples_DA_PB + samples_AA_PB


samples_AH_PB_Proteomic_RNAseq_Matched = ['...']
samples_CT_PB_Proteomic_RNAseq_Matched = ['...']
samples_AC_PB_Proteomic_RNAseq_Matched = ['...']

samples_AC_PB_Proteomic_RNAseq_Matched_New = ['...']

samples_AH_LV_Excluded = ['...']

samples_CT_LV = ['...']

samples_AC_LV = ['...']

samples_LV_3Way = samples_AH_LV_Excluded + samples_CT_LV + samples_AC_LV


samples_AH_LV_Proteomic_RNAseq_Matched = ['...']

samples_AH_LV_Proteomic_RNAseq_Matched_New = ['...']

samples_CT_LV_Proteomic_RNAseq_Matched = ['...']

samples_AC_LV_Proteomic_RNAseq_Matched = ['...']


from proteomic_data_analysis_AH_Project import *
from AH_Project_Codebase_SL import *

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import RocCurveDisplay


def classify_matched_with_unmatched_models_ensembl(root_dir_RNA:str, work_dir_RNA:str, model_RNA:str, FS_Mode_RNA:str,
                                                   filter_RNA:bool,filter_mode_RNA:str, feature_size_RNA:int,
                                                   std_thresh_RNA:float, FS_Test_RNA:int, root_dir_prot:str,
                                                   work_dir_prot:str, out_dir_prot:str, feature_size_prot:int,
                                                   dataset_prot:str, impute_prot:str, scaling_prot:str,
                                                   impute_thresh_prot:str, FS_Test_Prot:int, 
                                                   log_transform_prot:bool, tissue:str = 'LV', integrate:bool = True):
    # Assume the models have been trained on entire unmatched data with best features / hyper-parameters. 
    assert FS_Mode_RNA in ['DE', 'IG']
    
    if(FS_Mode_RNA == 'IG'):
        fpkm_fs = False
        fpkm_ml = False
    else:
        fpkm_fs = True
        fpkm_ml = False
    
    model_name_prot = 'SVM'
    num_conditions = 3
    cv_k = 5
    features_to_generate = 500
    balanced_accuracy = False
    conditions = ['AH', 'CT', 'AC']
    
    assert filter_mode_RNA in ['Variance', 'Union', 'Hybrid']
    assert tissue in ['LV', 'PBMC']
    
    rna_clf, genes_to_read = get_trained_RNAseq_classifier(root_dir_RNA, work_dir_RNA, model_RNA, FS_Mode_RNA,
                                                           num_conditions, cv_k, features_to_generate, filter_RNA,
                                                           filter_mode_RNA, balanced_accuracy, feature_size_RNA, 
                                                           fpkm_fs = fpkm_fs, fpkm_ml = fpkm_ml,
                                                           std_treshold = std_thresh_RNA, FS_Test = FS_Test_RNA)
    
    # print("RNAseq classifier finished.")
    
    prot_clf, proteins_to_read = get_trained_proteomics_classifier(root_dir_prot, work_dir_prot, out_dir_prot,
                                                                   feature_size_prot, cv_k, model_name_prot, conditions,
                                                                   dataset_prot, impute_prot, scaling_prot,
                                                                   impute_thresh_prot, FS_Test_Prot, log_transform_prot)
    
    print("Selected Genes: ", len(genes_to_read))
    print("Selected Proteins: ", len(proteins_to_read))
    
    
    if(tissue == 'LV'):
        # Now read in X and Y for matched samples only from both rnaseq and proteomics sides.
        input_dir_RNA = 'C:/.../'
        fname_prot = root_dir_prot + 'AH_TMT_global_biopsy_protein_ratio.csv'
        X, Y, rep_to_cond_map, gene_names, sample_order = generate_X_Y_from_cuffdiff(input_dir_RNA + 'Cuffdiff_GEOM_POOL/', 'ALL', True, True)
        X2, Y2, protein_names, sample_order2 = generate_proteomic_matrices(fname_prot, 'LV_3Way_Matched', impute_prot,
                                                                           scaling_prot, impute_thresh_prot)
    elif(tissue == 'PBMC'):
        input_dir_RNA = 'C:/.../'
        fname_prot = root_dir_prot + 'AH_PBMC_masterproteinratiofirst.csv'
        # print('Input Directory RNA: ', input_dir_RNA)
        X, Y, rep_to_cond_map, gene_names, sample_order = generate_X_Y_from_cuffdiff(input_dir_RNA + 'Cuffdiff_GEOM_POOL/', 'ALL', True, True)
        X2, Y2, protein_names, sample_order2 = generate_proteomic_matrices(fname_prot, 'PBMC_3Way_Matched_Balanced', impute_prot,
                                                                           scaling_prot, impute_thresh_prot)
    
    X = select_features_from_matrix(X, gene_names, genes_to_read)
    X2 = select_features_from_matrix(X2, protein_names, proteins_to_read)
    
    sample_order_trimmed = []
    for sample in sample_order:
        temp = sample.index('.')
        sample_trimmed = sample[0:temp]
        sample_order_trimmed.append(sample_trimmed)
    
    sample_order3 = []
    for sample in sample_order2:
        if(tissue == 'LV'):
            sample_translated = LV_3Way_Proteomic_to_RNAseq_New[sample]
        elif(tissue == 'PBMC'):
            sample_translated = PBMC_3Way_Proteomic_to_RNAseq_New[sample]
        sample_order3.append(sample_translated)
    # Also uses this as an oportunity to re-order RNAseq samples to match proteomics samples order.
    # Proteomic samples should be correct right away. 

    prot_samples_in_RNA_samples_indeces = []
    for sample in sample_order3:
        index = sample_order_trimmed.index(sample)
        prot_samples_in_RNA_samples_indeces.append(index)
    
    X = X[prot_samples_in_RNA_samples_indeces]
    Y = Y[prot_samples_in_RNA_samples_indeces]
    sample_order = np.array(sample_order_trimmed)[prot_samples_in_RNA_samples_indeces]
    
    # print(X2.shape)
    # print(X.shape)
    
    #print("Proteomics Sample Order in classify_matched_with_unmatched_models_ensembl: ", sample_order2)
    #print("RNAseq Sample Order in classify classify_matched_with_unmatched_models_ensembl: ", sample_order)
    
    if(integrate):
        if(tissue == 'LV'):
        # Select 2/3 of "balanced matched" samples. 16AH, 2CT, 2AC
            train_test_split = [[[0,1,2,3,4,5,7,8,9,10,11,12,13,18,19,21,22,23,24,28], [25,26,27,29,30,31,32,33,34,35]],
                                [[10,12,13,18,19,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35], [0,1,2,3,4,5,7,8,9,11]],
                                [[0,1,2,3,4,5,7,8,9,11,25,26,27,29,30,31,32,33,34,35], [10,12,13,18,19,21,22,23,24,28]]]
        elif(tissue == 'PBMC'):
            train_test_split = [[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,17,19,22], [14,16,18,20,21,23,24,25,26]], 
                                [[6,7,9,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26], [0,1,2,3,4,5,8,10,12]],
                                [[0,1,2,3,4,5,8,10,12,14,16,18,20,21,23,24,25,26], [6,7,9,11,13,15,17,19,22]]]
        
        accuracies = []
        conf_matrices = []
        AUCs = []
        AUCs_AH = []
        AUCs_CT = []
        AUCs_AC = []
        AUCs_class = {'AH': AUCs_AH, 'CT': AUCs_CT, 'AC': AUCs_AC}
        #AUCs2 = []
        
        tprs = []
        tprs_AH = []
        tprs_CT = []
        tprs_AC = []
        tprs_class = {'AH':tprs_AH, 'CT':tprs_CT, 'AC': tprs_AC}
        mean_fpr = np.linspace(0, 1, 100)
        
        rr = 0
        fig, ax = plt.subplots(figsize=(6, 6))
        for train_test in train_test_split:
        
            X_train, Y_train = X[train_test[0]], Y[train_test[0]]
            X_test, Y_test = X[train_test[1]], Y[train_test[1]]
            
            X2_train, Y2_train = X2[train_test[0]], Y2[train_test[0]]
            X2_test, Y2_test = X2[train_test[1]], Y2[train_test[1]]
            
            assert(list(Y_train) == list(Y2_train))
            assert(list(Y_test) == list(Y2_test))
            assert(list(sample_order) == list(sample_order3))
            
            # Now predict probabilities.
            rna_probs = rna_clf.predict_proba(X_train)
            rna_y_hat = rna_clf.predict(X_train)
            rna_y_hat2 = rna_clf.predict(X_test)
            prot_probs = prot_clf.predict_proba(X2_train)
            prot_y_hat = prot_clf.predict(X2_train)
            prot_y_hat2 = prot_clf.predict(X2_test)
            
            # print(rna_y_hat2)
            # print(prot_y_hat2)
            # print(Y_test)
            # print(rna_y_hat)
            # print(prot_y_hat)
            # print(Y_train)
            
            rna_probs_test = rna_clf.predict_proba(X_test)
            prot_probs_test = prot_clf.predict_proba(X2_test)
            
            # Use these to create X_Prob
            # Train a default LR classifier on X_Prob and Corresponding Y
            X_proba_train = np.concatenate((rna_probs, prot_probs), axis=1)
            X_proba_test = np.concatenate((rna_probs_test,prot_probs_test), axis=1)
            
            model = LogisticRegression()
            clf_proba = model.fit(X_proba_train, Y_train)
            
            # Test the classifier.
            # Report performance.
            Y_hat = clf_proba.predict(X_proba_test)
            # Setup for ROC/AUC
            Y_score = clf_proba.predict_proba(X_proba_test)
            
            # Setup for ROC curve
            label_binarizer = LabelBinarizer().fit(Y_train)
            Y_onehot_test = label_binarizer.transform(Y_test)
            
            #print(Y_test)
            #print(Y_score)
            #print(Y_onehot_test)
            
            acc = accuracy_score(Y_test, Y_hat)
            conf_matrix = confusion_matrix(Y_test, Y_hat)
            # auc = roc_auc_score(Y_test, Y_score, multi_class = 'ovr', average = 'macro')
            
            # print("Accuracy: ", acc)
            # print("Confusion Matrix: ", conf_matrix)
            
            # Draw the Micro-Averaged ROC
            viz = RocCurveDisplay.from_predictions(
            Y_onehot_test.ravel(),
            Y_score.ravel(),
            name=f"Micro-Averaged ROC fold {rr}",
            plot_chance_level=False,
            )
            print(Y_onehot_test)
            print(Y_score)
            print(Y_hat)
            
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            AUCs.append(viz.roc_auc)
            
            # Draw the Each Class vs Rest ROC
            classes = [0,1,2]
            class_names = ['AH', 'CT', 'AC']
            for cl in classes:
                viz = RocCurveDisplay.from_predictions(
                Y_onehot_test[:, cl],
                Y_score[:, cl],
                name=f"ROC fold {rr} {class_names[cl]} vs Rest",
                plot_chance_level=False)
                
                print(class_names[cl])
                print(Y_onehot_test[:, cl])
                print(Y_score[:, cl])
                #print(viz.roc_auc)
                #print(roc_auc_score(Y_onehot_test[:, cl], Y_score[:, cl]))
                
                fpr, tpr, thresholds = roc_curve(Y_onehot_test[:, cl], Y_score[:, cl])
                roc_auc = metrics.auc(fpr, tpr)
                print(fpr)
                print(tpr)
                print(thresholds)
                print(roc_auc)
                
                interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                #print(mean_fpr)
                #print(interp_tpr)
                interp_tpr[0] = 0.0
                tprs_class[class_names[cl]].append(interp_tpr)
                AUCs_class[class_names[cl]].append(viz.roc_auc)
            
            accuracies.append(acc)
            conf_matrices.append(conf_matrix)
            
            # AUCs2.append(auc)
            
            rr += 1
        
        mean_accuracy = np.mean(np.array(accuracies))
        i = 0
        mean_conf_matrix = 0
        while i < len(conf_matrices):
            if(i == 0):
                mean_conf_matrix = conf_matrices[0]
            else:
                mean_conf_matrix = mean_conf_matrix + conf_matrices[i]
            i += 1
            
        # Draw Mean ROC (Micro-Averaged One vs Rest)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(np.array(AUCs))
        std_auc = np.std(AUCs)
        
        ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC Micro-Averaged (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8)
        
        # Draw ROC for each class (One vs Rest)
        classes = [0,1,2]
        class_names = ['AH', 'CT', 'AC']
        colors = ['r', 'g', 'y']
        for cl in classes:
            mean_tpr = np.mean(tprs_class[class_names[cl]], axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = np.mean(np.array(AUCs_class[class_names[cl]]))
            std_auc = np.std(AUCs_class[class_names[cl]])
            
            ax.plot(
            mean_fpr,
            mean_tpr,
            color=colors[cl],
            label=r"%s ROC (AUC = %0.2f $\pm$ %0.2f)" % (class_names[cl], mean_auc, std_auc),
            lw=2,
            alpha=0.8)
        
        ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"ROC Curves One-vs-Rest\n",
        )
        ax.axis("square")
        ax.legend(loc="lower right")
        plt.show()
        
        print("Mean Accuracy: ", mean_accuracy)
        print("Mean AUC (ovr micro): ", mean_auc)
        # print("Mean AUC (ovr macro): ", mean_auc2)
        print("Mean Confusion Matrix: ", mean_conf_matrix)
    else:
        
        AUC_RNA = []
        AUC_PROT = []
        AUC_class_RNA = {'AH': [], 'CT': [], 'AC': []}
        AUC_class_PROT = {'AH': [], 'CT': [], 'AC': []}
        
        tpr_RNA = []
        tpr_PROT = []
        tpr_class_RNA = {'AH':[], 'CT':[], 'AC': []}
        tpr_class_PROT = {'AH':[], 'CT':[], 'AC': []}
        
        mean_fpr = np.linspace(0, 1, 100)
        
        if(tissue == 'LV'):
            test = [0,1,2,3,4,5,7,8,9,10,11,12,13,18,19,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
        elif(tissue == 'PBMC'):
            test = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
    
        X_test, Y_test = X[test], Y[test]
        X2_test, Y2_test = X2[test], Y2[test]
        
        print(X2_test.shape)
        print(X_test.shape)
        
        r = permutation_importance(rna_clf, X_test, Y_test, n_repeats=30, random_state=0)
        r2 = permutation_importance(prot_clf, X2_test, Y2_test, n_repeats=30, random_state=0)
        
        def sort_and_plot_feature_importance(out_path:str, feature_names:list, feature_rankings:list, feature_stds:list):
            feature_rank_mean = {}
            i = 0
            while i < len(feature_names):
                # if(abs(feature_rankings[i]) > 2*feature_stds[i]):
                feature_rank_mean[feature_names[i]] = feature_rankings[i]
                i += 1
                
            _feature_names = []
            _feature_rankings = []
                
            sorted_features = sorted(feature_rank_mean.items(), key = lambda x: abs(x[1]), reverse = True)
            for k in sorted_features:
                _feature_names.append(k[0])
                _feature_rankings.append(k[1])
                
            feature_rankings = np.array(_feature_rankings)
            
            plot_feature_importance_permutation(out_path, _feature_names, _feature_rankings, 20)
            
            return _feature_names, _feature_rankings
        
        out_dir = 'C:/.../'
        out_fname = out_dir + 'PBMC_3Way_Integrated_Genes'
        out_fname2 = out_dir + 'PBMC_3Way_Integrated_Proteins'
        
        sorted_gene_names, sorted_gene_rankings = sort_and_plot_feature_importance(out_fname, genes_to_read,
                                                                                   r.importances_mean, r.importances_std)
        sorted_prot_names, sorted_prot_rankings = sort_and_plot_feature_importance(out_fname2, proteins_to_read,
                                                                                   r2.importances_mean, r2.importances_std)
        
        out_fname3 = out_dir + 'top_model_features_permutation.txt'
        
        with open(out_fname3, 'w') as writer:
            writer.write("Permutation Importance Testing:\n")
            writer.write("RNA Model:\n")
            i = 0
            while i < len(genes_to_read):
                writer.write('Gene: ' + genes_to_read[i] + ' ' + str(r.importances_mean[i]))
                writer.write('+|- ' + str(r.importances_std[i]) + '\n')
                i += 1
                
            writer.write('\n')
                
            i = 0
            writer.write("Protein Model:\n")
            while i < len(proteins_to_read):
                writer.write('Protein: ' + proteins_to_read[i] + ' ' + str(r2.importances_mean[i]))
                writer.write('+|- ' + str(r2.importances_std[i]) + '\n')
                i += 1
        
        rna_y_hat = rna_clf.predict(X_test)
        prot_y_hat = prot_clf.predict(X2_test)
        
        # Setup for ROC/AUC
        Y_score_rna = rna_clf.predict_proba(X_test)
        Y_score_prot = prot_clf.predict_proba(X2_test)
        
        # Setup for ROC curve
        label_binarizer = LabelBinarizer().fit(Y_test)
        Y_onehot_test = label_binarizer.transform(Y_test)
        
        dtypes = ['RNA', 'PROT']
        
        for dtype in dtypes:
            fig, ax = plt.subplots(figsize=(6, 6))
        
            if(dtype == 'RNA'):
                Y_score = Y_score_rna
            else:
                Y_score = Y_score_prot
        
            # Draw the Micro-Averaged ROC
            viz = RocCurveDisplay.from_predictions(
            Y_onehot_test.ravel(),
            Y_score.ravel(),
            name=f"Micro-Averaged ROC " + dtype,
            plot_chance_level=False,
            )
            
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            if(dtype == 'RNA'):
                tpr_RNA.append(interp_tpr)
                AUC_RNA.append(viz.roc_auc)
            else:
                tpr_PROT.append(interp_tpr)
                AUC_PROT.append(viz.roc_auc)
        
            # Draw the Each Class vs Rest ROC
            classes = [0,1,2]
            class_names = ['AH', 'CT', 'AC']
            for cl in classes:
                viz = RocCurveDisplay.from_predictions(
                Y_onehot_test[:, cl],
                Y_score[:, cl],
                name=f"ROC {class_names[cl]} vs Rest " + dtype,
                plot_chance_level=False)
                
                print(cl)
                print(Y_onehot_test[:, cl])
                print(Y_score[:, cl])
                
                fpr, tpr, thresholds = roc_curve(Y_onehot_test[:, cl], Y_score[:, cl])
                print(fpr)
                print(tpr)
                print(thresholds)
                roc_auc = metrics.auc(fpr, tpr)
                
                interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0.0
                if(dtype == 'RNA'):
                    tpr_class_RNA[class_names[cl]].append(interp_tpr)
                    AUC_class_RNA[class_names[cl]].append(viz.roc_auc)
                else:
                    tpr_class_PROT[class_names[cl]].append(interp_tpr)
                    AUC_class_PROT[class_names[cl]].append(viz.roc_auc)
        
            # Draw Mean ROC (Micro-Averaged One vs Rest)
            if(dtype == 'RNA'):
                tpr = tpr_RNA[0]
                tpr[-1] = 1.0
                auc = AUC_RNA[0]
            else:
                tpr = tpr_PROT[0]
                tpr[-1] = 1.0
                auc = AUC_PROT[0]
                
            if(dtype == 'RNA'):
                label = r"Mean ROC Micro-Averaged RNA (AUC = %0.2f)" % (auc)
            else:
                label = r"Mean ROC Micro-Averaged PROT (AUC = %0.2f)" % (auc)
            
            ax.plot(
            mean_fpr,
            tpr,
            color="b",
            label=label,
            lw=2,
            alpha=0.8)
            
            # Draw ROC for each class (One vs Rest)
            classes = [0,1,2]
            class_names = ['AH', 'CT', 'AC']
            colors = ['r', 'g', 'y']
            for cl in classes:
                if(dtype == 'RNA'):
                    tpr = tpr_class_RNA[class_names[cl]][0]
                    tpr[-1] = 1.0
                    auc = AUC_class_RNA[class_names[cl]][0]
                else:
                    tpr = tpr_class_PROT[class_names[cl]][0]
                    tpr[-1] = 1.0
                    auc = AUC_class_PROT[class_names[cl]][0]
                
                if(dtype == 'RNA'):
                    label = r"%s ROC RNA (AUC = %0.2f)" % (class_names[cl], auc)
                else:
                    label = r"%s ROC PROT (AUC = %0.2f)" % (class_names[cl], auc)
                
                ax.plot(
                mean_fpr,
                tpr,
                color=colors[cl],
                label=label,
                lw=2,
                alpha=0.8)
            
            ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"ROC Curves " + dtype + " One-vs-Rest\n",
            )
            ax.axis("square")
            ax.legend(loc="lower right")
            plt.show()
        
        rna_acc = accuracy_score(Y_test, rna_y_hat)
        rna_conf_matrix = confusion_matrix(Y_test, rna_y_hat)
        prot_acc = accuracy_score(Y_test, prot_y_hat)
        prot_conf_matrix = confusion_matrix(Y_test, prot_y_hat)
        
        print("RNA Accuracy: ", rna_acc)
        print(rna_conf_matrix)
        print("Proteomics Accuracy: ", prot_acc)
        print(prot_conf_matrix)

def write_cuffdiff_HPC_batch_file(fname:str, dataset:str, samples:str, i:int, j:int = -1):
    assert dataset in ['PBMC_3Way_COMP', 'LV_3Way_COMP']
    
    if(dataset == 'LV_3Way_COMP'):
        with open(fname, 'w') as writer:
            writer.write('#!/bin/bash\n')
            if(j == -1):
                writer.write('#SBATCH --job-name=...' + str(i) + '\n')
            else:
                writer.write('#SBATCH --job-name=...' + str(i) + '_' + str(j) + '\n')
            writer.write('#SBATCH -A ...\n')
            writer.write('#SBATCH -p highmem\n')
            writer.write('#SBATCH --nodes=1\n')
            writer.write('#SBATCH --ntasks=1\n')
            writer.write('#SBATCH --cpus-per-task=32\n')
            writer.write('#SBATCH --mail-type=begin,end,fail\n')
            writer.write('#SBATCH --mail-user=...\n')
            writer.write('\n')
            writer.write('\n')
            writer.write('module load openmpi/4.0.3/gcc.8.4.0\n')
            writer.write('export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n')
            writer.write('module load cufflinks/2.2.1\n')
            writer.write('\n')
            writer.write('cd /.../cxbs/\n')
            writer.write('\n')
            writer.write('cuffdiff -p 64 --max-bundle-frags 1000000000 --library-norm-method geometric -o /.../')
            writer.write('.../')
            if(j == -1):
                writer.write('Cuffdiff_GEOM_POOL_FOLD' + str(i) + '/ hg38_ensembl.gtf ')
            else:
                writer.write('Cuffdiff_GEOM_POOL_FOLD' + str(i) + '_' + str(j) + '/ hg38_ensembl.gtf ')
            writer.write(samples)
    else:
        with open(fname, 'w') as writer:
            writer.write('#!/bin/bash\n')
            if(j == -1):
                writer.write('#SBATCH --job-name=SL_...' + str(i) + '\n')
            else:
                writer.write('#SBATCH --job-name=SL_...' + str(i) + '_' + str(j) + '\n')
            writer.write('#SBATCH -A ...\n')
            writer.write('#SBATCH -p highmem\n')
            writer.write('#SBATCH --nodes=1\n')
            writer.write('#SBATCH --ntasks=1\n')
            writer.write('#SBATCH --cpus-per-task=32\n')
            writer.write('#SBATCH --mail-type=begin,end,fail\n')
            writer.write('#SBATCH --mail-user=...\n')
            writer.write('\n')
            writer.write('\n')
            writer.write('module load openmpi/4.0.3/gcc.8.4.0\n')
            writer.write('export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n')
            writer.write('module load cufflinks/2.2.1\n')
            writer.write('\n')
            writer.write('cd /.../cxbs/\n')
            writer.write('\n')
            writer.write('cuffdiff -p 64 --max-bundle-frags 1000000000 --library-norm-method geometric -o /.../')
            writer.write('.../')
            if(j == -1):
                writer.write('Cuffdiff_GEOM_POOL_FOLD' + str(i) + '/ hg38_ensembl.gtf ')
            else:
                writer.write('Cuffdiff_GEOM_POOL_FOLD' + str(i) + '_' + str(j) + '/ hg38_ensembl.gtf ')
            writer.write(samples)

if __name__ == "__main__":
    # MUST BE PYTHON 3.7+ since, DICTIONARIES are ASSUMED to be ORDERED throughout the codebase.
    assert sys.version_info.major == 3
    assert sys.version_info.minor >= 7  
    
    # Intersection Analysis
    # LV3Way_Conv_Prot_to_Genes = ['...']
    
    # LV3Way_Genes = ['...']
    
    # intersection = set(LV3Way_Conv_Prot_to_Genes) & set(LV3Way_Genes)
    # print(intersection)
    
    # LV_3Way_Dir = 'C:/.../'
    # LV_3Way_Dir += 'hg38_Starcq_Ensembl/Cuffdiff_GEOM_POOL/'
    # LV_3Way_Diff = LV_3Way_Dir + 'gene_exp.diff'
    # PBMC_3Way_Dir = 'C:/.../'
    # PBMC_3Way_Dir += 'hg38_Starcq_Ensembl/Cuffdiff_GEOM_POOL/'
    # PBMC_3Way_Diff = PBMC_3Way_Dir + 'gene_exp.diff'
    # LV_3Way_Genes = 'C:/.../LV3Way_Genes.txt'
    # PBMC_3Way_Gense = 'C:/.../PBMC3Way_Genes.txt'
    # filter_cuffdiff_file_by_gene_list(LV_3Way_Genes, LV_3Way_Diff, LV_3Way_Dir)
    # filter_cuffdiff_file_by_gene_list(PBMC_3Way_Gense, PBMC_3Way_Diff, PBMC_3Way_Dir)
    
    # LV_3Way_Proteins = 'C:/.../LV3Way_Proteins.txt'
    # PBMC_3Way_Proteins = 'C:/.../PBMC3Way_Proteins.txt'
    # LV_3Way_Dir = 'C:/.../'
    # LV_3Way_Dir += 'LV_AH_CT_AC_Unmatched_Balanced/'
    # LV_3Way_Diff = LV_3Way_Dir + 'Samples_ALL_DE.csv'
    # PBMC_3Way_Dir = 'C:/.../'
    # PBMC_3Way_Dir += 'PB_AH_CT_AC_Unmatched_Balanced/'
    # PBMC_3Way_Diff = PBMC_3Way_Dir + 'Samples_ALL_DE.csv'
    # filter_inferno_by_protein_list(LV_3Way_Proteins, LV_3Way_Diff, LV_3Way_Dir, 'Samples_ALL_DE_Filtered.csv')
    
    
    # Integrated PBMC 3-Way Setup
    prot_settings = [(70, 5, 'Variance', 2.5)]
    rna_settings = [('LR', 'Hybrid', 'DE', 3, 25, 4)]
    
    # Integrated Liver 3-Way Setup
    # prot_settings = [(35, 3)] # size, ftest
    # rna_settings = [('Union', 200, 'LR', 5)]
    
    for rna_setting in rna_settings:
        for prot_setting in prot_settings:
            
            '''
            root_dir_RNA = 'C:/.../'
            work_dir_RNA = 'C:/.../'
            work_dir_RNA += 'LV_Unmatched_Ensembl3_'
            work_dir_RNA += rna_setting[2] + '_DE_' + rna_setting[0] + '_3.5'
            if(rna_setting[2] == 'SVM'):
                work_dir_RNA += '_Low'
            work_dir_RNA += '/'
            filter_RNA = True
            filter_mode_RNA = rna_setting[0]
            feature_size_RNA = rna_setting[1]
            std_thresh_RNA = 3.5
            FS_Test_RNA = rna_setting[3]
            model_RNA = rna_setting[2]
            FS_Mode_RNA = 'DE'
            
            root_dir_prot = 'C:/.../'
            work_dir_prot = 'C:/.../'
            out_dir_prot = 'C:/.../'
            out_dir_prot += '.../'
            feature_size_prot = prot_setting[0]
            dataset_prot = 'LV_3Way_Unmatched_Balanced'
            impute_prot = 'median'
            scaling_prot = None
            impute_thresh_prot = 0
            FS_Test_Prot = prot_setting[1]
            log_transform_prot = False
            
            # print("Feature Size RNA: ", feature_size_RNA)
            # print("Feature Size Proteomics: ", feature_size_prot)
            print('RNA Settings: ', rna_setting)
            print("Proteomics Settings: ", prot_setting)
            
            tissue = 'LV'
            '''
            
    
            
            # Integrated PBMC 3-Way Setup
            root_dir_RNA = 'C:/.../'
            work_dir_RNA = 'C:/.../'
            work_dir_RNA += 'PB_Unmatched_Balanced3_' + str(rna_setting[0]) + '_' + str(rna_setting[2])
            work_dir_RNA += '_' + str(rna_setting[1]) + '_' + str(rna_setting[3])
            if(rna_setting[0] == 'SVM'):
                work_dir_RNA += '_Low'
            work_dir_RNA += '/'
            filter_RNA = True
            filter_mode_RNA = rna_setting[1]
            feature_size_RNA = rna_setting[4]
            std_thresh_RNA = rna_setting[3]
            FS_Test_RNA = rna_setting[5]
            FS_Mode_RNA = rna_setting[2]
            model_RNA = rna_setting[0]
            
            root_dir_prot = 'C:/.../'
            work_dir_prot = 'C:/.../'
            out_dir_prot = 'C:/.../'
            if(prot_setting[2] == 'Variance'):
                if(prot_setting[3] == 2.5):
                    out_dir_prot += 'PB_AH_CT_AC_Excluded_Unmatched_Balanced/PBMC_3Way_Unmatched_Balanced_LR_DE_Variance_2.5_median_0/'
                elif(prot_setting[3] == 3.0):
                    out_dir_prot += 'PB_AH_CT_AC_Excluded_Unmatched_Balanced/PBMC_3Way_Unmatched_Balanced_LR_DE_Variance_3.0_median_0/'
            else:
                out_dir_prot += 'PB_AH_CT_AC_Excluded_Unmatched_Balanced/PBMC_3Way_Unmatched_Balanced_LR_DE_median_0/'
            feature_size_prot = prot_setting[0]
            dataset_prot = 'PBMC_3Way_Unmatched_Balanced'
            impute_prot = 'median'
            scaling_prot = None
            impute_thresh_prot = 0
            FS_Test_Prot = prot_setting[1]
            log_transform_prot = False
            
            # print("Feature Size RNA: ", feature_size_RNA)
            # print("Feature Size Proteomics: ", feature_size_prot)
            
            print('RNA Settings: ', rna_setting)
            print("Proteomics Settings: ", prot_setting)
            
            tissue = 'PBMC'
            
            classify_matched_with_unmatched_models_ensembl(root_dir_RNA, work_dir_RNA, model_RNA, FS_Mode_RNA, filter_RNA,
                                                           filter_mode_RNA, feature_size_RNA, std_thresh_RNA, FS_Test_RNA,
                                                           root_dir_prot, work_dir_prot, out_dir_prot, feature_size_prot,
                                                           dataset_prot, impute_prot, scaling_prot, impute_thresh_prot,
                                                           FS_Test_Prot, log_transform_prot, tissue, False)
    