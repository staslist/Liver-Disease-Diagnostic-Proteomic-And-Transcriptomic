# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 15:02:47 2022

@author: staslist
"""

def isfloat(x):
    try:
        float(x)
    except:
        return False
    else:
        return True
    
def read_in_gene_exp(fname:str, fc_cutoff:float = 0, reverse_directionality:bool = False):
    gene_diff_dict = {}
    with open(fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        i = 0
        for row in csv_reader:
            if(i > 1):
                if(float(row[7]) > 1 and float(row[8]) > 1 and abs(float(row[9])) > fc_cutoff and float(row[12]) < 0.05):
                    if(not reverse_directionality):
                        gene_diff_dict[row[0]] = (float(row[9]))
                    else:
                        gene_diff_dict[row[0]] = -(float(row[9]))
            i += 1
     
    result = sorted(gene_diff_dict.items(), key = lambda x: x[1], reverse = True)
    result2 = {}
    for tup in result:
        result2[tup[0]] = tup[1]
    return result2

def div_dict_in_pos_and_neg(gene_FC:dict):
    '''Assume the input is dictionary that maps gene names to log2(fold change).'''
    pos,neg = [],[]
    for k,v in gene_FC.items():
        if(v > 0):
            pos.append(k)
        else:
            neg.append(k)
            
    return pos, neg

from AH_Project_Codebase_SL import *
if __name__ == "__main__":
    # MUST BE PYTHON 3.7+ since, DICTIONARIES are ASSUMED to be ORDERED throughout the codebase.
    assert sys.version_info.major == 3
    assert sys.version_info.minor >= 7   
    
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600

    test_suite()
    
    # Code to generate read groups info files from cuffdiff batch commands.
    
    # full_path = 'C:/.../'
    # full_path += 'SL_22AH_vs_8CT_vs_32AC_PB_hg38_starcq_ensembl_GEOM_POOL_FOLD5_5.sh'
    # out_path = 'C:/.../'
    # out_path += 'Cuffdiff_GEOM_POOL_FOLD5_5/read_groups.info'
    # generate_read_groups_info_from_cuffdiff_command(full_path, out_path)
    
    # Generate HPC Files
    
    # out_dir = 'C:/.../'
    # generate_cuffnorm_or_cuffdiff_batch_file_HPC('hg38', 'starcq', 'ensembl', 'GEOM', ['AH', 'CT', 'AC'], 'AH_CT_AC_LV_Proteomic', 
    #                                               'AH_CT_AC_LV_Proteomic_Ensembl', out_dir, "Cuffdiff", 1, 'POOL', tissue = 'LV3_Proteomic_Ensembl')
    # generate_cuffnorm_or_cuffdiff_batch_file_HPC('hg38', 'starcq', 'ensembl', 'GEOM', ['AH', 'CT', 'AC'], 'AH_CT_DA_AA_NF_HP', 
    #                                               'AH_CT_AC_PB_Unmatched_Ensembl2', out_dir, "Cuffdiff", [5,5], 'POOL', tissue = 'PB3_Unmatched_Ensembl2')
    
    # GSEA & BTM BLOCK
    
    # root_dir = 'C:/.../'
    # root_dir += 'hg38_Starcq_Ensembl/Cuffdiff_GEOM_POOL/'
    # gene_list_file = root_dir + 'LV5Way_Genes.txt'
    # genes = read_in_csv_file_one_column(gene_list_file, 0, ',')
    # generate_bloodgen3module_counts(root_dir, True)
    # generate_GSEA_counts(genes)
    # filter_cuffdiff_file_by_gene_list(gene_list_file, root_dir + "gene_exp.diff", root_dir)
    
    # counts, sample_order = generate_GSEA_counts()
    
    
    
    # Run RNA-seq ML on local machine. 
    
    '''
    FS_Test = 5
    tissue = 'PB_Unmatched_Ensembl'
    num_conds = 3
    cv_k_outer = 5
    cv_k_inner = 5
    features_to_gen = 500
    Filter = True
    b_acc = False
    # f_sizes = [10,25,50,100,150,200,250,300,350,400,450,500]
    f_sizes = [10,25,50,100,150,200]
    # f_sizes = [2,3,4,5,10,15,20,25,50]
    
    for FS_Mode in ['DE']:
        for model_name in ['LR']:
            for Filter_Mode in ['Union']:
                for treshold in [3.5]:
                
                    if(FS_Mode == 'DE'):
                        fpkm_fs = True
                    else:
                        fpkm_fs = False
                        
                    fpkm_ml = False                          
            
                    work_dir = os.getcwd() + '/'
                    dir_name = tissue + str(num_conds) + '_' + model_name + '_' + FS_Mode + '_' + Filter_Mode + '_' + str(treshold)
                    # dir_name += '_FS_Test' + str(FS_Test)
                    if(b_acc):
                        dir_name += '_' + 'BACC'
                    dir_name += '/'
                        
                    work_dir += dir_name
                    try:
                        os.makedirs(work_dir)
                    except FileExistsError:
                        pass
                    
                    log_file = work_dir + 'execution_log.txt'
                    console = sys.stdout
                    sys.stdout = open(log_file, 'w')
                    
                    
                    #1) Establish correctness of inner and outer splits.
                    #2) Perform feature selection on each inner and outer training set.
                    #3) Perform the nested cross-validation.
                    root_dir = 'C:/.../'
                    
                    print("Pipeline Configuration:")
                    print("Model name: ", model_name)
                    print("Number of conditions: ", num_conds)
                    print("Feature selection mode: ", FS_Mode)
                    print("Features generated: ", features_to_gen)
                    print("Feature Sizes: ", f_sizes)
                    print("Filter Status: ", Filter)
                    print("Filter Mode: ", Filter_Mode)
                    print("Treshold: ", treshold)
                    print("K_Outer: ", cv_k_outer)
                    print("K_Inner: ", cv_k_inner)
                    print("Balanced Accuracy: ", b_acc)
                    #print("FS_Test: ", FS_Test)
                    
                    a = classify_with_nested_CV(root_dir, work_dir, model_name, num_conds, cv_k_outer, cv_k_inner, features_to_gen, f_sizes,
                                                FS_Mode, Filter, Filter_Mode, b_acc, tissue, fpkm_fs, fpkm_ml, treshold)
                    
                    # ---------------------------------------------------------------------------------------------------------------- 
                    # ***************************************************Validate in Test Data****************************************
                    # ---------------------------------------------------------------------------------------------------------------- 
                    
                    # b = validate_in_test_data(root_dir, work_dir, model_name, FS_Mode, num_conds, cv_k_outer, features_to_gen, Filter,
                    #                           Filter_Mode, b_acc, f_sizes, fpkm_fs, fpkm_ml, treshold, FS_Test)
                    
                    sys.stdout.close()
                    sys.stdout = console
    '''
    
    # Identify misclassified samples.
    
    '''
    sample_misclass_dict = {}
    sample_misclass_list = []
    home = 'C:/.../'
    for classifier in ['LR']:
        for filt in ['Hybrid', 'Union']:
            for tresh in ['2.5', '3', '3.5']:
                fname = home + 'PB5' + '_' + classifier + '_DE_' + filt + '_' + tresh + '/execution_log.txt'
                samples,samples_mis = identify_misclassified_samples(fname)
                sample_misclass_list.append(samples_mis)
                #print(classifier, ' ', filt, ' ', tresh)
                #print(samples)
                for sample in samples.keys():
                    if(samples[sample] == 12):
                        try:
                            sample_misclass_dict[sample] = sample_misclass_dict[sample] + 1
                        except KeyError:
                            sample_misclass_dict[sample] = 1
                            
    #print(sample_misclass_dict)
    #print(sample_misclass_list)
    
    for k,v in sample_misclass_dict.items():
        if v == 6:
            i = 0
            while i < 6:
                print(k)
                print(sample_misclass_list[i][k])
                i += 1
    '''
    
    # Code to parse execution logs and attain ML results in convenient format.
    '''
    for dataset in ['PB3']:
        for classifier in ['SVM']:
            for FS_Method in ['DE', 'IG']:
                for Filter_Method in ['Union', 'Hybrid']:
                    for treshold in ['2.5', '3', '3.5']:
                        # for FS_Test in ['FS_Test5', 'FS_Test4']:
                        # for fsize in ['Low', 'Med', 'High']:
                            try:
                                indir = 'C:/.../' 
                                indir += dataset + '_' + classifier + '_' + FS_Method
                                indir += '_' + Filter_Method + '_' + treshold# + '_' + fsize 
                                indir += '/' + 'execution_log.txt'
                                
                                out_name = dataset + '_' + classifier + '_' + FS_Method 
                                out_name += '_' + Filter_Method + '_' + treshold# + '_' + fsize
                                out_name += '.txt'
                                parse_execution_log(indir, out_name, 'PB')
                            except FileNotFoundError:
                                print("Fail")
                                continue
    '''
    
    '''
    # Code to parse biological validation summary files and attain these results in convenient format.
    for dataset in ['PB3']:
        for classifier in ['LR']:
            for FS_Method in ['DE', 'IG']:
                for Filter_Method in ['Union', 'Hybrid']:
                    for treshold in ['25', '3', '35']:
                        for FS_Test in ['FS_Test5', 'FS_Test4']:
                        # for fsize in ['Low', 'Med', 'High']:
                            try:
                                indir = 'C:/.../' 
                                indir += dataset + '_' + classifier + '_' + FS_Method
                                indir += '_' + Filter_Method + '_' + treshold + '_' + FS_Test 
                                indir += '_Enricher.txt'
                                
                                #print(indir)
                                
                                out_name = dataset + '_' + classifier + '_' + FS_Method 
                                out_name += '_' + Filter_Method + '_' + treshold + '_' + FS_Test
                                out_name += '.txt'
                                parse_biological_validation_log(indir, out_name)
                            except FileNotFoundError:
                                print("Fail")
                                continue
    '''
    
    # In Silico Manual Biological Validation
    
    # out_dir = "C:/.../"
    
    # total = ['GENE1', 'GENE2', 'etc']
    # r1, r2, r3 = biological_validation(total, 'PB3_DE', os.getcwd() + '/', 'Pathway')
    # r4, r5, r6 = biological_validation(total, 'PB3_DE', os.getcwd() + '/', 'Tissue')
    # r7, r8, r9 = biological_validation(total, 'PB3_DE', os.getcwd() + '/', 'Disease')
    
    
    # Gene set validation.
    '''
    genesets_dir = 'C:/.../'
    for dataset in ['LV_Unmatched_Ensembl3']:
        for ML_method in ['LR']:
            for fs_method in ['DE']:
                for filter_method in ['Union']:
                    for treshold in ['3.5']:
                        for FS_Test in ['FS_Test5']:
                            if(dataset in ['LV2', 'PB2']):
                                fs_sizes = [2,3,4,5,10,15,20,25,50]
                            else:
                                # fs_sizes = [10,25,50,100,150,200,250,300,350,400,450,500]
                                fs_sizes = [100]
                            for fs_size in fs_sizes:
                                names = []
                                i = 0
                                while i < 5:
                                    name = dataset + '_' + ML_method + '_' + fs_method 
                                    name += '_' + filter_method + '_' + treshold # + '_' + FS_Test
                                    name += '/top_rnaseq_features_set' + str(i) + '.txt'
                                    names.append(name)
                                    i+=1
                                tresh_no_dot = treshold.replace('.', '')
                                name_out = dataset + '_' + ML_method + '_' + fs_method
                                name_out += '_' + filter_method + '_' + tresh_no_dot
                                name_out += '_' + FS_Test
                                name_out += '_Enricher.txt'
                                totals = []
                                i = 0
                                while i < 5:
                                    total = read_in_csv_file_one_column(genesets_dir + names[i], 0, ',', 0, fs_size)
                                    totals.append(total)
                                    i += 1
                                N = int(FS_Test[-1])
                                print("N: ", N)
                                counts = count_elements_in_2dlist(totals)
                                total = []
                                for k,v in counts.items():
                                    if(v >= N):
                                        total.append(k)
                                if(len(total) == 0):
                                    continue
                                r1, r2, r3 = biological_validation(total, name_out, os.getcwd() + '/', 'Pathway')
                                r4, r5, r6 = biological_validation(total, name_out, os.getcwd() + '/', 'Tissue')
                                r7, r8, r9 = biological_validation(total, name_out, os.getcwd() + '/', 'Disease')
                                
                                print(len(total))
                                print(total)
                                # with open(os.getcwd() + '/' + name_out, 'a') as writer:
                                #     writer.write('Feature Size: ' + str(fs_size))
                                #     writer.write('\n')
                                #     writer.write('Actual Feature Size: ' + str(len(total)))
                                #     writer.write('\n')
                                #     writer.write(str(len(r3)))
                                #     writer.write('/')
                                #     writer.write(str(len(r6)))
                                #     writer.write('/')
                                #     writer.write(str(len(r9)))
                                #     writer.write('\n')
    '''
    
    # CONF MATRIX
    # out_dir = os.getcwd() + '/'
    # array = np.array([[32,  0 , 6],[ 0, 17,  3],[ 6,  1, 33]])
    # generate_confusion_matrix(out_dir, array, ['AH', 'CT', 'AC'], 0.83, 'PB 3-Way IG LR Union 2.5')
    
    
    # COUNT HEATMAPS
    # Generating custom RNA-seq (cuffnorm counts) heatmaps
    
    genes = ['GENE1', 'GENE2', 'etc']
    genes_to_read = genes
    
    # genes.fpkm_table
    out_dir = 'C:/.../'
    '''
    base = 'C:/.../'
    fname = base + 'genes.read_group_tracking'
    fname2 = base + 'read_groups.info'
    counts = read_cuffdiff_counts2(fname, 'ALL', True, True)
    
    filtered_counts_temp = {}
    for gene in genes:
        if(gene in counts.keys()):
            filtered_counts_temp[gene] = counts[gene]
    
    # Its crucial that genes counts are ordered identically for each replicate.
    p = 0 
    filtered_counts = {}            
    gene_order = []
    for gene, gene_feature in filtered_counts_temp.items():
        gene_order.append(gene)
        if(p == 0):
            for rep_count_tuple in gene_feature:
                rep_name = rep_count_tuple[0]
                filtered_counts[rep_name] = []
        for rep_count_tuple in gene_feature:
            rep_name = rep_count_tuple[0]
            gene_count = rep_count_tuple[1]
            filtered_counts[rep_name].append(gene_count)
        p += 1
    
    assert(len(gene_order) == len(genes))
    iii = 0
    while iii < len(gene_order):
        assert(gene_order[iii] == genes[iii])
        iii += 1
    # Reverse the order of replicates to account for heatmaps being rotated.
    filtered_counts_reverse = dict(reversed(list(filtered_counts.items())))
    
    plot_per_sample_counts_heatmap(filtered_counts_reverse, genes, fname2, 0, out_dir, len(genes))
    '''
    
    # ALTERNATIVE: PREP TEST DATA FOR PER SAMPLE HEATMAP GRAPHING
    '''
    out_dir = 'C:/.../'
    num_conditions = 3
    test_dataset = os.getcwd() + '/GSE142530_Annoted-RNAseq-with-SampleIDs.csv'
    counts_test = {}
    headers = []
    if(num_conditions == 2):
        conditions = ['RB_N', 'RB_A']
    else:
        conditions = ['RB_E', 'RB_N', 'RB_A']
    with open(test_dataset) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                headers = row
            else:
                i = 1
                gene_name = row[1]
                if(gene_name not in genes_to_read):
                    continue
                counts_test[gene_name] = []
                while i < len(row):
                    if(headers[i][0:4] in conditions):
                        counts_test[gene_name].append((headers[i], math.log(1 + float(row[i]))))
                    i = i + 1
            line_count += 1 
       
    # Have to arrange the features (genes) in exactly the same order for both training and testing data. 
    # This follows the ordering of genes_to_read AKA the ordering of the top_features file.
    
    counts_test2 = {}
    for gene in genes_to_read:
        v = counts_test[gene]
        for rep_count_tuple in v:
            rep = rep_count_tuple[0]
            count = rep_count_tuple[1]
            try:
                counts_test2[rep].append( count)
            except KeyError:
                counts_test2[rep] = [count]
                
    filtered_counts = counts_test2
    fname2 = None
    
    # Order filtered_counts (keys) according to the conditions list.
    filtered_counts_ordered = {}
    for cond in conditions:
        for k,v in filtered_counts.items():
            if(k.startswith(cond)):
                filtered_counts_ordered[k] = v
    
    plot_per_sample_counts_heatmap(filtered_counts_ordered, genes, fname2, 0, out_dir, len(genes))
    '''
    # Alternative process to generate mean counts per condition
    '''
    counts_mean_std = read_cuffdiff_counts_mean_std2(fname, True, True)
    
    mean_counts = {}
    num_conditions = 3
    conditions = ['AH', 'CT', 'AC']
    for gene in genes:
        if gene in counts_mean_std.keys():
             temp = counts_mean_std[gene][0:num_conditions]
             # Re-order the conditions in reverse.
             # mean_counts[gene] = temp[::-1]
             mean_counts[gene] = temp
            
    plot_per_condition_counts_heatmap(mean_counts, genes, conditions, 0, out_dir, len(genes))
    '''
    '''
    # ALTERNATIVE: PREP TEST DATA FOR PER CONDITION HEATMAP GRAPHING
    mean_counts = {}
    num_conditions = 3
    conditions = ['AH', 'CT', 'AC']
    for k,v in counts_test.items():
        AH_counts = []
        AC_counts = []
        CT_counts = []
        for rep_count_tuple in v:
            rep = rep_count_tuple[0]
            underscore = rep.index('_')
            cond = rep[underscore + 1:]
            if(cond.startswith('AH')):
                AH_counts.append(rep_count_tuple[1])
            elif(cond.startswith('N')):
                CT_counts.append(rep_count_tuple[1])
            elif(cond.startswith('EC')):
                AC_counts.append(rep_count_tuple[1])
            else:
                raise ValueError("Unknown condition in test dataset!")
        assert(len(AH_counts) == 10)
        assert(len(CT_counts) == 12)
        if(num_conditions == 3):
            assert(len(AC_counts) == 6)
        AH_mean = np.mean(np.array(AH_counts))
        CT_mean = np.mean(np.array(CT_counts))
        AC_mean = np.mean(np.array(AC_counts))
        
        if(num_conditions == 2):
            mean_counts[k] = [AH_mean, CT_mean]
        elif(num_conditions == 3):
            mean_counts[k] = [AH_mean, CT_mean, AC_mean]
        else:
            raise ValueError("Test dataset must have either 2 or 3 conditions selected.")
            
    mean_counts2 = {}
    for gene in genes:
        mean_counts2[gene] = mean_counts[gene]
    
    plot_per_condition_counts_heatmap(mean_counts2, genes, conditions, 0, out_dir, len(genes))
   '''