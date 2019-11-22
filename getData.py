from rdkit.Chem.Fingerprints import FingerprintMols
from sklearn.model_selection import KFold
from rdkit.Chem import AllChem
from rdkit import Chem

import pandas as pd
import numpy as np

def getData(L, cell_line, descriptors='ecfp', target='regression', n_fold=5, random_state=42, random_genes=True):
    """
    L: dictionary from L1000CDS_subset.json
    
    cell_line: cell_id
    
    descriptors: list of descriptors for chemical compounds.
        params:
            string: 'ecfp'(default), 'maccs', 'topological', 'shed', 'cats2d', 'jtvae'
            
    target: chdirLm or vector of upGenes, dnGenes (-1 for down, 1 for up)
        params:
            string: 'regression'(default), 'class'
            
    n_fold: number of folds
        params:
            int: 5(default)
            
    random_state: random_state for Kfold
        params:
            int: 42(default)
    
    random_genes: if it is true, returns random 20 genes from target values
        params:
            bool: True(default)
            list of random genes: [118,919,274,866,354,253,207,667,773,563,
                                   553,918,934,81,56,232,892,485,30,53]
            
    X: mxn matrix, m: number of compounds, n: size of descriptors
    Y: mxl matrix, m: number of compounds, l: 978
    """
    X = []
    Y = []
    perts = []
    LmGenes = []
    filepath = 'LmGenes.txt'
    with open(filepath) as fp:
        line = fp.readline()
        while line:
            LmGenes.append(line.strip())
            line = fp.readline()

    meta_smiles = pd.read_csv('meta_SMILES.csv')
    maccs = pd.read_csv('MACCS_bitmatrix.csv')
    shed = pd.read_csv('SHED.csv')
    cats2d = pd.read_csv('CATS2D.csv')
    jtvae = pd.read_csv('JTVAE.csv')
    random_index_list = [118,919,274,866,354,253,207,667,773,563,
                         553,918,934,81,56,232,892,485,30,53]

    if target == 'regression':
        if descriptors == 'ecfp':
            for pert_id in L[cell_line]:
                smiles = meta_smiles[meta_smiles['pert_id'] == pert_id]['SMILES'].values[0]
                if str(smiles) == 'nan' or str(smiles) == '-666':
                    continue
                mol = Chem.MolFromSmiles(smiles)
                fp_string = AllChem.GetMorganFingerprintAsBitVect(mol,4,nBits=1024).ToBitString()
                feature = (np.fromstring(fp_string,'u1') - ord('0')).tolist()
                labels = L[cell_line][pert_id]['chdirLm']
                X.append(feature)
                Y.append(labels)
                perts.append(pert_id)

        elif descriptors == 'topological':
            for pert_id in L[cell_line]:
                smiles = meta_smiles[meta_smiles['pert_id'] == pert_id]['SMILES'].values[0]
                if str(smiles) == 'nan' or str(smiles) == '-666':
                    continue
                mol = Chem.MolFromSmiles(smiles)
                fp_string = Chem.RDKFingerprint(mol, fpSize=1024).ToBitString()
                feature = (np.fromstring(fp_string,'u1') - ord('0')).tolist()
                labels = L[cell_line][pert_id]['chdirLm']
                X.append(feature)
                Y.append(labels)
                perts.append(pert_id)
                
        elif descriptors == 'shed':
            for pert_id in L[cell_line]:
                smiles = meta_smiles[meta_smiles['pert_id'] == pert_id]['SMILES'].values[0]
                if str(smiles) == 'nan' or str(smiles) == '-666':
                    continue
                mol = Chem.MolFromSmiles(smiles)
                smiles = Chem.rdmolfiles.MolToSmiles(mol, isomericSmiles=False)
                if shed[shed['SMILES'] == smiles].empty:
                    continue
                feature = shed[shed['SMILES'] == smiles].drop(['SMILES'], axis=1).values[0].tolist()
                labels = L[cell_line][pert_id]['chdirLm']
                X.append(feature)
                Y.append(labels)
                perts.append(pert_id)
                
        elif descriptors == 'cats2d':
            for pert_id in L[cell_line]:
                smiles = meta_smiles[meta_smiles['pert_id'] == pert_id]['SMILES'].values[0]
                if str(smiles) == 'nan' or str(smiles) == '-666':
                    continue
                mol = Chem.MolFromSmiles(smiles)
                smiles = Chem.rdmolfiles.MolToSmiles(mol, isomericSmiles=False)
                if cats2d[cats2d['SMILES'] == smiles].empty:
                    continue
                feature = cats2d[cats2d['SMILES'] == smiles].drop(['SMILES'], axis=1).values[0].tolist()
                labels = L[cell_line][pert_id]['chdirLm']
                X.append(feature)
                Y.append(labels)
                perts.append(pert_id)
    
        elif descriptors == 'jtvae':
            for pert_id in L[cell_line]:
                smiles = meta_smiles[meta_smiles['pert_id'] == pert_id]['SMILES'].values[0]
                if str(smiles) == 'nan' or str(smiles) == '-666':
                    continue
                mol = Chem.MolFromSmiles(smiles)
                smiles = Chem.rdmolfiles.MolToSmiles(mol, isomericSmiles=False)
                if jtvae[jtvae['SMILES'] == smiles].empty:
                    continue
                feature = jtvae[jtvae['SMILES'] == smiles].drop(['SMILES'], axis=1).values[0].tolist()
                labels = L[cell_line][pert_id]['chdirLm']
                X.append(feature)
                Y.append(labels)
                perts.append(pert_id)
        
        elif descriptors == 'maccs':
            for pert_id in L[cell_line]:
                feature = maccs[maccs['pert_id'] == pert_id].drop(['pert_id'], axis=1).values[0].tolist()
                labels = L[cell_line][pert_id]['chdirLm']
                X.append(feature)
                Y.append(labels)
                perts.append(pert_id)
    
    elif target == 'class':
        class_dict = {}
        for gene in LmGenes:
            class_dict.update({gene: 0})
                
        if descriptors == 'ecfp':
            for pert_id in L[cell_line]:
                smiles = meta_smiles[meta_smiles['pert_id'] == pert_id]['SMILES'].values[0]
                if str(smiles) == 'nan' or str(smiles) == '-666':
                    continue
                mol = Chem.MolFromSmiles(smiles)
                fp_string = AllChem.GetMorganFingerprintAsBitVect(mol,4,nBits=1024).ToBitString()
                feature = (np.fromstring(fp_string,'u1') - ord('0')).tolist()
                
                if ('dnGenes' not in L[cell_line][pert_id] or 
                    'upGenes' not in L[cell_line][pert_id]):
                        continue
                dn_genes = list(set(L[cell_line][pert_id]['dnGenes']))
                up_genes = list(set(L[cell_line][pert_id]['upGenes']))
                class_dict = dict.fromkeys(class_dict, 0)
                for gene in dn_genes:
                    if gene in class_dict:
                        class_dict.update({gene: -1})

                for gene in up_genes:
                    if gene in class_dict:
                        class_dict.update({gene: 1})

                labels = np.fromiter(class_dict.values(), dtype=int)
                X.append(feature)
                Y.append(labels)
                perts.append(pert_id)

        elif descriptors == 'topological':
            for pert_id in L[cell_line]:
                smiles = meta_smiles[meta_smiles['pert_id'] == pert_id]['SMILES'].values[0]
                if str(smiles) == 'nan' or str(smiles) == '-666':
                    continue
                mol = Chem.MolFromSmiles(smiles)
                fp_string = Chem.RDKFingerprint(mol, fpSize=1024).ToBitString()
                feature = (np.fromstring(fp_string,'u1') - ord('0')).tolist()
                
                if ('dnGenes' not in L[cell_line][pert_id] or 
                    'upGenes' not in L[cell_line][pert_id]):
                        continue               
                dn_genes = list(set(L[cell_line][pert_id]['dnGenes']))
                up_genes = list(set(L[cell_line][pert_id]['upGenes']))
                class_dict = dict.fromkeys(class_dict, 0)
                for gene in dn_genes:
                    if gene in class_dict:
                        class_dict.update({gene: -1})

                for gene in up_genes:
                    if gene in class_dict:
                        class_dict.update({gene: 1})

                labels = np.fromiter(class_dict.values(), dtype=int)
                X.append(feature)
                Y.append(labels)
                perts.append(pert_id)
                
        elif descriptors == 'shed':
            for pert_id in L[cell_line]:
                smiles = meta_smiles[meta_smiles['pert_id'] == pert_id]['SMILES'].values[0]
                if str(smiles) == 'nan' or str(smiles) == '-666':
                    continue
                mol = Chem.MolFromSmiles(smiles)
                smiles = Chem.rdmolfiles.MolToSmiles(mol, isomericSmiles=False)
                if shed[shed['SMILES'] == smiles].empty:
                    continue
                feature = shed[shed['SMILES'] == smiles].drop(['SMILES'], axis=1).values[0].tolist()

                if ('dnGenes' not in L[cell_line][pert_id] or 
                    'upGenes' not in L[cell_line][pert_id]):
                        continue              
                dn_genes = list(set(L[cell_line][pert_id]['dnGenes']))
                up_genes = list(set(L[cell_line][pert_id]['upGenes']))
                class_dict = dict.fromkeys(class_dict, 0)
                for gene in dn_genes:
                    if gene in class_dict:
                        class_dict.update({gene: -1})

                for gene in up_genes:
                    if gene in class_dict:
                        class_dict.update({gene: 1})

                labels = np.fromiter(class_dict.values(), dtype=int)
                X.append(feature)
                Y.append(labels)
                perts.append(pert_id)
                
        elif descriptors == 'cats2d':
            for pert_id in L[cell_line]:
                smiles = meta_smiles[meta_smiles['pert_id'] == pert_id]['SMILES'].values[0]
                if str(smiles) == 'nan' or str(smiles) == '-666':
                    continue
                mol = Chem.MolFromSmiles(smiles)
                smiles = Chem.rdmolfiles.MolToSmiles(mol, isomericSmiles=False)
                if cats2d[cats2d['SMILES'] == smiles].empty:
                    continue
                feature = cats2d[cats2d['SMILES'] == smiles].drop(['SMILES'], axis=1).values[0].tolist()
                
                if ('dnGenes' not in L[cell_line][pert_id] or 
                    'upGenes' not in L[cell_line][pert_id]):
                        continue               
                dn_genes = list(set(L[cell_line][pert_id]['dnGenes']))
                up_genes = list(set(L[cell_line][pert_id]['upGenes']))
                class_dict = dict.fromkeys(class_dict, 0)
                for gene in dn_genes:
                    if gene in class_dict:
                        class_dict.update({gene: -1})

                for gene in up_genes:
                    if gene in class_dict:
                        class_dict.update({gene: 1})

                labels = np.fromiter(class_dict.values(), dtype=int)
                X.append(feature)
                Y.append(labels)
                perts.append(pert_id)
                
        elif descriptors == 'jtvae':
            for pert_id in L[cell_line]:
                smiles = meta_smiles[meta_smiles['pert_id'] == pert_id]['SMILES'].values[0]
                if str(smiles) == 'nan' or str(smiles) == '-666':
                    continue
                mol = Chem.MolFromSmiles(smiles)
                smiles = Chem.rdmolfiles.MolToSmiles(mol, isomericSmiles=False)
                if jtvae[jtvae['SMILES'] == smiles].empty:
                    continue
                feature = jtvae[jtvae['SMILES'] == smiles].drop(['SMILES'], axis=1).values[0].tolist()
                
                if ('dnGenes' not in L[cell_line][pert_id] or 
                    'upGenes' not in L[cell_line][pert_id]):
                        continue               
                dn_genes = list(set(L[cell_line][pert_id]['dnGenes']))
                up_genes = list(set(L[cell_line][pert_id]['upGenes']))
                class_dict = dict.fromkeys(class_dict, 0)
                for gene in dn_genes:
                    if gene in class_dict:
                        class_dict.update({gene: -1})

                for gene in up_genes:
                    if gene in class_dict:
                        class_dict.update({gene: 1})

                labels = np.fromiter(class_dict.values(), dtype=int)
                X.append(feature)
                Y.append(labels)
                perts.append(pert_id)
        
        elif descriptors == 'maccs':
            for pert_id in L[cell_line]:
                feature = maccs[maccs['pert_id'] == pert_id].drop(['pert_id'], axis=1).values[0].tolist()
                
                if ('dnGenes' not in L[cell_line][pert_id] or 
                    'upGenes' not in L[cell_line][pert_id]):
                        continue              
                dn_genes = list(set(L[cell_line][pert_id]['dnGenes']))
                up_genes = list(set(L[cell_line][pert_id]['upGenes']))
                class_dict = dict.fromkeys(class_dict, 0)
                for gene in dn_genes:
                    if gene in class_dict:
                        class_dict.update({gene: -1})

                for gene in up_genes:
                    if gene in class_dict:
                        class_dict.update({gene: 1})

                labels = np.fromiter(class_dict.values(), dtype=int)
                X.append(feature)
                Y.append(labels)
                perts.append(pert_id)
    
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    x_columns = []
    if descriptors == 'ecfp':
        for i in range(X.shape[1]):
            x_columns.append('ecfp_' + str(i+1))
    elif descriptors == 'topological':
        for i in range(X.shape[1]):
            x_columns.append('topological_' + str(i+1))
    elif descriptors == 'maccs':
        for i in range(X.shape[1]):
            x_columns.append('maccs_' + str(i+1))
    elif descriptors == 'jtvae':
        for i in range(X.shape[1]):
            x_columns.append('jtvae_' + str(i+1))
    elif descriptors == 'shed':
        for i in range(X.shape[1]):
            x_columns.append('shed_' + str(i+1))
    elif descriptors == 'cats2d':
        for i in range(X.shape[1]):
            x_columns.append('cats2d_' + str(i+1))
    
    X = pd.DataFrame(X, index=perts, columns=x_columns)
    Y = pd.DataFrame(Y, index=perts)  
    folds = list(KFold(n_fold, shuffle = True, random_state=random_state).split(X))

    if random_genes:
        Y_random = []
        for i in random_index_list:
            Y_random.append(Y.iloc[:,i:i+1])
        df = Y_random[0]
        for i in range(len(Y_random)-1):
            df = pd.concat([df, Y_random[i+1]], axis=1)
        Y = df
  
    return (X, Y, folds)
