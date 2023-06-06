# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 14:42:06 2022

@author: 73273
"""


import pandas as pd
import numpy as np
import gudhi
import math


def extract_windowsize_info(filename, sites:list, size=20):
    """extract the corresponding infos within a given window size from pdb file
    """
    res_positions = [[(i + site) for i in range(-size, size+1)] for site in sites]  
    df_infos = []
    
    for i in range(len(sites)):
        temp_positions = res_positions[i]
        dict_info = {'res_name':[], 'position':[], 'x':[], 'y':[], 'z': [], 'element':[]}
        
        with open(filename) as f:
            contents = f.readlines()
            for content in contents:
                content = content.strip()
                # print(content)
                if (content[:4] == 'ATOM') and (int(content[22:26]) in temp_positions):
                    dict_info['res_name'].append(content[17:20])
                    dict_info['position'].append(content[22:26])
                    dict_info['x'].append(content[30:38])
                    dict_info['y'].append(content[38:46])
                    dict_info['z'].append(content[46:54])
                    dict_info['element'].append(content[-1])
        df_info = pd.DataFrame.from_dict(dict_info)
        df_infos.append(df_info)
    return  df_infos  


def computePH4sequence(df_info):
    """PH results from the filtration of VR complex 
    """
    coords = df_info[['x', 'y', 'z']].values.astype('float')
    rips = gudhi.RipsComplex(points=coords, max_edge_length=7)
    st = rips.create_simplex_tree(max_dimension=3)
    pers = st.persistence()
    
    dict_PH_info = {'dim': [], 'birth time': [], 'death time': []}
    for dim, (bt, dt) in pers:
        dict_PH_info['dim'].append(dim)
        dict_PH_info['birth time'].append(bt)
        dict_PH_info['death time'].append(dt)
    df_PH_info = pd.DataFrame(dict_PH_info, index=np.arange(len(dict_PH_info['dim'])))
    
    return df_PH_info  # dataframe: dim, birth time and death time


def computePH_AlphaComplex(df_info):
    """PH results from the filtration of Alpha complex
    """
    coords = df_info[['x', 'y', 'z']].values.astype('float')
    ac = gudhi.AlphaComplex(points=coords)
    stree = ac.create_simplex_tree()
    pers = stree.persistence()
    
    dict_acPH_info = {'dim': [], 'birth time': [], 'death time': []}
    for dim, (bt, dt) in pers:
        if dim != 0:
            dict_acPH_info['dim'].append(dim)
            dict_acPH_info['birth time'].append(bt)
            dict_acPH_info['death time'].append(dt)
    df_acPH_info = pd.DataFrame(dict_acPH_info, index=np.arange(len(dict_acPH_info['dim'])))
    return df_acPH_info


def compute0_PBNs(ph_info, fractions=[1.2 ,1.3, 1.4, 1.5, 1.6, 2.0]):
    """
    Topological features based on 0-bars from the filtration of VR complex
    """
    condition_0 = ph_info['dim'] == 0
    df_PH_0 = ph_info[condition_0]  

    pbn_0s = np.array([])
    for i in range(len(fractions) - 1):
        l, r = fractions[i], fractions[i+1]
        check = (df_PH_0['death time'] > l) & (df_PH_0['death time'] <= r)
        # log(PBNs+1)
        pbn_0 = np.log10(df_PH_0[check.values]['death time'].values.astype('float').shape[0] + 1)
        pbn_0s = np.append(pbn_0s, pbn_0)
    return pbn_0s


def barcode_statistics(ph_info, d_list:list, inftonum=7):
    """Barcode statistics
    """
    features = np.array([])
    for d in d_list:
        d_ph_info = ph_info[ph_info['dim'] == d]
        if d_ph_info.empty is not True:
            birth_vals = d_ph_info['birth time'].values

            sum_b, max_b, min_b = np.sum(birth_vals), np.max(birth_vals), np.min(birth_vals)
            avg_b, std_b = np.mean(birth_vals), np.std(birth_vals)

            if math.inf in d_ph_info['death time'].values:
                inf_ids = d_ph_info[d_ph_info['death time'] == math.inf].index.tolist()
                for id in inf_ids:
                    d_ph_info.loc[id, 'death time'] = inftonum
            modified_death_vals = d_ph_info['death time'].values

            sum_d, max_d, min_d = np.sum(modified_death_vals), np.max(modified_death_vals), np.min(modified_death_vals)
            avg_d, std_d = np.mean(modified_death_vals), np.std(modified_death_vals)

            bar_lengths = np.array(d_ph_info.loc[:, 'death time'].values.astype('float')) - np.array(d_ph_info.loc[:, 'birth time'].values.astype('float'))
            bar_lengths.sort()
            sum_l = np.sum(bar_lengths)
            max_l = bar_lengths[-1]
            min_l = bar_lengths[0]
            avg_l = np.mean(bar_lengths)
            std_l = np.std(bar_lengths)
            d_features = np.array([sum_l, max_l, min_l, avg_l, std_l,
                                   sum_b, max_b, min_b, avg_b, std_b,
                                   sum_d, max_d, min_d, avg_d, std_d])
            features = np.append(features, d_features)
        else:
            features = np.append(features, np.zeros(15, ))
    return features


def localRegion(df_info, site):
    """
    Topological features based on the local region of a given peptide
    """
    
    positions = [site + i for i in [-2, -1, 0, 1, 2]]
    sliced_infos_list = []
    for position in positions:
        sliced_info = df_info[df_info['position'] == position]
        sliced_infos_list.append(sliced_info)
    sliced_infos = pd.concat(sliced_infos_list, axis=0)
    sliced_ph = computePH4sequence(sliced_infos)
    sliced_0_PBNs = compute0_PBNs(sliced_ph, fractions=[1.2, 1.3, 1.4, 1.5, 1.6])  # (4)
    if len(barcode_statistics(sliced_ph, [1])) > 0:
        sliced_1_PH = barcode_statistics(sliced_ph, [1])  #（15）
    else:
        sliced_1_PH = np.zeros(15, )
        
    C_infos = sliced_infos[sliced_infos['element'] == 'C']
    C_ph = computePH4sequence(C_infos)
    C_0_PBNs = compute0_PBNs(C_ph, fractions=[1.5, 2, 2.5, 3])  # (3)
    if len(barcode_statistics(C_ph, [1])) > 0:
        C_1_PH = barcode_statistics(C_ph, [1])  # (15)
    else:
        C_1_PH = np.zeros(15, )
    
    N_infos = sliced_infos[sliced_infos['element'] == 'N']
    N_ph = computePH4sequence(N_infos)
    N_0_PBNs = compute0_PBNs(N_ph, fractions=[0, 10])  # (1个)
    
    return np.r_[sliced_0_PBNs, sliced_1_PH, C_0_PBNs, C_1_PH, N_0_PBNs]


def aa_TFs(df_info, site, fraction4aa=[1.25, 1.5, 1.75], size=20):
    """
    Topological features based on each residue (excluding the central lysine K) of a given peptide
    """
    positions = df_info['position'].unique()
    positions = np.delete(positions, np.where(positions==site))
    aa_features = np.array([])
    for position in positions:
        df_given_position = df_info[df_info['position'] == position]  # residue的原子信息
        df_aa_ph = computePH4sequence(df_given_position)
        
        n_fraction = len(fraction4aa) - 1
        aa_0_PBNS = compute0_PBNs(df_aa_ph, fractions=fraction4aa)
        aa_features = np.append(aa_features, aa_0_PBNS)

        fea0_0 = df_aa_ph[df_aa_ph['dim'] == 0].values.shape[0]
        
        inf_id = df_aa_ph[df_aa_ph['death time'] == math.inf].index.tolist()[0]
        df_aa_ph_noinf = df_aa_ph.drop(index=inf_id, axis=0, inplace=False)
        df_aa_ph_noinf0 = df_aa_ph_noinf[df_aa_ph_noinf['dim'] == 0]
        fea0_1 = np.sum(df_aa_ph_noinf0['death time'].values - df_aa_ph_noinf0['birth time'].values)
        
        df_ph1 = df_aa_ph[df_aa_ph['dim']==1]
        if df_ph1.empty:
            fea1_0 = 0
        else:
            betti1_lengths = df_ph1['death time'].values - df_ph1['birth time'].values
            fea1_0 = np.sum(betti1_lengths)
            
        df_ph2 = df_aa_ph[df_aa_ph['dim']==2]
        if df_ph2.empty:
            fea2_0 = 0
        else:
            betti2_lengths = df_ph2['death time'].values - df_ph2['birth time'].values
            fea2_0 = np.sum(betti2_lengths)
            
        other_features = np.array([fea0_0, fea0_1, fea1_0, fea2_0])
        aa_features = np.append(aa_features, other_features)
        
    if positions.shape[0] == 2*size:
        return  aa_features
    else:  
        check1 = np.where(positions < site, 1, 0)
        check2 = np.where(positions > site, 1, 0)
        if (np.sum(check1) < size) and (np.sum(check2) == size):
            add_0s = [0 for i in range((n_fraction+4)*(size - np.sum(check1)))]
            aa_features = np.insert(aa_features, 0, np.array(add_0s))
            return aa_features
        elif (np.sum(check1) == size) and (np.sum(check2) < size):
            add_0s = [0 for i in range((n_fraction+4)*(size - np.sum(check2)))]
            aa_features = np.append(aa_features, np.array(add_0s))
            return aa_features


def extractTFsfromAllAtoms(all_atom_ph_info_filepath, fractions1=[1.5, 2.7, 3.5, 4.5, 5, 6.7],
                           fractions2=[2.4, 2.9, 5.5, 6.7]):
    """
    Topological features based on 1-bars and 2-bars from the filtration of VR complex 
    """
    n1 = len(fractions1)
    n2 = len(fractions2)
    atom_ph = pd.read_csv(all_atom_ph_info_filepath, engine='python')
    
    atom_ph0 = atom_ph[atom_ph['dim'] == 0]
    inf0_id = atom_ph0[atom_ph0['death time'] == math.inf].index.tolist()[0]
    noinf_ph0 = atom_ph0.drop(index=inf0_id, axis=0, inplace=False)
    bar0_lengths = noinf_ph0['death time'].values - noinf_ph0['birth time'].values
    bar0_lengths.sort()
    fea0_0 = bar0_lengths[-1]
    fea0_1 = bar0_lengths[-2]
    fea0_2 = np.sum(bar0_lengths)
    fea0_3 = np.mean(bar0_lengths)
    fea0 = np.array([fea0_0, fea0_1, fea0_2, fea0_3])
    
    atom_ph1 = atom_ph[atom_ph['dim'] == 1]
    if atom_ph1.empty is False:
        bar1_lengths = atom_ph1['death time'].values - atom_ph1['birth time'].values
        longest1bar_rowid  = np.where(bar1_lengths==np.max(bar1_lengths))[0]
        fea1_0 = atom_ph1.iloc[longest1bar_rowid]['birth time'].values[0]

        pbn_1s = np.array([])
        for i in range(len(fractions1) - 1):
            l, r = fractions1[i], fractions1[i+1]
            check = (atom_ph1['death time'] > l) & (atom_ph1['death time'] <= r)
            # log(PBNs+1)
            pbn_1 = np.log10(atom_ph1[check.values]['death time'].values.astype('float').shape[0] + 1)
            pbn_1s = np.append(pbn_1s, pbn_1)
        fea1 = np.insert(pbn_1s, 0, fea1_0)
    else:
        fea1 =np.zeros(n1, )
        
    atom_ph2 = atom_ph[atom_ph['dim'] == 2]
    if atom_ph2.empty is False:
        pbn_2s = np.array([])
        for i in range(len(fractions2) - 1):
            l, r = fractions2[i], fractions2[i+1]
            check = (atom_ph2['death time'] > l) & (atom_ph2['death time'] <= r)
            # log(PBNs+1)
            pbn_2 = np.log10(atom_ph2[check.values]['death time'].values.astype('float').shape[0] + 1)
            pbn_2s = np.append(pbn_2s, pbn_2)
        fea2 = pbn_2s
    else:
        fea2 = np.zeros(n2-1, )
        
    return np.r_[fea0, fea1, fea2]
