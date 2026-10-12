from logging import raiseExceptions
import pandas as pd
import os
import torch
import numpy as np
import pickle
import json


aa_list = list("ACDEFGHIKLMNPQRSTVWY")
# ProteinMPNN's standard vocabulary
alphabet = 'ACDEFGHIKLMNPQRSTVWYX'  # X is mask/unknown
# Usually mapped as:
vocab = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
    'X': 20  # mask token
}
aa_indices = [vocab[a] for a in aa_list]

data_file = '/scratch/prj0000000224/yinghua_proj/ProteinMPNN/data/test_protein_mpnn_final.csv'
data_df = pd.read_csv(data_file)
critical_pos_file = '/scratch/prj0000000224/yinghua_proj/ProteinMPNN/data/critical_positions_final.csv'
critic_df = pd.read_csv(critical_pos_file)
seq_ids = critic_df['Sequence_ID']

logp_matrix_path = '/scratch/prj0000000224/yinghua_proj/ProteinMPNN/INFERENCE/PDB_FINAL/bn0.2/single/base/log_chain_probs'
logp_matrix_path_ = '/scratch/prj0000000224/yinghua_proj/ProteinMPNN/INFERENCE/PDB_FINAL/bn0.2/multi/base/log_chain_probs'

logp_matrix_path = '/scratch/prj0000000224/yinghua_proj/ProteinMPNN/INFERENCE/PDB_FINAL/mn0.2/single/mn0.2_bn0.1_lambda0.2_single/log_chain_probs'
logp_matrix_path_ = '/scratch/prj0000000224/yinghua_proj/ProteinMPNN/INFERENCE/PDB_FINAL/mn0.2/multi/mn0.2_bn0.1_lambda0.2_multi/log_chain_probs'

entropy_dict = []
critical_entropy_per_pos_all = []
noncritical_entropy_per_pos_all = []
for seq_id in seq_ids:
    try:
        seq = data_df.loc[data_df['Sequence_ID']==seq_id, 'GT_sequence'].values
        critical_idx = critic_df.loc[critic_df['Sequence_ID'] == seq_id, 'Critical_Idx_MPNN'].values
        if pd.isna(critical_idx[0]):
            print(seq_id)
            continue
        idx_array = np.array([int(x) for x in critical_idx[0].split(',')], dtype=int)
        logp_matrix_file = os.path.join(logp_matrix_path, seq_id+'.csv')
        if not os.path.exists(logp_matrix_file):
            logp_matrix_file = os.path.join(logp_matrix_path_, seq_id+'.csv')
        logp_all = pd.read_csv(logp_matrix_file)
        logp_matrix = logp_all.values
        logp_matrix = logp_matrix[:, aa_indices]  # only consider common aa
        #print(logp_matrix.shape)

        critical_logp_matrix = logp_matrix[idx_array]
        critical_prob_matrix = np.exp(critical_logp_matrix)
        total_critical_entropy = -np.sum(critical_prob_matrix * np.log2(critical_prob_matrix + 1e-10), axis=1)
        critical_entropy_per_pos = total_critical_entropy.mean()

        all_indices = np.arange(logp_matrix.shape[0])
        mask = ~np.isin(all_indices, idx_array)
        noncritical_logp_matrix = logp_matrix[mask]
        noncritical_prob_matrix =  np.exp(noncritical_logp_matrix)
        total_noncritical_entropy = -np.sum(noncritical_prob_matrix * np.log2(noncritical_prob_matrix + 1e-10), axis=1)
        noncritical_entropy_per_pos = total_noncritical_entropy.mean()

        entropy_dict.append(
            {'Sequence_ID': seq_id,
            'GT_sequence': seq,
            'Log_prob_matrix': np.array(logp_matrix),
            'Critical_pos': np.array(idx_array),
            'Critical_entropy_per_pos': float(critical_entropy_per_pos),
            'Noncritical_entropy_per_pos': float(noncritical_entropy_per_pos)}
        )
        critical_entropy_per_pos_all.append(critical_entropy_per_pos)
        noncritical_entropy_per_pos_all.append(noncritical_entropy_per_pos)

    except IndexError:
        print(f"ID {seq_id} not found in the data")
    
# df = pd.DataFrame(entropy_dict)
# df.to_csv(logp_matrix_path+'/Baseline_T0.1_entropy.csv', index=False)
# print(len(df))
# with open(logp_matrix_path+'/Baseline_T0.1_entropy.pkl', 'wb') as f:
#     pickle.dump(entropy_dict, f)
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

# Save to JSON
#with open(logp_matrix_path+'/20260131_143538_033_T0.1_entropy.json', 'w') as f:
with open('/scratch/prj0000000224/yinghua_proj/ProteinMPNN/INFERENCE/PDB_FINAL/mn0.2/mn0.2_bn0.1_lambda0.2_entropy.json', 'w') as f:
    json.dump(entropy_dict, f, cls=NumpyEncoder, indent=2)

print('Size of entropy dict: ', len(entropy_dict), len(critical_entropy_per_pos_all), len(noncritical_entropy_per_pos_all))
print('critical_entropy_per_pos: ', np.mean(critical_entropy_per_pos_all), np.std(critical_entropy_per_pos_all))
print('noncritical_entropy_per_pos: ', np.mean(noncritical_entropy_per_pos_all), np.std(noncritical_entropy_per_pos_all))
