import pandas as pd

import torch
import torch.utils.data

from src.utils import find_pdb_path, get_coords_and_sequence

class ESMIFPreferenceDataset(torch.utils.data.Dataset):
    
    def __init__(self, pdb_folders_path, csv_file):
        self.pdb_folders_path = [pdb_folders_path] if isinstance(pdb_folders_path, str) else pdb_folders_path
        
        self.data = pd.read_csv(csv_file)
        self.neg_columns = [c for c in self.data.columns if c.startswith("Neg_seq_")]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a dict with the following entires
         - sequence_id : Str (sequence identifier)
         - coords : np.array (3D coordinates)
         - positive_seq : Str (positive sequence)
         - all_negatives : List of Str (negative sequences)
        """
        row = self.data.iloc[idx]

        # 1) Load Positive and Negative sequences
        positive_seq = str(row['GT_sequence'])
        negative_seq = row[self.neg_columns].dropna().tolist()

        # 2) Load Coordinates
        seq_id = str(row['Sequence_ID'])
        if '_' not in seq_id:                                          
            raise ValueError(f"[Warning] Sequence_ID format invalid: {seq_id}")
        pdb_name, chain_id = seq_id.split('_')
        pdb_file = f'{pdb_name}.pdb'

        fpath = find_pdb_path(self.pdb_folders_path, pdb_file)
        coords, _ = get_coords_and_sequence(fpath, chain_id)
        
        item = {
            'sequence_id': seq_id,
            'coords': coords,
            'positive': positive_seq,
            'all_negative': negative_seq,
        }  
        return item

def collate_fn(batch):
    return {
        'sequence_ids': [b['sequence_id'] for b in batch],
        'coords': [b['coords'] for b in batch],
        'positives': [b['positive'] for b in batch],
        'all_negatives': [b['all_negative'] for b in batch]
    }