import os
import numpy as np
from dateutil import parser
import csv


def parse_PDB_biounits(x, atoms=['N','CA','C'], chain=None):
  '''
  input:  x = PDB filename
          atoms = atoms to extract (optional)
  output: (length, atoms, coords=(x,y,z)), sequence
  '''

  alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
  states = len(alpha_1)
  alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
             'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']
  
  aa_1_N = {a:n for n,a in enumerate(alpha_1)}
  aa_3_N = {a:n for n,a in enumerate(alpha_3)}
  aa_N_1 = {n:a for n,a in enumerate(alpha_1)}
  aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
  aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}
  
  def AA_to_N(x):
    # ["ARND"] -> [[0,1,2,3]]
    x = np.array(x);
    if x.ndim == 0: x = x[None]
    return [[aa_1_N.get(a, states-1) for a in y] for y in x]
  
  def N_to_AA(x):
    # [[0,1,2,3]] -> ["ARND"]
    x = np.array(x);
    if x.ndim == 1: x = x[None]
    return ["".join([aa_N_1.get(a,"-") for a in y]) for y in x]

  xyz,seq,min_resn,max_resn = {},{},1e6,-1e6
  for line in open(x,"rb"):
    line = line.decode("utf-8","ignore").rstrip()

    if line[:6] == "HETATM" and line[17:17+3] == "MSE":
      line = line.replace("HETATM","ATOM  ")
      line = line.replace("MSE","MET")

    if line[:4] == "ATOM":
      ch = line[21:22]
      if ch == chain or chain is None:
        atom = line[12:12+4].strip()
        resi = line[17:17+3]
        resn = line[22:22+5].strip()
        x,y,z = [float(line[i:(i+8)]) for i in [30,38,46]]

        if resn[-1].isalpha(): 
            resa,resn = resn[-1],int(resn[:-1])-1
        else: 
            resa,resn = "",int(resn)-1
#         resn = int(resn)
        if resn < min_resn: 
            min_resn = resn
        if resn > max_resn: 
            max_resn = resn
        if resn not in xyz: 
            xyz[resn] = {}
        if resa not in xyz[resn]: 
            xyz[resn][resa] = {}
        if resn not in seq: 
            seq[resn] = {}
        if resa not in seq[resn]: 
            seq[resn][resa] = resi

        if atom not in xyz[resn][resa]:
          xyz[resn][resa][atom] = np.array([x,y,z])

  # convert to numpy arrays, fill in missing values
  seq_,xyz_ = [],[]
  try:
      for resn in range(min_resn,max_resn+1):
        if resn in seq:
          for k in sorted(seq[resn]): seq_.append(aa_3_N.get(seq[resn][k],20))
        else: seq_.append(20)
        if resn in xyz:
          for k in sorted(xyz[resn]):
            for atom in atoms:
              if atom in xyz[resn][k]: xyz_.append(xyz[resn][k][atom])
              else: xyz_.append(np.full(3,np.nan))
        else:
          for atom in atoms: xyz_.append(np.full(3,np.nan))
      return np.array(xyz_).reshape(-1,len(atoms),3), N_to_AA(np.array(seq_))
  except TypeError:
      return 'no_chain', 'no_chain'

def parse_PDB(path_to_pdb, input_chain_list=None, ca_only=False):
    """
    Copy parse_PDB and parse_PDB_biounits from protein_mpnn_utils.py
    """
    c=0
    pdb_dict_list = []
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet
     
    if input_chain_list:
        chain_alphabet = input_chain_list  
 

    biounit_names = [path_to_pdb]
    for biounit in biounit_names:
        my_dict = {}
        s = 0
        concat_seq = ''
        concat_N = []
        concat_CA = []
        concat_C = []
        concat_O = []
        concat_mask = []
        coords_dict = {}
        for letter in chain_alphabet:
            if ca_only:
                sidechain_atoms = ['CA']
            else:
                sidechain_atoms = ['N', 'CA', 'C', 'O']
            xyz, seq = parse_PDB_biounits(biounit, atoms=sidechain_atoms, chain=letter)
            if type(xyz) != str:
                concat_seq += seq[0]
                my_dict['seq_chain_'+letter]=seq[0]
                coords_dict_chain = {}
                if ca_only:
                    coords_dict_chain['CA_chain_'+letter]=xyz.tolist()
                else:
                    coords_dict_chain['N_chain_' + letter] = xyz[:, 0, :].tolist()
                    coords_dict_chain['CA_chain_' + letter] = xyz[:, 1, :].tolist()
                    coords_dict_chain['C_chain_' + letter] = xyz[:, 2, :].tolist()
                    coords_dict_chain['O_chain_' + letter] = xyz[:, 3, :].tolist()
                my_dict['coords_chain_'+letter]=coords_dict_chain
                s += 1
        fi = biounit.rfind("/")
        my_dict['name']=biounit[(fi+1):-4]
        my_dict['num_of_chains'] = s
        my_dict['seq'] = concat_seq
        if s <= len(chain_alphabet):
            pdb_dict_list.append(my_dict)
            c+=1
    return pdb_dict_list


def get_pdbs_new(data_pn, data_path, max_length=10000, num_units=1000000):
    """
    This function follows get_pdbs in utils.py; but loading pdb data from .pdb files instead of .pt files.
    """
    seq_ids = data_pn["Sequence_ID"].tolist()
    neg_cols = [c for c in data_pn.columns if c.startswith("Neg_seq_ProteinMPNN_")]
    pdb_dict_list = []
    #random.shuffle(seq_ids)
    for seq_id in seq_ids:
        pdb_name, chain_id = seq_id.split('_')
        pdb_file = f'{pdb_name}.pdb'
        pdb_path = os.path.join(data_path, pdb_file) 
        pdb_dict = parse_PDB(pdb_path, input_chain_list=[chain_id])[0]
        pdb_dict['num_of_chains'] = 1
        pdb_dict['visible_list'] = []
        pdb_dict['masked_list'] = [chain_id]
        pdb_dict['name'] = seq_id
        
        row = data_pn[data_pn["Sequence_ID"] == seq_id].iloc[0]
        pos_seq = row['GT_sequence']
        neg_seqs = row[neg_cols].dropna().tolist()
        pdb_dict['seq'] = pos_seq
        pdb_dict[f'seq_chain_{chain_id}'] = pos_seq
        pdb_dict['neg_seqs']  = neg_seqs

        if len(pdb_dict['seq']) <= max_length:
            pdb_dict_list.append(pdb_dict)
        if len(pdb_dict_list) >= num_units:
            break
    return pdb_dict_list


def build_training_clusters(params, debug):
    val_ids = set([int(l) for l in open(params['VAL']).readlines()])
    test_ids = set([int(l) for l in open(params['TEST']).readlines()])
   
    if debug:
        val_ids = []
        test_ids = []

    rows = []
    with open(params['LIST'], 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for r in reader:
            if float(r[2]) <= params['RESCUT'] and parser.parse(r[1]) <= parser.parse(params['DATCUT']):
                rows.append([r[0], r[3], int(r[4])])
                if len(rows) >= 10000 and debug:
                    break
    
    # compile training and validation sets
    train = {}
    valid = {}
    test = {}

    if debug:
        rows = rows[:10000]
    for r in rows:
        if r[2] in val_ids:
            if r[2] in valid.keys():
                valid[r[2]].append(r[:2])
            else:
                valid[r[2]] = [r[:2]]
        elif r[2] in test_ids:
            if r[2] in test.keys():
                test[r[2]].append(r[:2])
            else:
                test[r[2]] = [r[:2]]
        else:
            if r[2] in train.keys():
                train[r[2]].append(r[:2])
            else:
                train[r[2]] = [r[:2]]
    if debug:
        valid=train       
    return train, valid, test


# class NewStructureLoader():
#     def __init__(self, dataset, batch_size=100, shuffle=True,
#         collate_fn=lambda x:x, drop_last=False):
#         self.dataset = dataset
#         self.size = len(dataset)
#         self.batch_size = batch_size  # Total sequence length per batch
#         self.shuffle = shuffle
#         self.drop_last = drop_last
#         self.collate_fn = collate_fn
#         # Pre-compute sequence lengths for efficiency
#         self.lengths = [len(dataset[i]['seq']) for i in range(self.size)]

#     def __len__(self):
#         return len(self._build_batches(shuffle=False))

#     def __iter__(self):
#         for batch_indices in self._build_batches(shuffle=self.shuffle):
#             batch = [self.dataset[j] for j in batch_indices]
#             yield self.collate_fn(batch)

#     def _build_batches(self, shuffle: bool):
#         indices = np.arange(self.size)
        
#         # Group indices by similar sequence lengths to reduce padding
#         # Use logarithmic binning to group sequences of similar sizes
#         bins = {}
#         for idx in indices:
#             length = self.lengths[idx]
#             # Use log2 binning to group similar sizes (reduces padding)
#             if length > 0:
#                 bin_key = int(np.log2(length + 1))
#             else:
#                 bin_key = 0
#             if bin_key not in bins:
#                 bins[bin_key] = []
#             bins[bin_key].append(idx)
        
#         # Shuffle within each bin for randomness
#         if shuffle:
#             for bin_key in bins:
#                 np.random.shuffle(bins[bin_key])
        
#         # Create batches from each bin separately, then collect all batches
#         all_batches = []
        
#         for bin_key in sorted(bins.keys()):
#             bin_indices = bins[bin_key]
#             batch = []
#             batch_total_length = 0
            
#             for idx in bin_indices:
#                 seq_length = self.lengths[idx]
#                 if seq_length > self.batch_size:
#                     if len(batch) > 0:
#                         all_batches.append(batch)
#                         batch = []
#                         batch_total_length = 0
#                     all_batches.append([idx])
#                     continue

#                 # Check if adding this sample would exceed batch_size
#                 if batch_total_length + seq_length > self.batch_size and len(batch) > 0:
#                     # Save current batch and start a new one
#                     all_batches.append(batch)
#                     batch = []
#                     batch_total_length = 0

#                 # Add to current batch
#                 batch.append(idx)
#                 batch_total_length += seq_length
            
#             # Add the last batch from this bin
#             if len(batch) > 0:
#                 if not self.drop_last:
#                     all_batches.append(batch)
        
#         # Shuffle the order of batches for randomness across epochs
#         if shuffle:
#             np.random.shuffle(all_batches)
        
#         return all_batches
# import numpy as np
# from typing import Callable, List, Any

# class NewStructureLoader():
#     """
#     Dynamic token/length-based batching.
#     - batch_size means: max(sum of (max_len_in_batch) * batch_count) ~ max_len * n <= batch_size
#       (same logic as your original: size * (len(batch)+1) <= batch_size)
#     - Batches are rebuilt every __iter__ call (not fixed at init).
#     """
#     def __init__(
#         self,
#         dataset,
#         batch_size: int = 100,
#         shuffle: bool = True,
#         collate_fn: Callable[[List[Any]], Any] = lambda x: x,
#         drop_last: bool = False,
#         sortish_window: int = 50,   # larger => more globally sorted; smaller => more randomness
#         seed: int | None = None,
#     ):
#         self.dataset = dataset
#         self.size = len(dataset)
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.collate_fn = collate_fn
#         self.drop_last = drop_last
#         self.sortish_window = max(1, int(sortish_window))
#         self.rng = np.random.default_rng(seed)

#         # cache lengths only (safe; doesn't fix batching)
#         self.lengths = np.array([len(dataset[i]["seq"]) for i in range(self.size)], dtype=np.int32)
#         print('max length of sequences: ', max(self.lengths))

#     def _make_batches(self, indices: np.ndarray) -> List[List[int]]:
#         batches: List[List[int]] = []
#         batch: List[int] = []
#         max_len = 0

#         for ix in indices:
#             L = int(self.lengths[ix])
#             new_max = max(max_len, L)

#             # original constraint: max_len_in_batch * batch_count <= batch_size
#             if (new_max * (len(batch) + 1)) <= self.batch_size:
#                 batch.append(int(ix))
#                 max_len = new_max
#             else:
#                 if batch:
#                     batches.append(batch)
#                 batch = [int(ix)]
#                 max_len = L

#         if batch and not self.drop_last:
#             batches.append(batch)

#         return batches

#     def __iter__(self):
#         # 1) start from all indices
#         indices = np.arange(self.size)

#         # 2) optional shuffle + "sortish" to avoid fixed batches
#         #    shuffle within windows after sorting by length (changes each epoch)
#         if self.shuffle:
#             self.rng.shuffle(indices)
#             # sort by length, then shuffle inside windows to inject randomness
#             indices = indices[np.argsort(self.lengths[indices], kind="mergesort")]

#             if self.sortish_window > 1:
#                 chunks = [indices[i:i+self.sortish_window] for i in range(0, len(indices), self.sortish_window)]
#                 for c in chunks:
#                     self.rng.shuffle(c)
#                 indices = np.concatenate(chunks)

#         else:
#             # deterministic: length-sorted traversal
#             indices = indices[np.argsort(self.lengths, kind="mergesort")]

#         # 3) build batches fresh every epoch
#         batches = self._make_batches(indices)

#         # 4) shuffle batch order (still not fixing membership because membership rebuilt above)
#         if self.shuffle:
#             self.rng.shuffle(batches)

#         for b_idx in batches:
#             batch = [self.dataset[i] for i in b_idx]
#             yield self.collate_fn(batch)

#     def __len__(self):
#         # Dynamic batching => exact length depends on shuffling.
#         # Return a deterministic approximation by using length-sorted order.
#         indices = np.arange(self.size)
#         indices = indices[np.argsort(self.lengths, kind="mergesort")]
#         return len(self._make_batches(indices))
import numpy as np
from typing import Callable, List, Any

class NewStructureLoader():
    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        collate_fn: Callable[[List[Any]], Any] = lambda x: x,
        drop_last: bool = False,
        seed: int | None = None,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.collate_fn = collate_fn
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = np.arange(len(self.dataset))

        if self.shuffle:
            self.rng.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start : start + self.batch_size]

            if self.drop_last and len(batch_idx) < self.batch_size:
                continue

            batch = [self.dataset[i] for i in batch_idx]
            yield self.collate_fn(batch)
# import numpy as np

# class NewStructureLoader:
#     def __init__(self, dataset, batch_size=100, shuffle=True,
#                  collate_fn=lambda x: x, drop_last=False, seed=None):
#         self.dataset = dataset
#         self.size = len(dataset)
#         self.batch_size = batch_size   # interpreted as B * L_max budget
#         self.shuffle = shuffle
#         self.drop_last = drop_last
#         self.collate_fn = collate_fn
#         self.seed = seed

#         # Optional: precompute lengths only (cheap). Remove if you want fully lazy.
#         self.lengths = [len(dataset[i]["seq"]) for i in range(self.size)]

#     def __iter__(self):
#         indices = np.arange(self.size)
#         if self.shuffle:
#             rng = np.random.default_rng(self.seed)
#             rng.shuffle(indices)

#         batch_idx = []
#         batch_max = 0  # current L_max in batch

#         for ix in indices:
#             L = self.lengths[ix]
#             new_max = max(batch_max, L)
#             new_B = len(batch_idx) + 1

#             # Enforce padding-aware budget: B * L_max <= batch_size
#             if new_max * new_B <= self.batch_size:
#                 batch_idx.append(ix)
#                 batch_max = new_max
#             else:
#                 # flush current batch
#                 if batch_idx or not self.drop_last:
#                     batch = [self.dataset[i] for i in batch_idx]
#                     yield self.collate_fn(batch)

#                 # start new batch with current sample
#                 batch_idx = [ix]
#                 batch_max = L

#         if batch_idx and not self.drop_last:
#             batch = [self.dataset[i] for i in batch_idx]
#             yield self.collate_fn(batch)

#     def __len__(self):
#         # Not exact under shuffling + online packing. Return a rough estimate.
#         # If you prefer, you can omit __len__ entirely.
#         total = sum(self.lengths)
#         avg_len = total / max(1, self.size)
#         # expected batch size ~ batch_size / avg_len
#         est_batches = int(np.ceil(self.size / max(1, self.batch_size / max(1.0, avg_len))))
#         return max(1, est_batches)
