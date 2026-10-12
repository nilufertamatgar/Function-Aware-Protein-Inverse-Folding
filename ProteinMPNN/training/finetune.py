import argparse
import os.path
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import json, time, os, sys, glob
import shutil
import warnings
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import queue
import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import os.path
import subprocess
from concurrent.futures import ProcessPoolExecutor    
from utils import worker_init_fn, loader_pdb, get_pdbs, build_training_clusters, PDB_dataset, StructureDataset, StructureLoader
from model_utils import featurize, loss_smoothed, loss_nll, get_std_opt, ProteinMPNN
from utils_new import get_pdbs_new, NewStructureLoader


def get_mask_sppo_loss_old(logp_pos, logp_neg, ref_logp_pos, ref_logp_neg, S_pos, S_neg, mask, beta=0.2):
    """
    Masked SPPO loss — focuses only on differing residues between positive and negative sequences.
    """
    logp_pos = logp_pos.gather(-1, S_pos.unsqueeze(-1)).squeeze(-1)
    ref_logp_pos = ref_logp_pos.gather(-1, S_pos.unsqueeze(-1)).squeeze(-1)
    logp_neg = logp_neg.gather(-1, S_neg.unsqueeze(-1)).squeeze(-1)
    ref_logp_neg = ref_logp_neg.gather(-1, S_neg.unsqueeze(-1)).squeeze(-1)
    diff_mask = (S_pos != S_neg).float()  # [B, L]

    a = beta*(logp_pos-ref_logp_pos)
    b = beta*(logp_neg-ref_logp_neg)

    mask = mask*diff_mask
    loss = (a - 0.5) ** 2 + (b + 0.5) ** 2
    loss = torch.sum(loss * mask, dim=1) / (mask.sum(dim=1).clamp(min=1))
    loss = loss.mean()

    return loss


def get_mask_sppo_loss(logp_pos, logp_neg, ref_logp_pos, ref_logp_neg, S_pos, S_neg, mask, beta=0.2):
    """
    Masked SPPO loss — focuses only on differing residues between positive and negative sequences.
    """
    logp_pos = logp_pos.gather(-1, S_pos.unsqueeze(-1)).squeeze(-1)
    ref_logp_pos = ref_logp_pos.gather(-1, S_pos.unsqueeze(-1)).squeeze(-1)
    logp_neg = logp_neg.gather(-1, S_neg.unsqueeze(-1)).squeeze(-1)
    ref_logp_neg = ref_logp_neg.gather(-1, S_neg.unsqueeze(-1)).squeeze(-1)
    diff_mask = (S_pos != S_neg).float()  # [B, L]

    mask = mask*diff_mask
    logp_pos_mask = torch.sum(logp_pos*mask, dim=1)/(mask.sum(dim=1).clamp(min=1))
    ref_logp_pos_mask = torch.sum(ref_logp_pos*mask, dim=1)/(mask.sum(dim=1).clamp(min=1))
    logp_neg_mask = torch.sum(logp_neg*mask, dim=1)/(mask.sum(dim=1).clamp(min=1))
    ref_logp_neg_mask = torch.sum(ref_logp_neg*mask, dim=1)/(mask.sum(dim=1).clamp(min=1))

    a = beta*(logp_pos_mask-ref_logp_pos_mask)
    b = beta*(logp_neg_mask-ref_logp_neg_mask)
    
    loss = (a - 0.5) ** 2 + (b + 0.5) ** 2
    loss = loss.mean()

    return loss


def get_sppo_loss(logp_pos, logp_neg, ref_logp_pos, ref_logp_neg, S_pos, S_neg, mask, beta=0.2):
    """
    Masked SPPO loss — focuses only on differing residues between positive and negative sequences.
    """
    logp_pos = logp_pos.gather(-1, S_pos.unsqueeze(-1)).squeeze(-1)
    ref_logp_pos = ref_logp_pos.gather(-1, S_pos.unsqueeze(-1)).squeeze(-1)
    logp_neg = logp_neg.gather(-1, S_neg.unsqueeze(-1)).squeeze(-1)
    ref_logp_neg = ref_logp_neg.gather(-1, S_neg.unsqueeze(-1)).squeeze(-1)

    logp_pos = (logp_pos*mask).sum(-1)
    ref_logp_pos = (ref_logp_pos*mask).sum(-1)
    logp_neg = (logp_neg*mask).sum(-1)
    ref_logp_neg = (ref_logp_neg*mask).sum(-1)

    a = beta*(logp_pos-ref_logp_pos)
    b = beta*(logp_neg-ref_logp_neg)

    loss = (a - 0.5) ** 2 + (b + 0.5) ** 2
    loss = loss.mean()

    return loss


def get_dpo_loss(logp_pos, logp_neg, ref_logp_pos, ref_logp_neg, S_pos, S_neg, mask, beta=0.2):
    """
    DPO loss — focuses only on differing residues between positive and negative sequences.
    """

    logp_pos = logp_pos.gather(-1, S_pos.unsqueeze(-1)).squeeze(-1)
    ref_logp_pos = ref_logp_pos.gather(-1, S_pos.unsqueeze(-1)).squeeze(-1)
    logp_neg = logp_neg.gather(-1, S_neg.unsqueeze(-1)).squeeze(-1)
    ref_logp_neg = ref_logp_neg.gather(-1, S_neg.unsqueeze(-1)).squeeze(-1)

    logp_pos = (logp_pos*mask).sum(-1)
    ref_logp_pos = (ref_logp_pos*mask).sum(-1)
    logp_neg = (logp_neg*mask).sum(-1)
    ref_logp_neg = (ref_logp_neg*mask).sum(-1)

    a = beta*(logp_pos-ref_logp_pos)
    b = beta*(logp_neg-ref_logp_neg)

    # DPO loss
    loss = -F.logsigmoid(a-b)

    return loss.mean()


def get_total_loss(logp_pos, logp_neg, ref_logp_pos, ref_logp_neg, pos_S, neg_S, mask_for_loss, alpha, beta, loss_type='mask_sppo_ce'):
    if loss_type == 'mask_sppo_ce':
        sppo_loss = get_mask_sppo_loss(
                logp_pos, logp_neg, ref_logp_pos, ref_logp_neg, pos_S, neg_S, mask_for_loss, beta=beta
            ) 
        _, loss_av_smoothed = loss_smoothed(pos_S, logp_pos, mask_for_loss)
        total_loss = sppo_loss + args.alpha * loss_av_smoothed
    elif loss_type == 'sppo':
        total_loss = get_sppo_loss(
                logp_pos, logp_neg, ref_logp_pos, ref_logp_neg, pos_S, neg_S, mask_for_loss, beta=beta
            ) 
    elif loss_type == 'dpo':
        total_loss = get_dpo_loss(
                logp_pos, logp_neg, ref_logp_pos, ref_logp_neg, pos_S, neg_S, mask_for_loss, beta=beta
            ) 
    elif loss_type == 'ce':
        _, loss_av_smoothed = loss_smoothed(pos_S, logp_pos, mask_for_loss)
        total_loss = loss_av_smoothed
    else:
        raise NotImplementedError
    
    return total_loss

    

def featurize_S_only(seqs, device, alphabet: str = 'ACDEFGHIKLMNPQRSTVWYX'):
    """
    Featurize only the sequence tokens (S) from `batch` using each entry's `b['seq']`.

    Returns:
      - S: LongTensor [B, L_max] with alphabet indices (padded with 0s)
    """
    B = len(seqs)
    if B == 0:
        return torch.zeros((0, 0), dtype=torch.long, device=device)

    # Use `b['seq']` directly (no chain re-ordering, no coords)
    lengths = [len(s) for s in seqs]
    L_max = max(lengths)
    S = np.zeros([B, L_max], dtype=np.int32)

    for i, seq in enumerate(seqs):
        # Convert to labels (unknown chars -> 'X')
        indices = np.asarray([alphabet.index(a) if a in alphabet else alphabet.index('X') for a in seq], dtype=np.int32)
        S[i, : len(seq)] = indices

    return torch.from_numpy(S).to(dtype=torch.long, device=device)


def main(args):

    scaler = torch.cuda.amp.GradScaler()
     
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    base_folder = time.strftime(args.path_for_outputs, time.localtime())

    if base_folder[-1] != '/':
        base_folder += '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    subfolders = ['model_weights']
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)

    PATH = args.previous_checkpoint

    logfile = base_folder + 'log.txt'
    if not PATH:
        with open(logfile, 'w') as f:
            f.write(str(args) + "\n")
            f.write('Epoch\tTrain\tValidation\n')

    data_path = args.path_for_training_data
    params = {
        "LIST"    : f"{data_path}/list.csv", 
        "VAL"     : f"{data_path}/valid_clusters.txt",
        "TEST"    : f"{data_path}/test_clusters.txt",
        "DIR"     : f"{data_path}",
        "DATCUT"  : "2030-Jan-01",
        "RESCUT"  : args.rescut, #resolution cutoff for PDBs
        "HOMO"    : 0.70 #min seq.id. to detect homo chains
    }


    LOAD_PARAM = {'batch_size': 1,
                  'shuffle': True,
                  'pin_memory':False,
                  'num_workers': 4}

   
    if args.debug:
        args.num_examples_per_epoch = 50
        args.max_protein_length = 1000
        args.batch_size = 1000

    #train, valid, test = build_training_clusters(params, args.debug)

    # train_set = PDB_dataset(list(train.keys()), loader_pdb, train, params)
    # train_loader = torch.utils.data.DataLoader(train_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    # valid_set = PDB_dataset(list(valid.keys()), loader_pdb, valid, params)
    # valid_loader = torch.utils.data.DataLoader(valid_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    # test_set = PDB_dataset(list(test.keys()), loader_pdb, test, params)
    # test_loader =torch.utils.data.DataLoader(test_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)

    # own data
    train_data_pn = pd.read_csv(args.train_csv_path)
    val_data_pn = pd.read_csv(args.val_csv_path)

    model = ProteinMPNN(node_features=args.hidden_dim, 
                        edge_features=args.hidden_dim, 
                        hidden_dim=args.hidden_dim, 
                        num_encoder_layers=args.num_encoder_layers, 
                        num_decoder_layers=args.num_encoder_layers, 
                        k_neighbors=args.num_neighbors, 
                        dropout=args.dropout, 
                        augment_eps=args.backbone_noise)
    model.to(device)

    if PATH:
        checkpoint = torch.load(PATH)
        total_step = checkpoint['step'] #write total_step from the checkpoint
        epoch = checkpoint['epoch'] #write epoch from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded previous pretrained weights.")
    else:
        total_step = 0
        epoch = 0

    if args.model_noise == 0.02:
        vanilla_path = "/scratch/prj0000000224/yinghua_proj/ProteinMPNN/vanilla_model_weights/v_48_002.pt"
    elif args.model_noise == 0.2:
        vanilla_path = "/scratch/prj0000000224/yinghua_proj/ProteinMPNN/vanilla_model_weights/v_48_020.pt"
    else:
        raise NotImplementedError

    checkpoint = torch.load(vanilla_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"✅ Loaded vanilla ProteinMPNN pretrained weights: {vanilla_path}.")
    #optimizer = get_std_opt(model.parameters(), args.hidden_dim, total_step)
    base_lr = 5e-6
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)

    ref_model = copy.deepcopy(model)
    # If model was wrapped with DataParallel, get underlying module
    try:
        ref_model = ref_model.to(device)
    except Exception:
        # ensure module is on device (for DataParallel)
        if hasattr(ref_model, 'module'):
            ref_model.module.to(device)
        else:
            ref_model.to(device)

    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()


    if PATH:
        optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    with ProcessPoolExecutor(max_workers=3) as executor:
        # q = queue.Queue(maxsize=3)
        # p = queue.Queue(maxsize=3)
        # t = queue.Queue(maxsize=3)
        # for i in range(3):
        #     q.put_nowait(executor.submit(get_pdbs_new, train_data_pn, args.train_pdb_path, args.max_protein_length, args.num_examples_per_epoch))
        #     p.put_nowait(executor.submit(get_pdbs_new, val_data_pn, args.val_pdb_path, args.max_protein_length, args.num_examples_per_epoch))
        #     t.put_nowait(executor.submit(get_pdbs, test_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
        # pdb_dict_train = q.get().result()
        # pdb_dict_valid = p.get().result()
        # pdb_dict_test_original = t.get().result()  # use proteinmpnn original test 
        fut_train = executor.submit(get_pdbs_new, train_data_pn, args.train_pdb_path,
                                    args.max_protein_length, args.num_examples_per_epoch)
        fut_valid = executor.submit(get_pdbs_new, val_data_pn, args.val_pdb_path,
                                    args.max_protein_length, args.num_examples_per_epoch)
        # fut_test  = executor.submit(get_pdbs, test_loader, 1,
        #                             args.max_protein_length, args.num_examples_per_epoch)

        pdb_dict_train = fut_train.result()
        pdb_dict_valid = fut_valid.result()
        # pdb_dict_test_original = fut_test.result()
       
        dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length) 
        dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)
        # dataset_test = StructureDataset(pdb_dict_test_original, truncate=None, max_length=args.max_protein_length)
        
        loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
        loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
        # loader_test = StructureLoader(dataset_test, batch_size=10000)  #args.batch_size)
        
        model.eval()
        with torch.no_grad():
            validation_sum, validation_weights = 0., 0.
            validation_acc = 0.
            validation_critical_acc, validation_critical_weights = 0., 0.
            for _, batch in enumerate(loader_valid):
                X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                logp_pos = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                mask_for_loss = mask*chain_M
                loss, loss_av, true_false = loss_nll(S, logp_pos, mask_for_loss)
                batch_seq_neg = []
                for bb in batch:
                    batch_seq_neg.append(random.choice(bb['neg_seqs']))
                S_neg = featurize_S_only(batch_seq_neg, device)
                
                validation_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                validation_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                validation_weights += torch.sum(mask_for_loss).cpu().data.numpy()

                diff_mask = (S != S_neg).float()
                validation_critical_acc += torch.sum(true_false*mask_for_loss*diff_mask).cpu().data.numpy()
                validation_critical_weights += torch.sum(mask_for_loss*diff_mask).cpu().data.numpy()
        
        validation_loss = validation_sum / validation_weights
        validation_accuracy = validation_acc / validation_weights
        validation_perplexity = np.exp(validation_loss)
        validation_critical_accuracy = validation_critical_acc / validation_critical_weights
        
        validation_perplexity_ = np.format_float_positional(np.float32(validation_perplexity), unique=False, precision=3)
        validation_accuracy_ = np.format_float_positional(np.float32(validation_accuracy), unique=False, precision=3)
        validation_critical_accuracy_ = np.format_float_positional(np.float32(validation_critical_accuracy), unique=False, precision=3)

        with open(logfile, 'a') as f:
            f.write(f'Before finetuning, valid: {validation_perplexity_}, valid_acc: {validation_accuracy_}, valid_critic_acc: {validation_critical_accuracy_}\n')
        print(f'Before finetuning, valid: {validation_perplexity_}, valid_acc: {validation_accuracy_}, valid_critic_acc: {validation_critical_accuracy_}')

        # with torch.no_grad():
        #     test_sum, test_weights = 0., 0.
        #     test_acc = 0.
        #     for _, batch in enumerate(loader_test):
        #         X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
        #         logp_pos = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
        #         mask_for_loss = mask*chain_M
        #         loss, loss_av, true_false = loss_nll(S, logp_pos, mask_for_loss)
                
        #         test_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
        #         test_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
        #         test_weights += torch.sum(mask_for_loss).cpu().data.numpy()

        # test_loss = test_sum / test_weights
        # test_accuracy = test_acc / test_weights
        # test_perplexity = np.exp(test_loss)
        
        # test_perplexity_ = np.format_float_positional(np.float32(test_perplexity), unique=False, precision=3)
        # test_accuracy_ = np.format_float_positional(np.float32(test_accuracy), unique=False, precision=3)
        # with open(logfile, 'a') as f:
        #     f.write(f'Before finetuning, test: {test_perplexity_}, test_acc: {test_accuracy_}\n')
        # print(f'Before finetuning, test: {test_perplexity_}, test_acc: {test_accuracy_}')

        reload_c = 0 
        best_valid_loss = float('inf')
        for e in range(args.num_epochs):
            t0 = time.time()
            e = epoch + e
            model.train()
            train_sum, train_weights = 0., 0.
            train_acc = 0.
            train_critical_acc, train_critical_weights = 0., 0.
            train_total_loss_all = 0.
            # if e % args.reload_data_every_n_epochs == 0:
            #     if reload_c != 0:
            #         pdb_dict_train = q.get().result()
            #         dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length)
            #         loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
            #         pdb_dict_valid = p.get().result()
            #         dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)
            #         loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
            #         pdb_dict_test_original = t.get().result()
            #         dataset_test = StructureDataset(pdb_dict_test_original, truncate=None, max_length=args.max_protein_length)
            #         loader_test = StructureLoader(dataset_test, batch_size=args.batch_size)
            #         q.put_nowait(executor.submit(get_pdbs_new, train_data_pn, args.train_pdb_path, args.max_protein_length, args.num_examples_per_epoch))
            #         p.put_nowait(executor.submit(get_pdbs_new, val_data_pn, args.val_pdb_path, args.max_protein_length, args.num_examples_per_epoch))
            #         t.put_nowait(executor.submit(get_pdbs, test_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
            #     reload_c += 1
            for _, batch in tqdm(enumerate(loader_train)):
                start_batch = time.time()
                X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                batch_seq_neg = []
                for bb in batch:
                    batch_seq_neg.append(random.choice(bb['neg_seqs']))
                S_pos = S
                S_neg = featurize_S_only(batch_seq_neg, device)
                elapsed_featurize = time.time() - start_batch
                optimizer.zero_grad()
                mask_for_loss = mask*chain_M
                
                if args.mixed_precision:
                    with torch.cuda.amp.autocast():
                        logp_pos = model(X, S_pos, mask, chain_M, residue_idx, chain_encoding_all)
                        logp_neg = model(X, S_neg, mask, chain_M, residue_idx, chain_encoding_all)
                        with torch.no_grad():
                            ref_logp_pos = ref_model(X, S_pos, mask, chain_M, residue_idx, chain_encoding_all)
                            ref_logp_neg = ref_model(X, S_neg, mask, chain_M, residue_idx, chain_encoding_all)
                        total_loss = get_total_loss(logp_pos, logp_neg, ref_logp_pos, ref_logp_neg, S_pos, S_neg, \
                            mask_for_loss, args.alpha, args.beta, loss_type=args.loss_type)
           
                    scaler.scale(total_loss).backward()
                     
                    if args.gradient_norm > 0.0:
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logp_pos = model(X, S_pos, mask, chain_M, residue_idx, chain_encoding_all)
                    logp_neg = model(X, S_neg, mask, chain_M, residue_idx, chain_encoding_all)
                    with torch.no_grad():
                        ref_logp_pos = ref_model(X, S_pos, mask, chain_M, residue_idx, chain_encoding_all)
                        ref_logp_neg = ref_model(X, S_neg, mask, chain_M, residue_idx, chain_encoding_all)
                    total_loss = get_total_loss(logp_pos, logp_neg, ref_logp_pos, ref_logp_neg, S_pos, S_neg, \
                            mask_for_loss, args.alpha, args.beta, loss_type=args.loss_type)
                    total_loss.backward()

                    if args.gradient_norm > 0.0:
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                    optimizer.step()
                
                loss, loss_av, true_false = loss_nll(S_pos, logp_pos, mask_for_loss)
            
                train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                train_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                train_weights += torch.sum(mask_for_loss).cpu().data.numpy()
                train_total_loss_all += total_loss.cpu().data.numpy()

                diff_mask = (S_pos != S_neg).float()
                train_critical_acc += torch.sum(true_false*mask_for_loss*diff_mask).cpu().data.numpy()
                train_critical_weights += torch.sum(mask_for_loss*diff_mask).cpu().data.numpy()

                total_step += 1

            model.eval()
            with torch.no_grad():
                validation_sum, validation_weights = 0., 0.
                validation_acc = 0.
                valid_total_loss_all = 0.
                validation_critical_acc, validation_critical_weights = 0., 0.
                for _, batch in enumerate(loader_valid):
                    X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                    logp_pos = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                    mask_for_loss = mask*chain_M
                    loss, loss_av, true_false = loss_nll(S, logp_pos, mask_for_loss)
                    batch_seq_neg = []
                    for bb in batch:
                        batch_seq_neg.append(random.choice(bb['neg_seqs']))
                    S_neg = featurize_S_only(batch_seq_neg, device)
                    S_pos = S
                    logp_neg = model(X, S_neg, mask, chain_M, residue_idx, chain_encoding_all)
                    with torch.no_grad():
                        ref_logp_pos = ref_model(X, S_pos, mask, chain_M, residue_idx, chain_encoding_all)
                        ref_logp_neg = ref_model(X, S_neg, mask, chain_M, residue_idx, chain_encoding_all)
                    val_total_loss = get_total_loss(logp_pos, logp_neg, ref_logp_pos, ref_logp_neg, S_pos, S_neg, \
                        mask_for_loss, args.alpha, args.beta, loss_type=args.loss_type)
                    
                    validation_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                    validation_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                    validation_weights += torch.sum(mask_for_loss).cpu().data.numpy()
                    valid_total_loss_all += val_total_loss.cpu().data.numpy()

                    diff_mask = (S != S_neg).float()
                    validation_critical_acc += torch.sum(true_false*mask_for_loss*diff_mask).cpu().data.numpy()
                    validation_critical_weights += torch.sum(mask_for_loss*diff_mask).cpu().data.numpy()
            
            train_loss = train_sum / train_weights
            train_accuracy = train_acc / train_weights
            train_perplexity = np.exp(train_loss)
            validation_loss = validation_sum / validation_weights
            validation_accuracy = validation_acc / validation_weights
            validation_perplexity = np.exp(validation_loss)
            validation_critical_accuracy = validation_critical_acc / validation_critical_weights
            train_critical_accuracy = train_critical_acc / train_critical_weights
            valid_total_loss = valid_total_loss_all/len(loader_valid)
            train_total_loss = train_total_loss_all/len(loader_train)
            
            train_perplexity_ = np.format_float_positional(np.float32(train_perplexity), unique=False, precision=3)     
            validation_perplexity_ = np.format_float_positional(np.float32(validation_perplexity), unique=False, precision=3)
            train_accuracy_ = np.format_float_positional(np.float32(train_accuracy), unique=False, precision=3)
            validation_accuracy_ = np.format_float_positional(np.float32(validation_accuracy), unique=False, precision=3)
            train_critical_accuracy_ = np.format_float_positional(np.float32(train_critical_accuracy), unique=False, precision=3)
            validation_critical_accuracy_ = np.format_float_positional(np.float32(validation_critical_accuracy), unique=False, precision=3)
            train_loss_ = np.format_float_positional(np.float32(train_loss), unique=False, precision=3) 
            valid_loss_ = np.format_float_positional(np.float32(validation_loss), unique=False, precision=3) 
            train_total_loss_ = np.format_float_positional(np.float32(train_total_loss), unique=False, precision=3) 
            valid_total_loss_ = np.format_float_positional(np.float32(valid_total_loss), unique=False, precision=3) 
    
            t1 = time.time()
            dt = np.format_float_positional(np.float32(t1-t0), unique=False, precision=1) 
            with open(logfile, 'a') as f:
                f.write(f"epoch: {e+1}, step: {total_step} | train ce loss: {train_loss_} | valid ce loss: {valid_loss_}")
                f.write(f"epoch: {e+1}, step: {total_step} | train total loss: {train_total_loss_} | valid total loss: {valid_total_loss_}")
                f.write(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, \
                    train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}, train_critic_acc: {train_critical_accuracy_}, valid_critic_acc: {validation_critical_accuracy_}\n')
            print(f"epoch: {e+1}, step: {total_step} | train ce loss: {train_loss_} | valid ce loss: {valid_loss_}s")
            print(f"epoch: {e+1}, step: {total_step} | train total loss: {train_total_loss_} | valid total loss: {valid_total_loss_}")
            print(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_} \
                train_critic_acc: {train_critical_accuracy_}, valid_critic_acc: {validation_critical_accuracy_}')
            
            checkpoint_filename_last = base_folder+'model_weights/epoch_last.pt'
            torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges' : args.num_neighbors,
                        'noise_level': args.backbone_noise,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, checkpoint_filename_last)

            if best_valid_loss > valid_total_loss:
                best_valid_loss = valid_total_loss
                print('best valid loss: ', best_valid_loss)
                checkpoint_filename_last = base_folder+'model_weights/epoch_best.pt'
                torch.save({
                            'epoch': e+1,
                            'step': total_step,
                            'num_edges' : args.num_neighbors,
                            'noise_level': args.backbone_noise,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, checkpoint_filename_last)

            if (e+1) % args.save_model_every_n_epochs == 0:
                checkpoint_filename = base_folder+'model_weights/epoch{}_step{}.pt'.format(e+1, total_step)
                torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges' : args.num_neighbors,
                        'noise_level': args.backbone_noise, 
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, checkpoint_filename)
                # model.eval()
                # with torch.no_grad():
                #     test_sum, test_weights = 0., 0.
                #     test_acc = 0.
                #     for _, batch in enumerate(loader_test):
                #         X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                #         logp_pos = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                #         mask_for_loss = mask*chain_M
                #         loss, loss_av, true_false = loss_nll(S, logp_pos, mask_for_loss)
                        
                #         test_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                #         test_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                #         test_weights += torch.sum(mask_for_loss).cpu().data.numpy()

                # test_loss = test_sum / test_weights
                # test_accuracy = test_acc / test_weights
                # test_perplexity = np.exp(test_loss)
                
                # test_perplexity_ = np.format_float_positional(np.float32(test_perplexity), unique=False, precision=3)
                # test_accuracy_ = np.format_float_positional(np.float32(test_accuracy), unique=False, precision=3)
                # with open(logfile, 'a') as f:
                #     f.write(f'epoch: {e+1}, step: {total_step}, time: {dt}, test: {test_perplexity_}, test_acc: {test_accuracy_}\n')
                # print(f'epoch: {e+1}, step: {total_step}, time: {dt}, test: {test_perplexity_}, test_acc: {test_accuracy_}')



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--path_for_training_data", type=str, default="/data/datasets/nilufer/CAPEMPNN/CAPE_MPNN/data/input/CAPE-MPNN/pdb_2021aug02", help="path for loading training data") 
    argparser.add_argument("--path_for_outputs", type=str, default="./exp_yyh_test", help="path for logs and model weights")
    argparser.add_argument("--previous_checkpoint", type=str, default="", help="path for previous model weights, e.g. file.pt")
    argparser.add_argument("--num_epochs", type=int, default=200, help="number of epochs to train for")
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=1, help="save model weights every n epochs")
    argparser.add_argument("--reload_data_every_n_epochs", type=int, default=2, help="reload training data every n epochs")
    argparser.add_argument("--num_examples_per_epoch", type=int, default=1000000, help="number of training example to load for one epoch")
    argparser.add_argument("--batch_size", type=int, default=10000, help="number of tokens for one batch")
    argparser.add_argument("--max_protein_length", type=int, default=10000, help="maximum length of the protein complext")
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers") 
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--num_neighbors", type=int, default=48, help="number of neighbors for the sparse graph")   
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout")
    argparser.add_argument("--backbone_noise", type=float, default=0.2, help="amount of noise added to backbone during training")   
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument("--debug", type=str2bool, default=True, help="minimal data loading for debugging")
    argparser.add_argument("--gradient_norm", type=float, default=-1.0, help="clip gradient norm, set to negative to omit clipping")
    argparser.add_argument("--mixed_precision", type=bool, default=True, help="train with mixed precision")
    argparser.add_argument("--train_csv_path", type=str, default="/scratch/prj0000000224/yinghua_proj/ProteinMPNN/data/train_protein_mpnn_final.csv", help="CSV with training positive/negative sequence pairs")
    argparser.add_argument("--val_csv_path", type=str, default="/scratch/prj0000000224/yinghua_proj/ProteinMPNN/data/valid_protein_mpnn_final.csv", help="CSV with validation positive/negative sequence pairs")
    argparser.add_argument("--alpha", type=float, default=0.2, help="loss balance weight")
    argparser.add_argument("--beta", type=float, default=0.2, help="beta for KL divergence")
    argparser.add_argument("--train_pdb_path", type=str, default="/scratch/prj0000000224/yinghua_proj/esm-if/data/train", help="path for loading training data") 
    argparser.add_argument("--val_pdb_path", type=str, default="/scratch/prj0000000224/yinghua_proj/esm-if/data/valid", help="path for loading training data") 
    argparser.add_argument("--loss_type", type=str, default="mask_sppo_ce", help="loss type") 
    argparser.add_argument("--model_noise", type=float, default=0.2, help="amount of noise added to backbone during training")   


    args = argparser.parse_args()    

    print('debug: ', args.debug)
    print('batch size: ', args.batch_size)
    print('loss type: ', args.loss_type)
    if args.loss_type == 'mask_sppo_ce':
        print('alpha: ', args.alpha)
        print('beta: ', args.beta)
    main(args)   


