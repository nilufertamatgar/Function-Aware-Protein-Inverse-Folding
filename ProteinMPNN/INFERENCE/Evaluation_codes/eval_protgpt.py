#!/usr/bin/env python3
import argparse, glob, os, numpy as np, torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import re

def read_first_sample_fasta(file_path):
    with open(file_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    for i, line in enumerate(lines):
        if line.startswith('>T=') and i + 1 < len(lines):
            return line, lines[i + 1]
    return None

def clean_protein_sequence(sequence):
    """Clean and validate protein sequence"""
    # Remove whitespace and convert to uppercase
    sequence = re.sub(r'\s+', '', sequence.upper())
    
    # Validate amino acid characters (standard 20 amino acids)
    valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
    if not all(aa in valid_aas for aa in sequence):
        invalid = [aa for aa in sequence if aa not in valid_aas]
        print(f"Warning: Invalid amino acids found: {set(invalid)}")   
    return sequence


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--input_dir', default='/scratch/prj0000000224/yinghua_proj/ProteinMPNN/INFERENCE/PDB_FINAL_old/bn0.2/multi/base/seqs', help='Directory with .fa files')
    parser.add_argument('--input_dir', default='/scratch/prj0000000224/yinghua_proj/ProteinMPNN/INFERENCE/PDB_FINAL/mn0.2/multi/mn0.2_bn0.1_lambda0.2_multi/seqs', help='Directory with .fa files')
    parser.add_argument('--model', default='nferruz/ProtGPT2')
    parser.add_argument('--pattern', default='*.fa')
    parser.add_argument('--mode', default='multi')
    args = parser.parse_args()
    
    # Read sequences
    samples = {}
    for f in glob.glob(os.path.join(args.input_dir, args.pattern)):
        if res := read_first_sample_fasta(f):
            samples[os.path.basename(f)] = res
    
    if not samples:
        print(f"No valid FASTA files found in {args.input_dir}")
        return
    
    print(f"Processing {len(samples)} sequences from {args.input_dir}")
    
    # Load model
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model)
    model.eval().to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Calculate
    avg_lls, lls, ppls = [], [], []
    for filename, (header, seq) in tqdm(samples.items()):
        seq = seq.replace('X', '')
        # input_ids = tokenizer(seq, return_tensors='pt')['input_ids'].to(model.device)
        # with torch.no_grad():
        #     loss = model(input_ids, labels=input_ids).loss
        #     lls.append((loss * (input_ids.shape[1]-1)).item())
        #     ppls.append(torch.exp(loss).item())
        try:
            split_seq = seq.split('/')
        except AttributeError:
            split_seq = [seq]
            print(f'\n{filename}')
        log_likelihood = 0
        n_tokens_total = 0
        for s in split_seq:
            if len(s) == 0:
                continue
            s = clean_protein_sequence(s)
            enc = tokenizer(s, return_tensors='pt')
            input_ids = enc['input_ids'].to(model.device)
            attn = enc['attention_mask'].to(model.device)
            with torch.no_grad():
                out = model(input_ids, attention_mask=attn, labels=input_ids)
                loss = out.loss.item()                      # mean NLL over (T-1)
                n_tokens = input_ids.shape[1] - 1
                if not np.isnan(loss):
                    log_likelihood += -(loss * n_tokens)
                    n_tokens_total += n_tokens

        if n_tokens_total > 0:
            avg_ll = log_likelihood/n_tokens_total
            perplexity = np.exp(-avg_ll)
            print(filename, avg_ll, perplexity, log_likelihood, n_tokens_total)
            if np.isnan(log_likelihood):
                import pdb; pdb.set_trace()
            avg_lls.append(avg_ll)
            lls.append(log_likelihood)
            ppls.append(perplexity)
        else:
            print(filename)
            print(split_seq)
    # Results
    print(f"\nResults for {args.input_dir}:")
    print(f"AVG_LL: {np.mean(avg_lls):.2f} ± {np.std(avg_lls):.2f}")
    print(f"AVG_LL range: {np.min(avg_lls):.2f} ± {np.max(avg_lls):.2f}")
    print(f"LL: {np.mean(lls):.2f} ± {np.std(lls):.2f}")
    print(f"PPL: {np.mean(ppls):.2f} ± {np.std(ppls):.2f}")
    print(f"PPL range: {np.min(ppls):.2f} to {np.max(ppls):.2f}")

if __name__ == '__main__':
    main()