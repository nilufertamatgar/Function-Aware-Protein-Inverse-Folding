# FPA on ProteinMPNN

This project extends the [ProteinMPNN repository](https://github.com/dauparas/ProteinMPNN) with FPA fine-tuning and evaluation workflows.

Please follow the original repository for environment setup and basic usage instructions.

## Dataset

### 1. Training Data

1. Download the constructed preference pairs from [Hugging Face](https://huggingface.co/datasets/yinghuayao/FPA-dataset).

2. Download the PDB training data by following the instructions in the ProteinMPNN repository:  
   https://github.com/dauparas/ProteinMPNN/tree/main/training

3. Place all downloaded training data under the `data/` directory.

### 2. Evaluation Data

1. Download the evaluation data from [Hugging Face](https://huggingface.co/datasets/yinghuayao/FPA-dataset/blob/main/ProteinMPNN_Eval_Inputs.tar.gz).

2. Place the downloaded files under the `INFERENCE/` directory.


## FPA Fine-Tuning
1. Move to the training directory:
	```bash
	cd training
	```

2. Start fine-tuning:
	```bash
	./finetune.sh
	```

## Evaluation
1. Move to the evaluation script directory:
	```bash
	cd inference/script
	```

### Sequence Recovery Rate
1. Run:
	```bash
	./Accuracy_{mode}.sh
	```
	where `mode` is `single` or `multi`.

	Output sequences are written to `PDB_FINAL`.

2. Evaluate sequence recovery:
	```bash
	./Evaluation.sh
	```

	Results are saved under `Results`.

### Perplexity and Pairwise Accuracy
1. Run:
	```bash
	./Accuracy_{mode}.sh
	```

	Output scores are written to `Scores_{mode}`.

2. Run:
	```bash
	./Perplexity_accuracy.sh
	```

	Results are saved under `Results`.
