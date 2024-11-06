# 1. Introduction
This is the official repository for the paper "Beyond Histogram Comparison: Distribution-Aware
Simple-Path Graph Kernels". This repository contains the code for the experiments in the paper.

# 2. File Structure
The repository is structured as follows:
- datasets: Contains the graph datasets used in the experiments. Can be downloaded from [here](https://chrsmrrs.github.io/datasets/docs/datasets/).
- transformers: save the trained BERT model and the log.
- DASP: The implementation of the DASP algorithm.
- DASP-BERT: The implementation of the DASP-BERT algorithm.
- pmd: The implementation of the probability minkowski distance.
- simple_path_tree: The implementation of generating the simple path tree encoding.
- utils: The implementation of the utility functions.

# 3. Requirements
The code is written in Python 3.8.18. The main requirement packages are:

* numpy==1.23.1
* torch==1.11.0
* networkx==2.6.3
* transformers==4.35.2
* datasets=2.15.0
* scikit-learn==0.24.1
* tqdm==4.63.0
* gensim==4.0.1
* joblib==1.2.0
* pandas==1.2.4
* scipy==1.8.0

And all the runing package environment is listed in the `requirements.txt` file. Note that the file is redundant.

# 4. Usage

## 4.1. DASP
To run the DASP algorithm, you can use the following command:
```bash
python DASP.py --dataset MUTAG --K 3 --H 2
```
This command will run the DASP algorithm on the MUTAG dataset with the parameters K=3 and H=2, and other parameters are set to the default values, which can be found in the `DASP.py` file.

## 4.2. DASP-BERT
To run the DASP-BERT algorithm, you can use the following command:
```bash
python DASP_BERT.py --dataset MUTAG --K 3 --H 2
```
This command will run the DASP-BERT algorithm on the MUTAG dataset with the parameters K=3 and H=2, and other parameters are set to the default values, which can be found in the `DASP_BERT.py` file.