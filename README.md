# 1. Introduction
This is the official repository for the paper "Beyond Histogram Comparison: Distribution-Aware Simple-Path Graph Kernels". This repository contains the code for reproducing the experiments in the paper. 

# 2. File Structure
The repository is structured as follows:
- datasets/: Contains the graph datasets used in the experiments. All the datasets in paper can be downloaded from [here](https://chrsmrrs.github.io/datasets/docs/datasets/).
- transformers/: save the trained BERT model and the log.
- DASP.py: The implementation of the DASP algorithm.
- DASP_BERT.py: The implementation of the DASP-BERT algorithm.
- pmd.py: The implementation of the probability minkowski distance.
- simple_path_tree.py: The implementation of generating the simple path tree encoding.
- utils.py: The implementation of the utility functions.

# 3. Requirements
The code is written in Python 3.8.18. The main requirement packages are:

* numpy==1.23.5
* torch==1.11.0
* networkx==2.6.3
* igraph==0.10.8
* transformers==4.35.2
* datasets=2.15.0
* tokenizers==0.15.0
* scikit-learn==1.3.0
* tqdm==4.63.0
* gensim==4.0.1
* joblib==1.4.0
* pandas==1.2.4
* scipy==1.8.0
* pot==0.9.2

And all the running package environments are listed in the `environment.yaml` file. You can build the environment by running the following command; note that the packages are redundant.
**We recommend that you install the above packages manually.

```bash
conda env create -f environment.yaml
```

# 4. Usage

## 4.1. DASP
To run the DASP algorithm, you can use the following command:
```bash
python DASP.py --dataset MUTAG --K 3 --H 2 --size 16
```
This command will run the DASP algorithm on the MUTAG dataset with the parameters K=3 and H=2, and set the embedding size for word2vec to 16.  
Other parameters are set to the default values, which can be found in the `DASP.py` file.

## 4.2. DASP-BERT
To run the DASP-BERT algorithm, you can use the following command:
```bash
python DASP_BERT.py --dataset MUTAG --K 3 --H 2 --save_model
```
This command will run the DASP-BERT algorithm on the MUTAG dataset with the parameters K=3 and H=2, and save the trained BERT model.
Other parameters are set to the default values, which can be found in the `DASP_BERT.py` file.