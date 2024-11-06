import argparse
import os
import time
from tokenizers.pre_tokenizers import Whitespace

import numpy as np
import igraph as ig
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers import Tokenizer
import torch
from tqdm import tqdm
from pmd import compute_probability_Minkowski_distance
from simple_path_tree import simple_path_tree
from utils import get_node_labels, load_data_ori, custom_grid_search_cv
from transformers import (
    BertConfig,
    BertForMaskedLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BertModel,
    PreTrainedTokenizerFast,
)


def compute_bert_feats(
    graphs,
    maxh,
    depth,
    dataset,
    label_type="label",
    out_batch_size=512,
    epoch=3,
    size=64,
    layer=4,
    head=4,
    random_state=42,
    save_model=False,
    max_length=512,
):

    all_labels = {
        0: get_node_labels(graphs, label_type=label_type),
    }
    # generate all simple paths
    igraphs = [ig.Graph.from_networkx(g) for g in graphs]
    sps = []
    for i, g in enumerate(graphs):
        # paths_graph = list(dfs_paths_with_depth(g, 0, depth))
        # igraph = ig.Graph.from_networkx(g)
        paths_graph = [
            igraphs[i].get_all_simple_paths(vs, cutoff=depth) for vs in igraphs[i].vs
        ]
        sps.append(paths_graph)

    # generate labels, for each deep
    for deep in range(1, maxh):
        labeledtrees = []
        labeledtrees_set = set()

        for igraph, graph in zip(igraphs, graphs):
            # generate simple path tree encoding
            subtrees = simple_path_tree(igraph, graph, deep)

            labeledtrees.append(subtrees)
            labeledtrees_set.update(subtrees)
        labeledtrees_set = sorted(list(labeledtrees_set))

        # extend labels
        all_labels[deep] = {}
        for gid, lt in enumerate(labeledtrees):
            all_labels[deep][gid] = np.array([labeledtrees_set.index(t) for t in lt])

    # compute node embeddings
    all_corpus = []
    all_graph_paths = {}
    for deep in range(maxh):

        graph_label_paths = {}
        for gid, graph_sps in enumerate(sps):
            graph_label_paths[gid] = {}
            all_label_paths = []

            for node, sp in enumerate(graph_sps):
                graph_label_paths[gid][node] = []
                for path in sp:
                    path_str = ",".join([str(all_labels[deep][gid][n]) for n in path])
                    graph_label_paths[gid][node].append(path_str)

                # sort the simple paths from the same node
                graph_label_paths[gid][
                    node
                ].sort()  # by default, sort by lexicographical order
                all_label_paths.append(graph_label_paths[gid][node])
            all_corpus.extend(all_label_paths)
        all_graph_paths[deep] = graph_label_paths

    model_file = f"transformers/{dataset}"
    if not os.path.exists(model_file):
        os.makedirs(model_file)
    model_file = f"{model_file}/bert_K{maxh}_H{depth}_s{size}_l{layer}_h{head}_e{epoch}"

    if not os.path.exists(model_file):

        print("Training BERT model by MLM 15% ...")

        # use BERT to train the model
        from datasets import Dataset
        from transformers import PreTrainedTokenizerFast

        # to dataset
        text_corpus = [" ".join(c) for c in all_corpus]
        transformer_dataset = Dataset.from_dict({"text": text_corpus})
        transformer_dataset = transformer_dataset.train_test_split(
            test_size=0.1, seed=42
        )

        tokenizer_path = f"transformers/tokenizers"
        if not os.path.exists(tokenizer_path):  # mkdir
            os.makedirs(tokenizer_path)
        tokenizer_path += f"/tokenizer_{dataset}_K{maxh}_H{depth}.json"

        if not os.path.exists(tokenizer_path):

            # create tokenizer
            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()  # 使用空格分词

            # use word level trainer
            token_trainer = WordLevelTrainer(
                special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
            )
            tokenizer.train_from_iterator(text_corpus, token_trainer)

            # save tokenizer
            tokenizer.save(tokenizer_path)

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_path,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )

        tokenized_dataset = transformer_dataset.map(
            tokenize_function, batched=True, num_proc=4
        )

        # BERTconfig
        config = BertConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=size,
            num_hidden_layers=layer,
            num_attention_heads=head,
            max_position_embeddings=max_length,
        )

        # MLM training
        model = BertForMaskedLM(config)

        # Training Arguments
        out_path = f"transformers/log/{dataset}/model_log"
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        training_args = TrainingArguments(
            output_dir=out_path,
            per_device_train_batch_size=64,  # batch size
            per_device_eval_batch_size=64,
            num_train_epochs=epoch,  # epochs
            seed=random_state,
            evaluation_strategy="epoch",
            overwrite_output_dir=True,
            logging_strategy="no",
        )

        # DataCollator, mask 15% tokens
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm_probability=0.15
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=data_collator,
        )

        trainer.train()

        # 转为Bert
        model = model.bert

        if save_model:
            model.save_pretrained(model_file)
            tokenizer.save_pretrained(model_file)

    else:
        print(
            f"Loading pre-trained BERT model on {dataset} with K={maxh}, H={depth}, size={size}, layer={layer}, head={head}, epoch={epoch}"
        )

        model = BertModel.from_pretrained(model_file)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_file)

    model.eval()

    model.to("cuda")
    node_embeddings = []
    num_nodes = [len(g.nodes()) for g in graphs]
    for deep in range(maxh):

        dth_corpus = []
        for gid, graph_sps in all_graph_paths[deep].items():

            graph_sentences = []
            for node, sp in graph_sps.items():
                graph_sentences.append(" ".join(sp))
            dth_corpus.extend(graph_sentences)

        inputs = tokenizer(
            dth_corpus,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        # 计算embedding

        num_batches = (
            int(len(dth_corpus) // out_batch_size + 1)
            if len(dth_corpus) % out_batch_size != 0
            else int(len(dth_corpus) // out_batch_size)
        )
        new_embeddings = torch.zeros((0, size))
        with torch.no_grad():
            model.eval()
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            for i in range(num_batches):
                batch_inputs = {
                    k: v[i * out_batch_size : (i + 1) * out_batch_size]
                    for k, v in inputs.items()
                }
                outputs = model(**batch_inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]
                new_embeddings = torch.cat((new_embeddings, embeddings.cpu()), dim=0)

        # 按num_nodes拆分
        node_embeddings.append(
            np.split(new_embeddings.numpy(), np.cumsum(num_nodes)[:-1])
        )

    return node_embeddings


def main(
    dataset,
    K,
    H,
    label_type="label",
    data_path="datasets_ori",
    gridsearch=True,
    crossvalidation=True,
    random_state=42,
    size=64,
    layer=4,
    head=4,
    epoch=3,
    gamma=None,
    save_model=False,
    out_batch_size=512,
    max_length=512,
):
    print(
        f"Running S2P_transformer_GMM on {dataset} with K={K}, H={H}, size={size}, layer={layer}, head={head}, epoch={epoch}, out_batch_size={out_batch_size}"
    )

    graphs = load_data_ori(dataset, data_path)
    print("loading done.")

    start = time.time()

    graph_embeds = compute_bert_feats(
        graphs,
        K,
        H,
        dataset=dataset,
        random_state=random_state,
        label_type=label_type,
        size=size,
        epoch=epoch,
        layer=layer,
        head=head,
        save_model=save_model,
        out_batch_size=out_batch_size,
        max_length=max_length,
    )

    distance_matrix = np.zeros((len(graphs), len(graphs)))
    for i in tqdm(range(K), desc="computing distance matrix"):
        means = []
        vars = []
        for graph_embed in graph_embeds[i]:
            means.append(np.mean(graph_embed, axis=0))
            var_temps = np.var(graph_embed, axis=0)
            for k in range(len(var_temps)):
                if var_temps[k] <= 0.001:
                    var_temps[k] = 0.001
            vars.append(var_temps)

        distance_matrix += compute_probability_Minkowski_distance(means, vars)

    end = time.time()
    print(f"total time: {end - start} s")

    if gridsearch:

        if gamma is not None:
            gammas = gamma
        else:
            gammas = np.logspace(-6, 1, num=8)
        param_grid = [{"C": np.logspace(-3, 3, num=7)}]
    else:
        gammas = [0.001]

    kernel_matrices = []
    kernel_params = []
    # Generate the full list of kernel matrices from which to select
    M = distance_matrix
    for ga in gammas:
        K = np.exp(-ga * M)
        kernel_matrices.append(K)
        kernel_params.append(ga)
    # kernel_path = OUTPUT_DIR + '/' + ds_name
    # sci.savemat("%s/s2p_kernel_%s_maxh_%d_depth_%d.mat"%(kernel_path, ds_name, maxh - 1, depth - 1), mdict={'kernel': kernel_matrices})
    # ---------------------------------
    # Classification
    # ---------------------------------
    # Run hyperparameter search if needed
    print(
        f"Running SVMs, crossvalidation: {crossvalidation}, gridsearch: {gridsearch}."
    )

    y = np.array([g.graph["label"] for g in graphs])

    # Contains accuracy scores for each cross validation step; the
    # means of this list will be used later on.
    accuracy_scores = []
    np.random.seed(random_state)

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    # Hyperparam logging
    best_C = []
    best_gamma = []

    for train_index, test_index in cv.split(kernel_matrices[0], y):
        K_train = [K[train_index][:, train_index] for K in kernel_matrices]
        K_test = [K[test_index][:, train_index] for K in kernel_matrices]
        y_train, y_test = y[train_index], y[test_index]

        # Gridsearch
        if gridsearch:
            gs, best_params = custom_grid_search_cv(
                SVC(kernel="precomputed"),
                param_grid,
                K_train,
                y_train,
                cv=5,
                random_state=random_state,
            )
            # Store best params
            C_ = best_params["params"]["C"]
            gamma_ = kernel_params[best_params["K_idx"]]
            y_pred = gs.predict(K_test[best_params["K_idx"]])
        else:
            gs = SVC(C=100, kernel="precomputed").fit(K_train[0], y_train)
            y_pred = gs.predict(K_test[0])
            gamma_, C_ = gammas[0], 100
        best_C.append(C_)
        best_gamma.append(gamma_)

        accuracy_scores.append(accuracy_score(y_test, y_pred))
        if not crossvalidation:
            break

    # ---------------------------------
    # Printing and logging
    # ---------------------------------
    if crossvalidation:
        print(
            "Mean 10-fold accuracy: {:2.2f} +- {:2.2f} %".format(
                np.mean(accuracy_scores) * 100, np.std(accuracy_scores) * 100
            )
        )
    else:
        print("Final accuracy: {:2.3f} %".format(np.mean(accuracy_scores) * 100))

    return (
        np.mean(accuracy_scores),
        np.std(accuracy_scores),
        end - start,
    )


def arg_parser():
    arg_parser = argparse.ArgumentParser()

    # DASP parameters
    arg_parser.add_argument("--K",
        type=int,
        default=3,
        help="K for simple-path-tree"
    )
    arg_parser.add_argument(
        "--H",
        type=int,
        default=2,
        help="H for simple paths to generate node embeddings",
    )

    # dataset parameters
    arg_parser.add_argument("--data_path", type=str, default="datasets")
    arg_parser.add_argument("--dataset", type=str, default="MUTAG")
    arg_parser.add_argument("--random_state", type=int, default=42)
    arg_parser.add_argument("--gridsearch", type=bool, default=True)
    arg_parser.add_argument("--crossvalidation", type=bool, default=True)
    arg_parser.add_argument("--label_type", type=str, default="label")

    # transformer parameters
    arg_parser.add_argument("--size", type=int, default=64)
    arg_parser.add_argument("--layer", type=int, default=4)
    arg_parser.add_argument("--head", type=int, default=4)
    arg_parser.add_argument("--epoch", type=int, default=3)
    arg_parser.add_argument("--device", type=str, default="1,2,4")
    arg_parser.add_argument("--save_model", type=bool, default=False)
    
    # Other bert parameters such as 
    # batch size: the batch size for training, default is 64
    # out_batch_size: the batch size for output, default is 512
    # max_length: the max length of the model, default is 512
    # Please refer to the default values in the main function.

    return arg_parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main(
        args.dataset,
        args.K,
        args.H,
        label_type=args.label_type,
        data_path=args.data_path,
        gridsearch=args.gridsearch,
        crossvalidation=args.crossvalidation,
        random_state=args.random_state,
        size=args.size,
        layer=args.layer,
        head=args.head,
        epoch=args.epoch,
        save_model=args.save_model,
    )
