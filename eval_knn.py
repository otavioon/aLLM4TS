#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path
import numpy as np
from collections import defaultdict
import pandas as pd

from data_provider.data_loader import DAGHAR


# In[ ]:


root_data_path = Path(
    "/workspaces/HIAAC-KR-Dev-Container/some_datasets/DAGHAR/standardized_view"
)

datasets = [
    "KuHar",
    "MotionSense",
    "RealWorld_thigh",
    "RealWorld_waist",
    "UCI",
    "WISDM",
]


# In[ ]:


seed = 2021
train_datasets = defaultdict(dict)
test_datasets = dict()
training_point = "after"

if training_point not in ["before", "after"]:
    raise ValueError("training_point must be 'before' or 'after'")



for dataset in datasets:
    for percent in [1, 10, 50, 100]:
        print(f"Dataset: {dataset}, Percent: {percent}")
        train_datasets[dataset][percent] = DAGHAR(
            root_path=root_data_path / dataset,
            flag="train",
            perform_instance_norm=True,
            percent=percent,
            seed=seed
        )
        
    print(f"Dataset: {dataset}, Test")
    test_datasets[dataset] = DAGHAR(
        root_path=root_data_path / dataset,
        flag="test",
        perform_instance_norm=True,
        percent=100,
        seed=seed
    )
    
    print()


# In[ ]:


def get_dataset(dataset):
    Xs, ys = [], []
    for i in range(len(dataset)):
        X, y = dataset[i]
        Xs.append(X)
        ys.append(y)
    Xs = np.array(Xs)
    ys = np.array(ys).astype(np.int64)
    return Xs, ys


def parse_path_name(root_path: Path):
    path = root_path.name    
    return {
        "name": path,
        "path": str(root_path),
        "dataset_name": path.split("_patch")[0].strip(),
        "patch_size": path.split("_patch-")[1].split("_")[0].strip(),
        "stride": path.split("_stride-")[1].split("_")[0].strip(),
        "pretrain_dataset": path.split("_aLLM4TS-")[1].split("_")[0].strip(),
        "normalization": path.split("_norm-")[1].split("_")[0].strip(),
        "percent": int(path.split("_percent-")[1].split("_")[0].strip()),
        "ft_strategy": path.split("_freeze-")[1].split("_")[0].strip(),
        "head": path.split("_head-")[1].split("_")[0] if "_head-" in path else "aLLM4TS",
    }  


# In[ ]:


ckpts_path = Path("/workspaces/HIAAC-KR-Dev-Container/workspace/aLLM4TS/checkpoints/classification")

ckpts = [
    parse_path_name(ckpt) for ckpt in ckpts_path.iterdir()
]

df = pd.DataFrame(ckpts)
df


# In[ ]:


df = df[(df["normalization"] == "yes")].reset_index(drop=True)
df.sort_values(by="percent", ascending=False, inplace=True)
df


# In[ ]:


from tqdm import tqdm
from sklearn.metrics import accuracy_score


results = []
results_csv_path = f"allmt4s_results_knn_{training_point}.csv"

for row_index, line in tqdm(df.iterrows(), desc="Processing models", total=len(df)):
    train_dataset = train_datasets[line["dataset_name"]][line["percent"]]
    test_dataset = test_datasets[line["dataset_name"]]
    X_train, y_train = get_dataset(train_dataset)
    X_test, y_test = get_dataset(test_dataset)
    
    y_train = np.expand_dims(y_train, axis=1)
    y_test = np.expand_dims(y_test, axis=1)
    
    name = line["name"]
    root_path = Path(line["path"])
    for train_embedding_path in root_path.rglob(f"train_projections_{training_point}_training.npy"):
        test_embedding_path = train_embedding_path.parent / f"test_projections_{training_point}_training.npy"
        exp_no = int(train_embedding_path.parent.name.split("_exp_")[1])
        # print(f"Dataset: {line['dataset_name']}, Percent: {line['percent']}, Exp: {exp_no}, Embedding: {train_embedding_path}")
        train_embeddings = np.load(train_embedding_path)
        test_embeddings = np.load(test_embedding_path)
        # print(f"Train Embedding Shape: {train_embeddings.shape}")
        # print(f"Test Embedding Shape: {test_embeddings.shape}")
        
        # for n_neighorns in [2, 3, 5]:
        for n_neighorns in [2, 3, 5]:
            result_dict = line.to_dict()
            result_dict["train_embedding"] = str(train_embedding_path)
            result_dict["test_embedding"] = str(train_embedding_path.parent / f"test_projections_{training_point}_training.npy")
            result_dict["n_neighbors"] = n_neighorns
            result_dict["exp_no"] = exp_no
            
            knn = KNeighborsClassifier(n_neighbors=n_neighorns)
            # print(f"Train embedding shape: {train_embeddings.shape}, y_train shape: {y_train.shape}")
            # print(f"Test embedding shape: {test_embeddings.shape}, y_test shape: {y_test.shape}")
            
            knn.fit(train_embeddings, y_train)
            y_pred = knn.predict(test_embeddings)
            score = accuracy_score(y_test, y_pred)
            
            result_dict["accuracy"] = score
            results.append(result_dict)
            # print(f"Accuracy: {score}")
            # print()
            
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_csv_path, index=False)
        # print(f"Results saved to {results_csv_path}")
        # print()            

