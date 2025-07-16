# load_data.py

import numpy as np
from datasets import load_dataset

def cargar_datos():
    dataset = load_dataset("cirimus/super-emotion")
    
    train_data = dataset["train"]
    val_data = dataset["validation"]
    test_data = dataset["test"]
    
    train_df = train_data.to_pandas()
    train_df = train_df[['text', 'labels', 'labels_str']]
    
    # Hay mas de una clasificacion para algunos textos, esto lo hacemos para quedarnos con una sola
    train_df['labels_str_unif'] = train_df['labels_str'].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)
    train_df['labels_unif'] = train_df['labels'].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)
    
    test_df = test_data.to_pandas()
    test_df = test_df[['text', 'labels', 'labels_str']]
    test_df['labels_str_unif'] = test_df['labels_str'].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)
    test_df['labels_unif'] = test_df['labels'].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)
    
    val_df = val_data.to_pandas()
    val_df = val_df[['text', 'labels', 'labels_str']]
    val_df['labels_str_unif'] = val_df['labels_str'].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)
    val_df['labels_unif'] = val_df['labels'].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)
    
    return train_df, val_df, test_df
