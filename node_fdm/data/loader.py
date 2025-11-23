from node_fdm.data.dataset import SeqDataset

def get_train_val_data(
    data_df,
    model_cols,
    shift=60,
    seq_len=60,
    custom_fn=(None, None),
    load_parallel=True,
    train_val_num=(5000, 500),
):
    
    train_files = data_df[data_df.split == "train"].filepath.tolist()
    validation_files = data_df[data_df.split == "val"].filepath.tolist()

    train_dataset = SeqDataset(
        train_files[:train_val_num[0]], 
        model_cols,
        seq_len=seq_len, 
        shift=shift,
        custom_fn=custom_fn,
        load_parallel=load_parallel
    )
    val_dataset = SeqDataset(
        validation_files[:train_val_num[1]], 
        model_cols,
        seq_len=seq_len, 
        shift=shift,
        custom_fn=custom_fn,
        load_parallel=load_parallel
    )
    return train_dataset, val_dataset
