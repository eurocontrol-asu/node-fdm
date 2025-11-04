from node_fdm.data.dataset import SeqDataset

def get_train_val_data(
    data_df,
    model_cols,
    shift=60,
    seq_len=60,
    custom_fn=(None, None),
):
    
    train_files = data_df[data_df.split == "train"].filepath.tolist()
    validation_files = data_df[data_df.split == "val"].filepath.tolist()

    train_dataset = SeqDataset(
        train_files, 
        model_cols,
        seq_len=seq_len, 
        shift=shift,
        custom_fn=custom_fn
    )
    val_dataset = SeqDataset(
        validation_files, 
        model_cols,
        seq_len=seq_len, 
        shift=shift,
        custom_fn=custom_fn
    )
    return train_dataset, val_dataset
