"""
data_downstream.py
==================
Data loading and processing utilities for training masked language modeling
using tokenized inputs saved in WebDataset .tar.gz format.
This version is tailored for the ConvNet model in dilated_cnn.py.
"""

import os
import glob
from functools import partial
from typing import List, Union

import torch
import webdataset as wds


#############################################
#   Padding and Collation Utilities
#############################################

def pad_to_longest(sequences: List[torch.Tensor], padding_value: int = -100, batch_first: bool = True):
    """
    Pad a list of 1D token tensors to the length of the longest sequence.
    
    Parameters
    ----------
    sequences : list[torch.Tensor]
        A list of 1D tensors containing token IDs.
    padding_value : int, optional
        Value to pad with. For input_ids, use 0 (pad token). For labels, use -100.
    batch_first : bool, optional
        Whether to return the batch dimension first. Default is True.
    
    Returns
    -------
    torch.Tensor
        A 2D tensor of shape [batch_size, max_seq_length].
    """
    return torch.nn.utils.rnn.pad_sequence(sequences, padding_value=padding_value, batch_first=batch_first)

def collate_fn_masked(batch, padding_value_input: int = 0, padding_value_labels: int = -100):
    """
    Collate function for masked LM that pads each batch properly.
    
    Parameters
    ----------
    batch : list of dict
        Each dict contains keys "input_ids" and "labels", each being a 1D tensor.
    padding_value_input : int, optional
        Padding value for the inputs (default 0).
    padding_value_labels : int, optional
        Padding value for the labels (default -100).
    
    Returns
    -------
    dict
        A dictionary with padded "input_ids" and "labels" tensors.
    """
    input_ids = [sample["input_ids"] for sample in batch]
    labels = [sample["labels"] for sample in batch]
    return {
        "input_ids": pad_to_longest(input_ids, padding_value=padding_value_input, batch_first=True),
        "labels": pad_to_longest(labels, padding_value=padding_value_labels, batch_first=True)
    }

def worker_init_fn(worker_id):
    """
    Initialization function for dataloader workers that ensures each worker 
    uses a distinct subset of shards.
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    split_size = len(dataset.data) // worker_info.num_workers
    dataset.data = dataset.data[worker_id * split_size : (worker_id + 1) * split_size]


#############################################
#   Masked LM Processing Functions
#############################################

def apply_mask(
    seq: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    mlm_prob: float = 0.15
):
    """
    Apply the Masked LM transformation to a token sequence.
    
    For each token in the sequence, with probability mlm_prob:
      - 80%: replace with mask_token_id.
      - 10%: replace with a random token.
      - 10%: keep unchanged.
    Tokens not selected for masking get their label set to -100 to ignore them in loss calculation.
    
    Parameters
    ----------
    seq : torch.Tensor
        A 1D tensor of token IDs (shape: [seq_length]).
    mask_token_id : int
        The token ID to be used for masking (e.g. 4).
    vocab_size : int
        Total number of tokens in the vocabulary.
    mlm_prob : float, optional
        Probability that a token is masked (default 0.15).
    
    Returns
    -------
    dict
        A dictionary with keys "input_ids" (masked sequence) and "labels" (targets for loss).
    """
    # Ensure no extra singleton dimensions remain.
    seq = torch.squeeze(seq)
    labels = seq.clone()
    
    # Determine which positions to mask.
    mask = torch.rand(seq.shape) < mlm_prob
    # Set labels of unmasked tokens to -100, so that loss is computed only on masked tokens.
    labels[~mask] = -100

    rand = torch.rand(seq.shape)
    seq_masked = seq.clone()
    
    # 80% of the masked positions: replace with mask token.
    mask80 = mask & (rand < 0.8)
    seq_masked[mask80] = mask_token_id
    # 10% of the masked positions: replace with a random token.
    mask10 = mask & (rand >= 0.8) & (rand < 0.9)
    seq_masked[mask10] = torch.randint(0, vocab_size, (mask10.sum().item(),), dtype=torch.long)
    # 10%: keep original token.
    return {"input_ids": seq_masked, "labels": labels}


#############################################
#   Dataloader Functions
#############################################

def return_dataloader(
    data: Union[str, List[str]],
    vocab_size: int,
    mask_token_id: int,
    mlm_prob: float = 0.15,
    batch_size: int = 2,
    num_workers: int = 0,
    padding_value_input: int = 0,
    padding_value_labels: int = -100,
    shuffle: int = None
):
    """
    Create and return a WebLoader dataloader for masked LM training.
    
    Steps:
      1. Ensure data is a list of tar files.
      2. Create a WebDataset instance (optionally shuffling if requested).
      3. Decode samples (assumes each sample has an "input.npy" field).
      4. Unpack the tuple and convert the numpy array to a torch tensor.
      5. Apply the masked LM transformation via the apply_mask function.
      6. Batch the samples and then pad them using the custom collate_fn_masked.
    
    Parameters
    ----------
    data : Union[str, List[str]]
        A single tar file path or list of tar file paths.
    vocab_size : int
        Total vocabulary size.
    mask_token_id : int
        Token ID used for masking (e.g., 4).
    mlm_prob : float, optional
        Masking probability (default 0.15).
    batch_size : int, optional
        Batch size (default 2).
    num_workers : int, optional
        Number of workers (default 0).
    padding_value_input : int, optional
        Padding value for input_ids (default 0).
    padding_value_labels : int, optional
        Padding value for labels (default -100).
    shuffle : int, optional
        If provided, shuffles data with this buffer size.
    
    Returns
    -------
    WebLoader
        An iterable WebLoader dataloader.
    """
    # If a single path is provided, wrap it in a list.
    if isinstance(data, str):
        data = [data]
    
    dataset = wds.WebDataset(data)
    if shuffle is not None:
        dataset = dataset.shuffle(shuffle)
    
    dataset = dataset.decode()
    # Use only the "input.npy" field from each sample.
    dataset = dataset.to_tuple("input.npy")
    # Each sample comes as a 1-tuple; unpack and convert to a torch tensor.
    dataset = dataset.map(lambda arr: torch.from_numpy(arr[0]).squeeze())
    # Apply the masked LM transformation.
    dataset = dataset.map(lambda seq: apply_mask(seq, mask_token_id, vocab_size, mlm_prob))
    # Batch samples.
    dataset = dataset.batched(batch_size, collation_fn=None)
    # Pad each batch to the longest sequence.
    dataset = dataset.map(partial(
        collate_fn_masked,
        padding_value_input=padding_value_input,
        padding_value_labels=padding_value_labels
    ))
    
    dataloader = wds.WebLoader(dataset, num_workers=num_workers, batch_size=None)
    return dataloader

def get_data(
    data_dir: str,
    train_data: List[str] = None,
    valid_data: List[str] = None,
    test_data: List[str] = None,
    cross_validation: Union[bool, int] = False,
    batch_size: int = 2,
    num_workers: int = 0,
    padding_value_input: int = 0,
    padding_value_labels: int = -100,
    shuffle: int = None,
    vocab_size: int = 7,       # e.g., A, C, G, T, N, MASK, PAD
    mask_token_id: int = 4,      # adjust this to match your model's [MASK] token index
    mlm_prob: float = 0.15,
    **kwargs
):
    """
    Create train/validation/test dataloaders from a directory of WebDataset tar.gz files.
    
    If cross_validation is used, one tar is selected as test, one as validation, and the remainder as training.
    
    Parameters
    ----------
    data_dir : str
        Directory containing tar.gz files.
    train_data, valid_data, test_data : List[str], optional
        Lists of file paths for each split; if None, they are inferred from filename patterns.
    cross_validation : Union[bool, int], optional
        If specified, use cross-validation splitting.
    batch_size : int, optional
        Batch size (default 2).
    num_workers : int, optional
        Number of data loader workers (default 0).
    padding_value_input : int, optional
        Padding value for input_ids (default 0).
    padding_value_labels : int, optional
        Padding value for labels (default -100).
    shuffle : int, optional
        Shuffle buffer size.
    vocab_size : int
        Total vocabulary size.
    mask_token_id : int
        Index of the mask token.
    mlm_prob : float, optional
        Masking probability.
    
    Returns
    -------
    Tuple[WebLoader, WebLoader, WebLoader]
        (train_dataloader, valid_dataloader, test_dataloader)
    """
    if not os.path.exists(data_dir):
        raise SystemExit(f'The data directory {data_dir} does not exist. Exiting script.')
    
    if cross_validation is not False:
        cross_validation = int(cross_validation) - 1
        tars = sorted(glob.glob(f'{data_dir}/*.tar.gz'), key=lambda x: int(x.split('/')[-1].split('.')[0][4:]))
        test_data = tars[cross_validation]
        valid_data = tars[0] if cross_validation == len(tars) - 1 else tars[cross_validation + 1]
        tars.remove(test_data)
        tars.remove(valid_data)
        train_data = tars
    else:
        tars = glob.glob(f'{data_dir}/*.tar.gz')
        train_data = [x for x in tars if os.path.split(x)[-1].startswith('train')]
        valid_data = [x for x in tars if os.path.split(x)[-1].startswith('valid')]
        test_data  = [x for x in tars if os.path.split(x)[-1].startswith('test')]
    
    train_dataloader = return_dataloader(
        train_data, vocab_size=vocab_size, mask_token_id=mask_token_id, mlm_prob=mlm_prob,
        batch_size=batch_size, num_workers=num_workers,
        padding_value_input=padding_value_input, padding_value_labels=padding_value_labels,
        shuffle=shuffle
    ) if train_data else None

    valid_dataloader = return_dataloader(
        valid_data, vocab_size=vocab_size, mask_token_id=mask_token_id, mlm_prob=mlm_prob,
        batch_size=batch_size, num_workers=num_workers,
        padding_value_input=padding_value_input, padding_value_labels=padding_value_labels,
        shuffle=shuffle
    ) if valid_data else None

    test_dataloader = return_dataloader(
        test_data, vocab_size=vocab_size, mask_token_id=mask_token_id, mlm_prob=mlm_prob,
        batch_size=batch_size, num_workers=num_workers,
        padding_value_input=padding_value_input, padding_value_labels=padding_value_labels,
        shuffle=shuffle
    ) if test_data else None

    return train_dataloader, valid_dataloader, test_dataloader
