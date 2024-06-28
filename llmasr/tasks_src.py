import random
import numpy as np
import torch
from loguru import logger
from typing import Dict, List, Callable, Any, Optional, Union, Type
import pandas as pd
from transformers import AutoTokenizer
import gin
from pathlib import Path
from llmasr.utils import get_model_config
from ginpipe.core import gin_configure_externals
import copy
from .inference import get_model_for_inference
from tqdm import tqdm
import json
from whisper_normalizer.basic import BasicTextNormalizer
from jiwer import wer, cer

def set_seed(state, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    state.seed = seed
    return state

def load_tokenizer(state, hf_path, key_out='tokenizer'):
    tokenizer = AutoTokenizer.from_pretrained(hf_path)
    state[key_out] = tokenizer
    return state

def get_dataloaders(state: Dict[str, Any], 
                    split_function: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
                    collate_fn: Optional[Callable] = None,
                    dataset_cls: Optional[Type[Any]] = None, 
                    dataloader_cls: Optional[Type[Any]] = None, 
                    dataset_key_in: str = 'dataset_metadata',
                    dataset_key_out: str = 'datasets',
                    partitions_key_out: str = 'partitions',
                    dataloaders_key_out: str = 'dataloaders') -> Dict[str, Any]:
    """
    Constructs dataloaders from the given state and configurations.

    Args:
        state (dict): The dictionary representing the current state.
        split_function (callable, optional): A function to split the dataset. Defaults to None.
        dataset_cls (type, optional): The class to instantiate datasets. Defaults to None.
        dataloader_cls (type, optional): The class to instantiate dataloaders. Defaults to None.
        dataset_key_in (str, optional): The key in the state to get the dataset metadata. Defaults to 'dataset_metadata'.
        dataset_key_out (str, optional): The key to use for storing datasets in the state. Defaults to 'datasets'.
        partitions_key_out (str, optional): The key to use for storing dataset partitions in the state. Defaults to 'partitions'.
        dataloaders_key_out (str, optional): The key to use for storing dataloaders in the state. Defaults to 'dataloaders'.

    Returns:
        dict: The updated state dictionary with datasets, partitions, and dataloaders.

    Raises:
        TypeError: If split_function, dataset_cls, or dataloader_cls is not callable.
    """

    if split_function is not None:
        partitions = split_function(state[dataset_key_in])
    else:
        partitions = {p: state[dataset_key_in].loc[state[dataset_key_in]['partition']==p] for p in state[dataset_key_in]['partition'].unique()}

    datasets = {k: dataset_cls[k](v) for k,v in partitions.items() if k in dataset_cls}
    if collate_fn is not None:
        collate_fn = collate_fn(state['tokenizer'])
        dataloaders = {k: dataloader_cls[k](v, collate_fn=collate_fn) for k,v in datasets.items() if k in dataloader_cls}
    else:
        dataloaders = {k: dataloader_cls[k](v) for k,v in datasets.items() if k in dataloader_cls}

    state[partitions_key_out] = partitions
    state[dataset_key_out] = datasets
    state[dataloaders_key_out] = dataloaders

    return state

def load_dataset(state: Dict[str, Any], 
                 reader_fns: List[Callable[[], pd.DataFrame]], 
                 cache: bool = True, 
                 postprocessors: List[Callable[[pd.DataFrame], pd.DataFrame]] = [], 
                 key_out: str = 'dataset_metadata',
                 rename: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Loads dataset into the state.

    Args:
        state (dict): The dictionary representing the current state.
        reader_fns (list): List of functions that return DataFrames when called.
        cache (bool, optional): Whether to cache the dataset in the state. Defaults to True.
        postprocessors (list, optional): List of functions to apply to the resulting dataframe. Each function should accept and return a DataFrame. Defaults to [].
        key_out (str, optional): The key to use for storing the dataset in the state. Defaults to 'dataset_metadata'.
        rename (list, optional): List of dictionaries specifying renaming rules. Each dictionary should contain keys 'column', 'value', and 'new_value' for renaming. Defaults to None.

    Returns:
        dict: The updated state dictionary with the loaded dataset.

    Raises:
        TypeError: If reader_fns is not a list.
    """
    
    if not (cache and key_out in state):
        if not isinstance(reader_fns, list):
            raise TypeError("reader_fns must be a list of reader functions.")
        elif len(reader_fns) == 0:
            raise Exception("reader_fns is empty. Supply at least one reader function")
        dfs = [fn() for fn in reader_fns]
        df = pd.concat(dfs).reset_index()
        state[key_out] = df
    else:
        logger.info('Caching dataset metadata from state')
    
    postprocessors = [p() for p in postprocessors]
    for f in postprocessors:
        state[key_out] = f(state[key_out])

    if rename is not None:
        for r in rename:
            state[key_out][r['column']] = state[key_out][r['column']].apply(lambda x: r['new_value'] if x == r['value'] else x)
    
    return state

def fit_model(state, model_cls, trainer_cls):
    model = model_cls()
    trainer = trainer_cls()
    trainer.fit(model, state['dataloaders']['train'], state['dataloaders']['dev'])

class Config(dict):
   def __init__(self, *arg, **kw):
      super(Config, self).__init__(*arg, **kw)

def get_model_from_checkpoint(ckpt,import_path='configs/imports',device='cuda:0'):
    gin.enter_interactive_mode()
    config_str = gin.config_str()
    #registers = copy.deepcopy(gin.config._REGISTRY)
    gin.clear_config()
    #registry_clear_keys = [k for k,v in gin.config._REGISTRY.items() if k not in ['gin.macro', 'gin.constant', 'gin.singleton', 'ginpipe.core.execute_pipeline', 'encodecmae.hub.get_model']]
    #for k in registry_clear_keys:
    #    gin.config._REGISTRY.pop(k)
    config_file = Path(Path(ckpt).parent,'config.gin')
    model_config = get_model_config(config_file,
                                    targets=['fit_model.model_cls', 'get_model_for_inference.tokenizer', 'get_dataloaders.collate_fn', 'DictDataset.out_cols', 'DictDataset.processors'],
                                    replacements={'fit_model.model_cls':'get_model_for_inference.model_cls', 
                                                  'DictDataset.processors': 'get_model_for_inference.preprocessors',
                                                  'get_dataloaders.collate_fn': 'get_model_for_inference.collate_fn',
                                                  'DictDataset.out_cols': 'get_model_for_inference.out_cols'},
                                    additions={'get_model_for_inference.tokenizer':'@tasks.load_tokenizer'})

    flag = {'module_list': [import_path]}
    gin_configure_externals(flag)
    gin.parse_config(model_config)
    model, tokenizer = get_model_for_inference()
    model.load_state_dict(torch.load(ckpt,map_location='cpu')['state_dict'])
    gin.clear_config()
    #gin.config._REGISTRY.clear()
    #gin.config._REGISTRY = registers
    gin.parse_config(config_str)
    model.eval()
    model = model.half()
    print(device)
    model.to(device)
    return model, tokenizer

def load_model(state, ckpt_path, device='cuda:0'):
    if isinstance(device, list):
        device = 'cuda:{}'.format(device[0])
    state['model'], state['tokenizer'] = get_model_from_checkpoint(ckpt_path, device=device)
    state['ckpt_path'] = Path(ckpt_path).parent
    return state

def generate(state):
    if 'predictions' not in state:
        df_metadata = state['dataset_metadata']
        df_test = df_metadata.loc[df_metadata['partition']=='test']
        predictions = []
        for idx, row in tqdm(df_test.iterrows()):
            x = {'filename': row['filename'],
                'transcription': ''}
            for p in state['model'].input_processors:
                x = p(x)
            xin = state['model'].collate_fn([x])
            xin = {k: v.to(state['model'].device) if isinstance(v, torch.Tensor) else v for k,v in xin.items()}
            xin = {k: v.to(state['model'].dtype) if v.dtype not in [torch.int64, torch.int32, torch.int16] else v for k,v in xin.items()}
            out = state['model'].generate(xin, tokenizer=state['tokenizer'])
            pred_i = {'prediction': out.replace('<|endoftext|>',''), 'transcription': row['transcription'], 'filename': row['filename'], 'idx': row['idx'], 'dataset': row['dataset']}
            predictions.append(pred_i)
            with open(Path(state['output_dir'], 'predictions.json'),'a') as f:
                f.write(json.dumps(pred_i, indent=4))
                f.write(',\n')

        state['predictions'] = pd.DataFrame(predictions)
    else:
        logger.info('Caching generations')

    return state

def calculate_metrics(state):
    df_predictions = state['predictions']
    ypred = df_predictions['prediction'].values
    ytrue = df_predictions['transcription'].values

    normalizer = BasicTextNormalizer()
    ypred = [normalizer(x) for x in ypred]
    ytrue = [normalizer(x) for x in ytrue]

    state['metrics'] = {'wer': wer(ytrue, ypred), 'cer': cer(ytrue, ypred)}
    with open(Path(state['output_dir'], 'metrics.json'),'w') as f:
        json.dump(state['metrics'],f,indent=4)
    
    return state