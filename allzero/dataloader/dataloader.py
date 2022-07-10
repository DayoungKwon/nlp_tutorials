from dataclasses import dataclass
from random import shuffle
from tokenize import Token
from allzero.tokenizers.base import Tokenizer
from allzero.config.config import Config
from typing import Callable

from datasets import DataSet, DataSetDict
import torch 
import os 

@dataclass 
class DataConfig(Config):
    name:str 
    type:str 
    train:str 
    valid:str 
    test:str 
    train_batch_size : int 
    eval_batch_size : int
    batch : dict 
    shuffle : bool

    def __post_init__(self):
        self.batch = {
            'train' : False,
            'eval' : False,
            'test' : False
        }
        if self.train_batch_size > 0 :
            self.batch['train'] = True
        
        if self.eval_batch_size > 0 : 
            self.batch['eval'] = True

    @classmethod
    def from_dict(cls, cfg: dict):
        return cls(
            type=cfg.get('type'),
            name=cfg.get('name'),
            train=cfg.get('train', None),
            valid=cfg.get('valid', None),
            shuffle=cfg.get('shuffle', True),
            train_batch_size=cfg.get('train_batch_size', None),
            eval_batch_size=cfg.get('eval_batch_size', None),
            test=cfg.get('test', None)
        )

class DataLoader:
    """DataSet is based on huggingface DataSet"""
    def __init__(self, cfg:DataConfig, preprocess_fn : Callable, model_type: str = 'tensorflo', **kwargs):
        """Init DataLoader
            Args :
                preprocess_fn
                model_type : str = None, 'numpy', 'torch', 'tensorflow'

        """

        # init config
        self.cfg = cfg
        self.dataset:DataSetDict = None
        self.preprocess_fn : Callable = preprocess_fn
        
        assert model_type in [None, 'numpy', 'torch', 'tensorflow'], f'Supported model_type = [None, tensorflow, torch, numpy]. (in : {model_type})'
        self.model_type = model_type
        self.loader_columns = None
        self.columns = None
        self.label_columns = None

        self.__setup()
        self.__prepare()
    

    def __setup(self):
        # load dataset 
        # load tokenizer ..?? 
        pass 

    def __prepare(self):
        # preprocess
        # tokenizing, one-hot encoding .. etc 
        if not self.preprocess_fn or not callable(self.preprocess_fn):
            raise ValueError('Please set preprocess function')
        
        # TODO: map? dependency with tokenizer..? 
        for split in self.dataset.keys() :
            self.dataset[split] = self.dataset[split].map(
                batched=self.cfg.batch[split]
            )
            # set format based model_type (torch, tf, ndarray)
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type=self.model_type, columns=self.columns)

    def train_dataloader(self):
        if self.model_type == 'torch':
            return torch.utils.data.DataLoader(
                self.dataset['train'],
                batch_size=self.cfg.train_batch_size,
                shuffle=self.cfg.shuffle
            )

        if self.model_type == 'tensorflow' : 
            #TODO : columns
            self.columns = ['input_ids', 'token_type_ids', 'attention_mask'],
            self.label_columns = ['labels']

            return self.dataset['train'].to_tf_dataset(
                columns=self.columns,
                label_cols=self.label_columns,
                batch_size=self.cfg.train_batch_size,
                shuffle=self.cfg.shuffle
            )
        
        return self.dataset['train']

    def val_dataloader(self):
        if self.model_type == 'torch':
            return torch.utils.data.DataLoader(
                self.dataset['validation'],
                batch_size=self.cfg.train_batch_size,
                shuffle=self.cfg.shuffle
            )

        if self.model_type == 'tensorflow' : 
            #TODO : columns
            self.columns = ['input_ids', 'token_type_ids', 'attention_mask'],
            self.label_columns = ['labels']

            return self.dataset['validation'].to_tf_dataset(
                columns=self.columns,
                label_cols=self.label_columns,
                batch_size=self.cfg.train_batch_size,
                shuffle=self.cfg.shuffle
            )
        
        return self.dataset['validation']

    def test_dataloader(self):
        if self.model_type == 'torch':
            return torch.utils.data.DataLoader(
                self.dataset['test'],
                batch_size=self.cfg.eval_batch_size,
                shuffle=self.cfg.shuffle
            )

        if self.model_type == 'test' : 
            #TODO : columns
            self.columns = ['input_ids', 'token_type_ids', 'attention_mask'],
            self.label_columns = ['labels']

            return self.dataset['test'].to_tf_dataset(
                columns=self.columns,
                label_cols=self.label_columns,
                batch_size=self.cfg.eval_batch_size,
                shuffle=self.cfg.shuffle
            )
        
        return self.dataset['test']
