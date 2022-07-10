from abc import *
from typing import Union, List
from dataclasses import dataclass

# TODO : move this to config folder
@dataclass
class TokenizerConfig:
    type: str  # tokenizer type to load
    name: str
    num_norm: bool = False
    spacing: bool = False
    padding: bool = False
    add_special_token: bool = False
    max_length: bool = None
    truncation: bool = False
    return_tensors: str = None

    def __post_init__(self):
        # post init like bert-type output..?
        pass

    # TODO: use this only not using hydra
    # REMOVE this after integration hydra config
    @classmethod
    def from_dict(cls, cfg: dict):
        return cls(
            type=cfg.get('type'),
            name=cfg.get('name'),
            num_norm=cfg.get('num_norm', False),
            spacing=cfg.get('spacing', False),
            padding=cfg.get('padding', False),
            add_special_token=cfg.get('add_special_token', False),
            max_length=cfg.get('max_length', None),
            truncation=cfg.get('truncation', False),
            return_tensors=cfg.get('return_tensors', None),
        )


class Tokenizer(metaclass=ABCMeta):

    @abstractmethod
    def load(self, name_or_path: str):
        """Load tokenizer by path or hf name"""
        raise NotImplementedError

    @abstractmethod
    def tokenize(self, texts: Union[str, List]) -> List[str]:
        """ Encode texts to features """
        raise NotImplementedError

    @abstractmethod
    def encode(self, texts: Union[str, List], **kwargs):
        """ Encode texts to features 
            https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.encode

        Args : 
            ** padding (bool): Padding size. Defaults to False
            ** add_special_tokens (bool): Add special tokens (SEP, CLS)  Defaults to False
            ** max_length (int): Max length. Defaults to None
            ** truncation (bool): Truncation.  Defaults to False
            ** return_tensors (str): Return type among [int array(None), torch, tf, numpy]. Defaults to None.  

        Returns : 
            Union[List[int], torch.Tensor, tf.Tensor, np.ndarray

        """
        raise NotImplementedError
