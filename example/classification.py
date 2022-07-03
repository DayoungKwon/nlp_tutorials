import numpy as np 
import pandas as pd 
import torch 

from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel

import hydra 
from omegaconf import DictConfig, OmegaConf

def load_tokenizer(config: DictConfig):
    tokenizer = None 
    
    if config.type == 'huggingface' : 
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.name)
    else : 
        raise NotImplementedError    
    
    return tokenizer
    
def load_data(config: DictConfig):
    # load data
    print(OmegaConf.to_yaml(config))
    
    from datasets import load_dataset
    dataset = load_dataset("nsmc")
    
    train_df = pd.read_csv(config.train, sep='\t')    
    test_df = pd.read_csv(config.test, sep='\t')

    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    train_df = train_df.sample(frac=0.4, random_state=999)
    test_df = test_df.sample(frac=0.4, random_state=999)

    # tokenize func 
    tokenizer = load_tokenizer(config.tokenizer)
    
    # create dataset
    class NsmcDataset(Dataset):
        def __init__(self, df):
            self.df = df 
            
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, index):
            text = self.df.iloc[index, 1]
            label = self.df.iloc[index, 2]
            return text, label

    nsmc_train_dataset = NsmcDataset(train_df)
    tokenize_fn = lambda x, tk : tk.encode(x, add_special_tokens=True)
    
    # create dataloader
    train_loader = DataLoader(nsmc_train_dataset, batch_size=2, shuffle=True, num_workers=2)
    
    nsmc_eval_dataset = NsmcDataset(test_df)
    test_loader = DataLoader(nsmc_eval_dataset, batch_size=2, shuffle=False, num_workers=2)

    return train_loader, test_loader, tokenizer

@hydra.main(version_base=None, config_path="config", config_name="classification")
def main(config : DictConfig):
    print(OmegaConf.to_yaml(config))
    train_loader, test_loader = load_data(config.data)
    
    
if __name__ == "__main__": 
    main()
    
    
