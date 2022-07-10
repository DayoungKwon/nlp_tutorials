from dataclasses import dataclass

@dataclass 
class Config: 
    def save_config(self):
        raise NotImplementedError