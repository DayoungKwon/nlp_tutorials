import hydra

@hydra.main(config_path="config", config_name="config")
def func(config):
    print(type(config))
    print(config)
    

if __name__ == "__main__":
    func()