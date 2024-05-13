import hydra
import pyat

"""
Directly run the main function of pyat with commandline hydra configuration
"""


@hydra.main(config_path="configs/", config_name="config")
def func(cfg):
    pyat.main(cfg)


func()
