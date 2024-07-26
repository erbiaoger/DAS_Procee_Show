import numpy as np
from tqdm import tqdm
from .cog import ncf_corre_cog

class CaculateCog():
    def __init__(self, MyProgram):
        self.MyProgram = MyProgram
        self.__dict__.update(self.MyProgram.__dict__)

        pass