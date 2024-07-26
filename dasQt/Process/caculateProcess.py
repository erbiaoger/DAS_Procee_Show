import numpy as np
from tqdm import tqdm
from .freqattributes import (spectrum, spectrogram, fk_transform)

class CaculateProcess():
    def __init__(self, MyProgram):
        self.MyProgram = MyProgram
        self.__dict__.update(self.MyProgram.__dict__)

        pass