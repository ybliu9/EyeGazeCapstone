#!pip install Pillow # used to resize images
#!pip install psychopy
#!pip install python-pygaze
#!pip install pygame


import os
import time
import pickle
import random
import pathlib
import shutil

%matplotlib inline
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf


import pygaze
print("Running setup for PyGaze version {}".format(pygaze.__version__))

