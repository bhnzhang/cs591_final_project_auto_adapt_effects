"""
Amplitude modulation effects
"""

# -------------------------
# Imports
# -------------------------

import numpy as np

import math

# helper functions
from f_aux_functions import *

# playing audio
# from IPython.display import Audio, display
# import sounddevice as sd
# import simpleaudio as sa

# plotting
import matplotlib.pyplot as plt


# -------------------------
# function defs
# -------------------------

def Tremolo( X, depth, modrate, rate=SR ):
    """
    Tremolo
    :param X:
    :param depth:
        from 0 to 1
    :param modrate:
    :param rate:
    :return:
    """

    LFO = 1 - make_sin_LFO(len(X), modrate / rate, depth/2.0) - depth/2.0

    # plt.figure()
    # plt.plot(LFO)
    # plt.show()

    return X * LFO