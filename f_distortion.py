"""
Distortion effects
"""

# -------------------------
# Imports
# -------------------------

# numpy
import numpy as np

import math

# circular buffer class
from c_circ_buffer import Circ_buffer

# helper functions
from f_aux_functions import *

# plotting
import matplotlib.pyplot as plt

# -------------------------
# function defs
# -------------------------

def Distortion( X, shape, amount, gain=1.0 ):
    """
    Applies distortion

    Recommended that amount range from 0 to 1

    Distortion/waveshaper functions implemented:
        "SIG" - Sigmoid

    :param X:
        type: array, int16 signed
        desc: input signal
    :param shape:
        desc: Name of waveshaper/distortion function to use
    :param amount:
        desc: amount of distortion. can be a vector too
    :param gain:
        desc: Output volume, range from 0 to 1
    :return:
    """

    # first, input signal needs to be re-scaled to range -1, 1
    X = X/((2**16)/2 - 1.0)

    amount = amount*100

    # now apply distortion
    if shape == 'SIG':
        # sigmoid distortion
        Y = 2 / (1 + np.exp(-amount * X)) - 1
    elif shape == 'SIG2':
        # sigmoid 2 distortion, mild?
        xtrafactor = 10
        Y = (np.exp(amount*X/xtrafactor) - 1) * (np.exp(1) + 1) / (np.exp(amount*X/xtrafactor) + 1) / (np.exp(1) - 1)
    elif shape == 'TANH':
        # hyperbolic tangent, good for diode simulation
        Y = np.tanh( amount*X )/np.tanh(amount)
    elif shape == 'ATAN':
        # arctangent
        Y = np.arctan(amount * X) / np.arctan(amount)
    elif shape == 'FEXP1':
        # fuzz exponential 1
        Y = np.sign(X) * (1 - np.exp( - np.abs(amount*X) ) ) / (1 - np.exp(-amount) )
    elif shape == 'TANH_FW':
        # full wave rectifier hyperbolic tangent, octave fuzz?
        Y = np.abs( np.tanh(amount * X) / np.tanh(amount) )
    else:
        print('ERROR: Shape ' + shape + ' not recognized')
        raise

    # return
    return np.array( Y * ((2**16)/2 - 1.0) * gain  ).astype(np.int16)

# end function Distortion()


# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    print('hi')

    # read a wave file
    # mysong = readWaveFile('G:\My Drive\Personal projects\iphone recordings\zebra.wav')
    # mysong = readWaveFile('G:\My Drive\Personal projects\iphone recordings\\2020 06 08\Clap jmann.wav')
    # mysong = readWaveFile('G:\My Drive\Personal projects\iphone recordings\\2020 06 20\G maj chord.wav')
    # mysong = readWaveFile('G:\My Drive\Personal projects\iphone recordings\\2020 06 21\Desire lines solo crop.wav')
    # mysong = readWaveFile('G:\My Drive\Personal projects\iphone recordings\\2020 06 21\Major leagues riff crop.wav')
    # mysong = readWaveFile('G:\My Drive\Personal projects\iphone recordings\\2020 06 21\Forever chords crop.wav')
    mysong = readWaveFile('G:\My Drive\Personal projects\iphone recordings\\2020 06 21\White ferrari chords crop.wav')
    t = (1 / SR) * np.arange(0, len(mysong))

    # distortion
    shape = 'TANH_FW'
    amount = 5
    mysong = Distortion(mysong, shape, amount)

    # save
    mysong = np.array(mysong * (25000 / np.amax(mysong))).astype(np.int16)  # rescale sound
    writeWaveFile('test.wav', mysong, rate=SR)