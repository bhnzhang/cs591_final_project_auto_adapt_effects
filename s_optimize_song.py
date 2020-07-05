"""
testing optimization
"""

# -------------------------
# Imports
# -------------------------

from f_distortion import *
from f_delay import *
from f_conv_reverb import *
from f_aux_functions import *
from f_optimize_effects import *

# numpy
import numpy as np

# optimization
from scipy.optimize import minimize, basinhopping

import math

# circular buffer class
from c_circ_buffer import Circ_buffer

# helper functions
from f_aux_functions import *

# plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    print('s_optimize_song.py')

    import time

    start_time = time.time()

    # read my guitar recording
    filepath = 'D:\Google Drive\Personal projects\py_audio_effects\\2020_06_23_effects\samples\\'
    # filepath = 'G:\My Drive\Personal projects\py_audio_effects\\2020_06_23_effects\samples\\'
    filename = 'E2 pluck.wav'

    myguitar_raw = readWaveFile(filepath + filename)
    # t = (1 / SR) * np.arange(0, len(myguitar_raw))

    # read desired sound
    # filename = 'E4 pluck_revtrem.wav'
    # effects_for_optm = [
    #     ('conv_reverb', {'room_size': (20,False), 'cutoff_freq': (10e3,False),
    #                      'decay_time': (1.0,False), 'duration': (2.0,False),
    #                      'SNR': (0.5,False), 'sigma_f': (50,False), 'wetdry': (0.5,False)}),
    #     ('tremolo', {'depth': (0.3,True), 'modrate': (0.3,True)})
    # ]

    # filename = 'E4 pluck_delaychorus.wav'
    # effects_for_optm = [
    #     ('delay', {'delay_time': (0.2,True), 'feedback_amt': (0.6,True), 'wetdry': (0.4,True)}),
    #     ('chorus', {'delay_time': (0.01,True), 'moddepth': (0.3,True), 'modrate':(0.4,True), 'wetdry': (0.3,True)})
    # ]

    # filename = 'E2 pluck_distort_filt.wav'
    # effects_for_optm = [
    #     ('distortion', {'shape': ('FEXP1',False), 'amount': (0.9,True), 'gain': (0.2,True)}),
    #     ('filter', {'filtertype': ('HPShelf_1storder',False), 'fc': (0.5,True), 'gain': (-10,False)})
    # ]

    filename = 'E2 pluck_delaychorusvib.wav'
    effects_for_optm = [
        ('delay', {'delay_time': (0.2,True), 'feedback_amt': (0.9,True), 'wetdry': (0.4,True)}),
        ('chorus', {'delay_time': (0.01,True), 'moddepth': (0.3,True), 'modrate': (0.4,True), 'wetdry': (0.3,True)}),
        ('vibrato', {'moddepth': (0.8,True), 'modrate': (0.01,True)})
    ]

    desiredsound = readWaveFile(filepath + filename)
    # t = (1 / SR) * np.arange(0, len(myguitar_raw))

    # ------- Optimize

    # effects_for_optm = [ ('filter', {'filtertype': 'LPF_1storder', 'fc': (0,True)} ),
    #     ('delay', {'delay_time': (0.5,True), 'feedback_amt': (0.5, True), 'wetdry': (0.5,True)}), ]
    # effects_for_optm = [('delay', {'delay_time': (0.5, True), 'feedback_amt': (0.5, True), 'wetdry': (0.5, True)}), ]

    import pprint
    pp = pprint.PrettyPrinter()

    n_iters = 10
    myguitar_modded, optimized_effects_chain, res = \
        Optimize_Effects(myguitar_raw,
                         desiredsound,
                         effects_for_optm,
                         n_iters)

    print("--- %s seconds ---" % (time.time() - start_time))
    #
    pp.pprint(res)
    pp.pprint(optimized_effects_chain)


    from datetime import datetime
    now = datetime.now()
    nowstr = now.strftime("%y-%m-%d-%H-%M-%S")

    myguitar_modded = np.array(myguitar_modded * (25000 / np.amax(myguitar_modded))).astype(np.int16)  # rescale sound
    writeWaveFile(filepath + 'myguitar_optim' + nowstr + '.wav', myguitar_modded, rate=SR)


    print('----')
    print('Correlation between raw guitar and song:')
    print(compare_signal_similarity(myguitar_raw, desiredsound))
    print('----')
    print('Correlation between modded guitar and song:')
    print(compare_signal_similarity(myguitar_modded, desiredsound))