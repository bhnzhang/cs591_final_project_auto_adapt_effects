"""
seeing how close i can get to the guitar sound on sugar for the pill by slowdive
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
    print('s_optimize_sugarforthepill.py')

    import time

    start_time = time.time()

    # read my guitar recording
    filepath = 'D:\Google Drive\Personal projects\py_audio_effects\\2020_06_23_effects\samples\\'
    # filepath = 'G:\My Drive\Personal projects\py_audio_effects\\2020_06_23_effects\samples\\'
    filename = 'E4 pluck.wav'

    myguitar_raw = readWaveFile(filepath + filename)
    # t = (1 / SR) * np.arange(0, len(myguitar_raw))

    # read sugar for the pill
    filename = 'Sugar for the Pill intro Enote.wav'

    sugarforthepill = readWaveFile(filepath + filename)
    # t = (1 / SR) * np.arange(0, len(myguitar_raw))

    # ------- Optimize
    #

    import pprint
    pp = pprint.PrettyPrinter()

    # effects_for_optm = [ ('filter', {'filtertype': 'LPF_1storder', 'fc': (0,True)} ),
    #     ('delay', {'delay_time': (0.5,True), 'feedback_amt': (0.5, True), 'wetdry': (0.5,True)}), ]
    effects_for_optm = [('delay', {'delay_time': (0.5, True), 'feedback_amt': (0.5, True), 'wetdry': (0.5, True)}),
                        ('filter', {'filtertype': ('HPShelf_1storder', False), 'fc': (0.5, True), 'gain': (-10, False)})]


    n_iters = 20

    myguitar_modded, optimized_effects_chain, res = Optimize_Effects(myguitar_raw, sugarforthepill, effects_for_optm, n_iters)
    print("--- %s seconds ---" % (time.time() - start_time))
    #

    print(res)

    print('--- original effect settings---')
    pp.pprint(effects_for_optm)

    print('--- optimized effect settings ---')
    pp.pprint(optimized_effects_chain)



    from datetime import datetime
    now = datetime.now()
    nowstr = now.strftime("%y-%m-%d-%H-%M-%S")

    myguitar_modded = np.array(myguitar_modded * (25000 / np.amax(myguitar_modded))).astype(np.int16)  # rescale sound
    writeWaveFile(filepath + 'sugar for the pill optim' + nowstr + '.wav', myguitar_modded, rate=SR)


    print('----')
    print('Correlation between raw guitar and song:')
    print(compare_signal_similarity(myguitar_raw, sugarforthepill))
    print('----')
    print('Correlation between modded guitar and song:')
    print(compare_signal_similarity(myguitar_modded, sugarforthepill))