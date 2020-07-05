"""
Automatic optimization of effects chain
"""

# -------------------------
# Imports
# -------------------------

from f_distortion import *
from f_delay import *
from f_conv_reverb import *
from f_aux_functions import *
from f_filter import *
from f_amplitude_mod import *

# numpy
import numpy as np

# optimization
from scipy.optimize import minimize, basinhopping, differential_evolution

import math

# circular buffer class
from c_circ_buffer import Circ_buffer

# helper functions
from f_aux_functions import *

# plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from copy import deepcopy

# -------------------------
# function defs
# -------------------------

def Effects_Chain( X, effects_chain ):
    """
    This function will take an input signal and apply the desired effects chain

    IMPORTANT: the parameters with numerical values are normalized to range from 0-1

    for filters, must also specify a 'filtertype' key

    Inputs:
        X
        effects_chain
            type: tuple of tuples
            desc: A tuple (or other iterable) of effects and their arguments
                    Each inner tuple contains the following:
                    ( 'name of effect', { dictionary of arguments } )
                An example:
                    ( 'flanger', {'moddepth': 0.002, 'modrate': 0.5, 'feedback_amt': 0.2},
                      'delay', {'delay_time': 0.5, 'feedback_amt': 0.8, 'wetdry': 0.5} )
                The above will create an effects chain with a flanger that goes into a delay


    :return:
    """

    effect_chain_copy = deepcopy(effects_chain)

    for effect in effect_chain_copy:

        effect_params = effect[1]

        if effect[0] == 'delay':
            # delay
            effect_params['delay_time'] = (5e-2) * np.power(10, 2*effect_params['delay_time'])
            X = Delay(X, **effect_params)

        elif effect[0] == 'chorus':
            # chorus
            effect_params['delay_time'] = (1e-3) * np.power(10, 2*effect_params['delay_time'])
            effect_params['moddepth'] = (1e-4) * np.power(10, 2 * effect_params['moddepth'])
            effect_params['modrate'] =  (2e-1) * np.power(10, 2*effect_params['modrate'])
            X = Chorus(X, **effect_params)

        elif effect[0] == 'flanger':
            # flanger
            effect_params['moddepth'] = (1e-4) * np.power(10, 2 * effect_params['moddepth'])
            effect_params['modrate'] = (2e-1) * np.power(10, 2 * effect_params['modrate'])
            X = Flanger(X, **effect_params)

        elif effect[0] == 'vibrato':
            # vibrato
            effect_params['moddepth'] = (1e-4) * np.power(10, 2 * effect_params['moddepth'])
            effect_params['modrate'] = (1e-1) * np.power(10, 1 * effect_params['modrate'])
            X = Vibrato(X, **effect_params)

        elif effect[0] == 'distortion':
            # distortion
            X = Distortion(X, **effect_params)

        elif effect[0] == 'conv_reverb':
            # convolution reverb
            X = Convolution_Reverb(X, **effect_params)

        elif effect[0] == 'filter':
            # filter
            filtertype = effect_params.pop('filtertype', None)

            # let have cutoff freq range from 0 to 20khz
            # effect_params['fc'] = (2) * np.power(10, 4 * effect_params['fc'])
            # effect_params['fc'] = effect_params['fc']  * 20e3
            effect_params['fc'] = (20) * np.power(10, 3 * effect_params['fc'])

            if filtertype == 'LPF_1storder':
                X = LPF_firstorder( X, **effect_params )
            elif filtertype == 'HPF_1storder':
                X = HPF_firstorder(X, **effect_params)
            elif filtertype == 'LPF_2ndorder':
                X = LPF_secondorder(X, **effect_params)
            elif filtertype == 'HPF_2ndorder':
                X = HPF_secondorder(X, **effect_params)
            elif filtertype == 'BPF_2ndorder':
                X = BPF_secondorder(X, **effect_params)
            elif filtertype == 'BSF_2ndorder':
                X = BSF_secondorder(X, **effect_params)
            elif filtertype == 'LPF_2ndorder_butterworth':
                X = LPF_secondorder_butterworth(X, **effect_params)
            elif filtertype == 'HPF_2ndorder_butterworth':
                X = HPF_secondorder_butterworth(X, **effect_params)
            elif filtertype == 'APF_1storder':
                X = APF_firstorder(X, **effect_params)
            elif filtertype == 'LPShelf_1storder':
                X = LPShelf_firstorder(X, **effect_params)
            elif filtertype == 'HPShelf_1storder':
                X = HPShelf_firstorder(X, **effect_params)
            elif filtertype == 'PEQ':
                X = PEQ_nonconstantQ(X, **effect_params)

        elif effect[0] == 'tremolo':
            # tremolo
            effect_params['modrate'] = (1e-1) * np.power(10, 2 * effect_params['modrate'])
            X = Tremolo(X, **effect_params)

        else:
            print('Error, unsupported effect: ' + effect[0])
            raise

    return X
# end function Effects_Chain()

def compare_signal_similarity(X, Y):
    """
    Compares two signals using cross correlation
    returns the largest value

    :param X:
    :param Y:
    :return:
    """

    # zero pad if necessary
    if len(Y) > len(X):
        X = np.hstack((X, np.zeros(len(Y) - len(X))))
    elif len(X) > len(Y):
        Y = np.hstack((Y, np.zeros(len(X) - len(Y))))

    # first normalize X and Y to unity gain
    MAXVAL = (2**16)/2 - 1
    X = (X/MAXVAL).astype(float)
    Y = (Y/MAXVAL).astype(float)
    X = X/(np.sqrt((np.sum(np.abs(X)**2))))
    Y = Y /(np.sqrt((np.sum(np.abs(Y) ** 2))))

    # cross correlation
    xcorr = np.fft.ifft( np.conj(np.fft.fft(X)) * np.fft.fft(Y) )
    # xcorr = xcorr/np.sqrt( np.sum(np.abs(X)**2) * np.sum(np.abs(Y)**2) ) # second normalization probably unnecessary

    return np.max(np.abs((xcorr)))
# end function compare_signal_similarity()

def Optimize_Effects( X, Y, effects_chain, n_iter = 10 ):
    """

    :param X:
        raw audio
    :param Y:
        effected audio
    :param effects_chain:
         type: tuple of tuples
            desc: A tuple (or other iterable) of effects and their arguments
                    Each inner tuple contains the following:
                    ( 'name of effect', { dictionary of arguments } )
                The dictionary of arguments has key-value pairs like the following:
                'moddepth': (0.002, True)
                The first value is the value to try, the second is whether to optimize for this parameter or not
                An example:
                    (
                    ( 'flanger', {'moddepth': (0.002, False), 'modrate': (0.5, True), 'feedback_amt': (0.2, False} ),
                    ( 'delay', {'delay_time': (0.5, True), 'feedback_amt': (0.8, False), 'wetdry': (0.5, True)} )
                    )
    :return:
    """

    # initial parse of the effects_chain values
    init_effect_vals    = []
    effect_name_params  = []
    static_effect_params = [] # a list of dictionaries of the effects that won't be changed
    for effect in effects_chain:

        # parse the effect
        effect_params   = effect[1]                 # dictionary of effect parameters
        this_effect_nameparams = [ effect[0], ]     # effect[0] = effect name
        this_effect_static_effects = {}

        # add effect values to lists
        for pname in effect_params:
            # loop thru effect dictionary
            if effect_params[pname][1] == True:
                init_effect_vals.append( effect_params[pname][0] )
                this_effect_nameparams.append( pname )
            else:
                this_effect_static_effects[pname] = effect_params[pname][0]

        effect_name_params.append(this_effect_nameparams)
        static_effect_params.append(this_effect_static_effects)

    # end for effect in effects_chain:
    # print('debug')

    def convert_merit_func_input_to_effects_chain( effect_param_values, effect_name_params_list, static_effects_list ):
        """
        helper function for converting from merit function input structure to effects_chain input structure

        :param effect_param_values:
        :param effect_name_params_list:
        :return:
        """
        ii = 0
        effects_chain = []
        # basically need to re-construct the effects_chain structure
        for effect, static_effect_dict in zip(effect_name_params_list,static_effects_list):
            param_dict = {}
            for param_name in effect[1:]:
                param_dict[param_name] = effect_param_values[ii]
                ii += 1
            param_dict.update(static_effect_dict)
            effects_chain.append((effect[0], param_dict))
        # end for effect in effect_name_params_list:
        return effects_chain
    # end function convert_merit_func_input_to_effects_chain

    def merit_function( effect_param_values, effect_name_params_list, static_effects_list ):
        """
        applies effects and returns correlation
        this is the function for the optimizer to minimize

        :param effect_param_values:
            list of just the values of each effect
        :param effect_name_params_list:
            a list of tuples that looks like:
            [ ('effect name1', 'name of param1', 'name of param2', ... ),
              ('effect name2', 'name of param1', 'name of param2', ... ) ]
        static_effects_list:
            a list of equal length to effect_name_params_list
            made of dicts where each dict is made of the static effects
        :return:
        """
        print('executing merit function')
        effects_chain = convert_merit_func_input_to_effects_chain( effect_param_values, effect_name_params_list, static_effects_list )

        # run the input through the effects chain
        X_effected = Effects_Chain(X, effects_chain)

        # return 1 - correlation
        return 1 - compare_signal_similarity(X_effected, Y)
    # end function merit_function()

    # make bounds
    bounds = []
    for ii in range(len(init_effect_vals)):
        bounds.append( (0.0,1.0) )

    # res = minimize(lambda x: merit_function(x,effect_name_params,static_effect_params),
    #                init_effect_vals,
    #                method='Nelder-Mead',
    #                options={'disp': True, 'fatol': 1e-6})
    # res = minimize(lambda x: merit_function(x, effect_name_params, static_effect_params),
    #                init_effect_vals,
    #                method='BFGS',
    #                options={'disp': True})
    # res = basinhopping(lambda x: merit_function(x,effect_name_params,static_effect_params),
    #                    init_effect_vals,
    #                    niter=10,
    #                    T=0.0,
    #                    stepsize=0.25)
    res = differential_evolution(lambda x: merit_function(x,effect_name_params,static_effect_params),
                     bounds=bounds,
                       maxiter=n_iter,
                     disp=True,)

    # generate the modulated song
    optimized_effects_chain = convert_merit_func_input_to_effects_chain(res.x.astype(float),effect_name_params,static_effect_params)
    X_mod = Effects_Chain( X, optimized_effects_chain )
    optimized_effects_chain = convert_merit_func_input_to_effects_chain(res.x.astype(float), effect_name_params,
                                                                        static_effect_params) # annoying but run again to avoid pass by reference stuff changing

    return X_mod, optimized_effects_chain, res
# end function Optimize_Effects()

# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    print('f_optimize_effects.py')

    import time

    start_time = time.time()

    # read a wave file
    # mysong = readWaveFile('G:\My Drive\Personal projects\iphone recordings\zebra.wav')
    # mysong = readWaveFile('G:\My Drive\Personal projects\iphone recordings\\2020 06 08\Clap jmann.wav')
    # mysong = readWaveFile('G:\My Drive\Personal projects\iphone recordings\\2020 06 20\G maj chord.wav')
    # mysong = readWaveFile('G:\My Drive\Personal projects\iphone recordings\\2020 06 21\Desire lines solo crop.wav')
    # mysong = readWaveFile('G:\My Drive\Personal projects\iphone recordings\\2020 06 21\Major leagues riff crop.wav')
    # mysong = readWaveFile('G:\My Drive\Personal projects\iphone recordings\\2020 06 21\Forever chords crop.wav')
    # mysong = readWaveFile('G:\My Drive\Personal projects\iphone recordings\\2020 06 21\White ferrari chords crop.wav')
    # t = (1 / SR) * np.arange(0, len(mysong))

    # filepath = 'G:\My Drive\Personal projects\py_audio_effects\\2020_06_23_effects\samples\\'
    filepath = 'D:\Google Drive\Personal projects\py_audio_effects\\2020_06_23_effects\samples\\'
    filename = 'A string pluck.wav'
    # filename = 'let it live solo.wav'

    mysong = readWaveFile(filepath + filename)
    t = (1 / SR) * np.arange(0, len(mysong))

    # effects = [('delay', {'delay_time': 0.75,
    #                       'feedback_amt': 0.9,
    #                       'wetdry': 0.5,
    #                       'apply_fb_input': False} ), ]

    # effects = [('chorus', {'delay_time': 0.75,
    #                       'moddepth': 0.7,
    #                        'modrate': 0.7,
    #                       'wetdry': 0.5 }), ]

    # effects = [('flanger', {'moddepth': 0.7,
    #                        'modrate': 0.3,
    #                        'feedback_amt': 0.5}), ]

    effects = [('vibrato', {'moddepth': 0.6,
                            'modrate': 0.4}), ]

    effects_for_optm = [('vibrato', {'moddepth': (0.2, True),
                            'modrate': (0.2, True)}), ]
    #
    # effects = [( 'delay', {'delay_time': 0.5,
    #                 'feedback_amt': 0.5,
    #                 'wetdry': 0.5,
    #                 'apply_fb_input': True} ) ]

    # effects = [ ( 'delay', {'delay_time': 0.5,
    #                 'feedback_amt': 0.4,
    #                 'wetdry': 0.3,
    #                 'apply_fb_input': True} ),
    #             ('flanger', {'moddepth': 0.005,
    #                        'modrate': 0.2,
    #                        'feedback_amt': 0.1})
    #             ]

    # effects = [
    #            ('flanger', {'moddepth': 0.3,
    #                         'modrate': 0.7,
    #                         'feedback_amt': 0.5})
    #            ]

    # effects = [
    #     ('conv_reverb', {'rep_time': 1.0,
    #                  'pulse_width': 0.001,
    #                  'decay_time': 1,
    #                   'duration': 5.0,
    #                   'wetdry': 1.0 } ),
    # ]

    shape = 'FEXP1'
    # effects = [
    #     ('distortion', {'amount': 0.4, 'gain':0.5, 'shape': shape}),
    # ]

    # effects = [
    #     ('vibrato', {'moddepth':0.6,
    #                  'modrate': 0.7}),
    #     ('distortion', {'amount': 0.2,
    #                     'gain': 0.2,
    #                     'shape': shape}),
    # ]

    mysong2 = Effects_Chain(mysong, effects)
    print("--- %s seconds ---" % (time.time() - start_time))
    #
    # # save
    mysong2 = np.array(mysong2 * (25000 / np.amax(mysong))).astype(np.int16)  # rescale sound
    writeWaveFile(filepath + 'songmod2.wav', mysong2, rate=SR)

    print('correlation between original and mod''ed song:')
    print(compare_signal_similarity(mysong, mysong2))
    print("--- %s seconds ---" % (time.time() - start_time))

    # ------- Optimize
    #
    mysong3, optimized_effects_chain, res = Optimize_Effects(mysong, mysong2, effects_for_optm)
    print("--- %s seconds ---" % (time.time() - start_time))
    #
    print(res)
    print(optimized_effects_chain)
    #
    mysong3 = np.array(mysong3 * (25000 / np.amax(mysong))).astype(np.int16)  # rescale sound
    writeWaveFile(filepath + 'songoptim.wav', mysong3, rate=SR)

    print('correlation between optimized and mod''ed song:')
    print(compare_signal_similarity(mysong2, mysong3))


    # # Sweep different values of the effect and plot the correlation
    # vals = np.arange(0,1,0.05)
    # corrs = np.zeros(len(vals)) # dimensions corr vs. mod depth vs. mod rate
    # for ii in range(len(vals)):
    #     print( 'loop ' + str(ii) + ' of ' + str(len(vals)) )
    #
    #     effects = [
    #         ('distortion', {'amount': vals[ii], 'gain': 0.5, 'shape': 'FEXP1'}),
    #     ]
    #
    #     corrs[ii] = compare_signal_similarity(mysong2, Effects_Chain(mysong, effects))
    #
    #     print("--- %s seconds ---" % (time.time() - start_time))
    # # end for
    #
    # plt.figure()
    # plt.plot( vals, corrs )
    # plt.xlabel('gain')
    # plt.show()

    # # 2D search --------------
    # vals = np.arange(0,1,0.1)
    # corrs = np.zeros([len(vals), len(vals)])  # dimensions corr vs. mod depth vs. mod rate
    # icount = 1
    # for ii in range(len(vals)):
    #     for jj in range(len(vals)):
    #         print('loop ' + str(icount) + ' of ' + str(len(vals) ** 2))
    #
    #         # effects = [('vibrato', {'moddepth': vals[ii],
    #         #                         'modrate': vals[jj]}), ]
    #
    #         # effects = [('delay', {'delay_time': vals[ii],
    #         #                       'feedback_amt': vals[jj],
    #         #                       'wetdry': 0.5,
    #         #                       'apply_fb_input': True})]
    #
    #         # effects = [
    #         #     ('flanger', {'moddepth': vals[ii],
    #         #                  'modrate': vals[jj],
    #         #                  'feedback_amt': 0.5})
    #         # ]
    #
    #         effects = [
    #             ('vibrato', {'moddepth': vals[ii],
    #                          'modrate': 0.7} ),
    #             ('distortion', {'amount': vals[jj],
    #                             'gain': 0.5,
    #                             'shape': shape }),
    #         ]
    #
    #         corrs[ii, jj] = compare_signal_similarity(mysong2, Effects_Chain(mysong, effects))
    #
    #         print("--- %s seconds ---" % (time.time() - start_time))
    #
    #         icount += 1
    # # end for
    #
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    #
    # # Make data.
    # X, Y = np.meshgrid(vals, vals)
    #
    # # Plot the surface.
    # surf = ax.plot_surface(X, Y, corrs, linewidth=0, antialiased=False, cmap=cm.coolwarm)
    # # surf = ax.scatter(X, Y, corrs, cmap=cm.coolwarm, depthshade=True)
    # # surf = ax.scatter(X.flatten(), Y.flatten(), corrs.flatten(), c=corrs.flatten(), depthshade=True)
    # # scatter(xs, ys, zs=0, zdir='z', s=20, c=cm.coolwarm, depthshade=True)
    #
    # # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    #
    # # ax.set_ylabel('mod depth')
    # # ax.set_xlabel('mod rate')
    #
    # # ax.set_ylabel('delay time')
    # # ax.set_xlabel('feedback amount')
    #
    # # ax.set_xlabel('mod rate')
    # # ax.set_ylabel('mod depth')
    #
    # ax.set_xlabel('distortion gain')
    # ax.set_ylabel('vibrato depth')
    #
    # plt.show()