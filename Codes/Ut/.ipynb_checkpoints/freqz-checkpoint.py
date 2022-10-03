# Copyright (c) 2020-2022 Chenxi SHAN <cxshan@hey.com>
# Freqz Tools
__version__ = "0.0.2"
__date__    = "2022-06-12"

import numpy as np

#=================== freq & z tools ===================#

def freq21cm( ref_freq, default=True ):
    r"""
    *** Get the rest-frame freq of a 21cm emission in MHz ***
    
    :Params ref_freq: the referenced rest-frame freq of the 21cm line in Hz;
    :Params default: default = False uses the ref_freq, else uses the default value 1420405751.7667 Hz;
    :Output freq21cm: the rest-frame freq of a 21cm emission in MHz;
    """
    if default==False:
        freq21cm = ref_freq / 1e6  # [MHz]
    else:
        freq21cm = 1420405751.7667 / 1e6  # [MHz]
    return freq21cm

def fz_list( begin, step, stop ):
    r"""
       *** Generate a list of freqs or z *** # Basic function
       !!!  !!! # Important note
       +++  +++ # Improvements
       :Params begin: start of the list
       :Params step: step of the list
       :Params stop: end of the list
       :Output values: freqs or z list
   """
    values = []
    begin, step, stop = float(begin), float(step), float(stop)
    v = np.arange(start=begin, stop=stop+step/2, step=step)
    values += list(v)
    return values

def z2freq( redshifts, print_=False ):
    r"""
    *** Convert the redshift to freq ***
    :Params redshifts: input redshift;
    :Params print_: print indicator;
    :Output freqs: output freq [a list of float];
    """
    redshifts = np.asarray(redshifts)
    freqs = freq21cm(1) / (redshifts + 1.0)
    if print_:
        print("# redshift  frequency[MHz]")
        for z, f in zip(redshifts, freqs):
            print("%.4f  %.2f" % (z, f))
    return freqs

def freq2z( freqs, print_=False ):
    r"""
       *** Convert the redshift to freq *** # Basic function
       !!!  !!! # Important note
       +++  +++ # Improvements
       :Params freqs: input freqs;
       :Params print_: print indicator;
       :Output redshifts: output redshifts [a list of float];
   """
    freqs = np.asarray(freqs)
    redshifts = freq21cm(1) / freqs - 1.0
    if print_:
        print("# frequency[MHz]  redshift")
        for f, z in zip(freqs, redshifts):
            print("%.2f  %.4f" % (f, z))
    return redshifts

def wlist( list_, filename ):
    r"""
       *** Write a list to file w/ line break *** # Basic function
       !!!  !!! # Important note
       +++  +++ # Improvements
       :Params list_: python list object;
       :Params filename: path & name to the file;
       :Output None:
   """
    with open(filename, mode="w") as outfile:
        for s in list_[:-1]:
            outfile.write("%s\n" % s)
        outfile.write("%s" % s)
    print(filename, 'is saved!')

def genfz( type_, begin, step, stop, print_=False ):
    r"""
       *** Generate a freq & z dict *** # Basic function
       !!!  !!! # Important note
       +++  +++ # Improvements
       :Params type_: 'z' redshift; 'f' frequency;
       :Params begin: start of the list;
       :Params step: step of the list;
       :Params stop: end of the list;
       :Output fz: freq & z dict;
   """
    fz = {}
    inlist = fz_list( begin, step, stop )
    if type_ == 'z':
        fz['z'] = inlist
        fz['f'] = z2freq( inlist )
    elif type_ == 'f':
        fz['f'] = inlist
        fz['z'] = freq2z( inlist )
    else:
        print("Check your list type!")
    if print_:
        for z, f in zip(fz['z'], fz['f']):
            print("z=%06.3f, freq=%06.2f MHz" % (z, f))
    return fz
    