import csv
import numpy as np
from matplotlib import pyplot as plt
import math
import glob
import pandas as pd
import matplotlib.ticker as ticker
import scipy as sp
from scipy import stats
from scipy.interpolate import splev, splrep
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from math import sqrt
from Gaze_Corrector import Pupil_Data


class Corrector(object):
    def __init__(self):
        #Instantiating the arrays
        self.psize = []
        self.xpos = []
        self.stimOn = []
        self.indices = []
        self.sorted_tuple = []
        self.crct_pupil = []
        self.crct_tuples = []


    def mean_val(self, arry):
        av = np.array(map(lambda x: np.nan if (x=='NaN' or x == None) else x, arry))
        av1 = np.asarray(av, dtype = float)
        return np.nanmean(av1)

    def Trial_data(self, tnum):

        # importing the pupil measurement data for the trial
        psize = []
        for x in range(20):
            name = "/Users/cdlab/Documents/LRB/Mohamed/AllPupilData-desacc/p"+str(tnum)+"b"+str(x)+"_psize.csv"
            with open(name) as x:
                mycsv = csv.reader(x)
                mycsv = list(mycsv)
                col = mycsv[0]
                psize.append(col)
        pupil_size = pd.DataFrame(psize)
        pupil_size = pupil_size.drop([0], axis = 1)
        pdim = pupil_size.values
        pdim = np.asarray(pdim, dtype = float)
        self.psize = pdim

        # Importing the X gaze position data
        xpos = []
        for x in range(20):
            name = "/Users/cdlab/Documents/LRB/Mohamed/AllXData-desacc/p"+str(tnum)+"b"+str(x)+"_xpos.csv"
            with open(name) as x:
                mycsv = csv.reader(x)
                mycsv = list(mycsv)
                col = mycsv[0]
                xpos.append(col)
        x_pos = pd.DataFrame(xpos)
        x_pos = x_pos.drop([0], axis = 1)
        xgaze = x_pos.values
        xgaze = np.asarray(xgaze, dtype = float)
        self.xpos = xgaze

        # retrieving the stimOn data of when the stimuli appeared on the screen
        stim_On = []
        for x in range(20):
            name = "/Users/cdlab/Documents/LRB/Mohamed/AllStimData/p"+str(tnum)+"b"+str(x)+"_stimOn.csv"
            with open(name) as x:
                mycsv = csv.reader(x)
                mycsv = list(mycsv)
                col = mycsv[0]
                stim_On.append(col)
        stim_On = pd.DataFrame(stim_On)
        stim_on = stim_On.values
        stimOn = np.asarray(stim_on, dtype = float)
        self.stimOn = stimOn

        # the link between the arbitrary time of the eyetracker and actual miliseconds
        samples1 = []
        for x in range(20):
            name =  "/Users/cdlab/Documents/LRB/Mohamed/AllSampleData/p"+str(tnum)+"b"+str(x)+"_sample.csv"
            with open(name) as x:
               mycsv = csv.reader(x)
               mycsv = list(mycsv)
               col = mycsv[0]
               samples1.append(col)
        samples1 = pd.DataFrame(samples1)
        sample2 = samples1.values
        sample3 = np.asarray(sample2, dtype=float)
        self.indices = sample3

    def get_intertrial(self):
        stim_ind = []
        for i in range(20):
            run_ind = []
            for cell in self.stimOn[i]:
                a = np.where(self.indices[i] == cell)
                out1 = a[0][0]
                run_ind.append(out1)
            stim_ind.append(run_ind)

        # creating an array of run averages of pupil diameter
        run_pavg = []
        for i in range(20):
            ind = stim_ind[i][1]
            indo = stim_ind[i][0]
            arry = self.psize[i][indo:ind]
            a = self.mean_val(arry)
            #a = self.mean_val(self.psize[i])
            run_pavg.append(a)

        # creating an array of arrays that are each a block of data from an intertrial period.
        pup_nrm = []
        for i in range(20):

            ##lag adjustor
            arry1 = self.psize[i]
            arry = [(a / float(run_pavg[i])) - 1 for a in arry1]
            pup_nrm.append(arry)
        self.psize = pup_nrm


    def Correct(self, tnum):
        self.Trial_data(tnum)
        self.get_intertrial()

        xvp = []
        for r in range(20):
            tup = zip(self.xpos[r], self.psize[r])
            xvp.append(tup)

        crct = Pupil_Data()
        crct.spl_makr(tnum,20)

        corrected_tuples = []
        corrected_pupil = []
        for row in xvp:
            msr_errors = []
            new_pupil = []
            for e in row:
                xpos = e[0]
                err = crct.spl_evaluate(xpos)
                corrected = e[1] - err

                new_pupil.append(corrected)
                msr_errors.append((e[0],corrected))
            corrected_tuples.append(msr_errors)
            corrected_pupil.append(new_pupil)

        self.crct_pupil = corrected_pupil
        self.crct_tuples = corrected_tuples
