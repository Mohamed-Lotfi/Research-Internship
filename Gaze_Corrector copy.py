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


class Pupil_Data(object):
    def __init__(self):
        #Instantiating the arrays
        self.psize = []
        self.xpos = []
        self.ypos = []


        self.stimOn = []
        self.outcome = []
        self.indices = []


        self.xblks = []
        self.pblks = []
        self.yblks = []

        self.xall = []
        self.pall = []
        self.yall = []


        self.sorted_tuple = []
        self.xspots_per = []
        self.binmeans_per = []

        self.xerr = []
        self.sem = []
        self.rms = 0
        self.rms1 = 0
        self.rms2 = 0

        self.W = []
        self.N = []

        self.spl = []

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

        # Importing the Y gaze position data
        ypos = []
        for x in range(20):
            name = "/Users/cdlab/Documents/LRB/Mohamed/AllYData-desacc/p"+str(tnum)+"b"+str(x)+"_ypos.csv"
            with open(name) as x:
                mycsv = csv.reader(x)
                mycsv = list(mycsv)
                col = mycsv[0]
                ypos.append(col)
        y_pos = pd.DataFrame(ypos)
        y_pos = y_pos.drop([0], axis = 1)
        ygaze = y_pos.values
        ygaze = np.asarray(ygaze, dtype = float)
        self.ypos = ygaze

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

        # retrieving the output data of when the feedback appeared on the screen
        out_On = []
        for x in range(20):
            name = "/Users/cdlab/Documents/LRB/Mohamed/AllOutcomeData/p"+str(tnum)+"b"+str(x)+"_outcome.csv"
            with open(name) as x:
                mycsv = csv.reader(x)
                mycsv = list(mycsv)
                col = mycsv[0]
                out_On.append(col)
        out_On = pd.DataFrame(out_On)
        outcome2 = out_On.values
        outcome3 = np.asarray(outcome2, dtype = float)
        outcome = np.array(map(lambda x: x + 500, outcome3))
        self.outcome = outcome

        # the link between the arbitrary time of the eyetracker and actual miliseconds
        samples1 = []
        for x in range(20):
            name = "/Users/cdlab/Documents/LRB/Mohamed/AllSampleData/p"+str(tnum)+"b"+str(x)+"_sample.csv"
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
        # converting the output data to milisecond indices
        runout_ind = []
        for i in range(20):
            run_ind = []
            for cell in self.outcome[i]:
                a = np.where(self.indices[i] == cell)
                out1 = a[0][0]
                run_ind.append(out1)
            runout_ind.append(run_ind)
        # converting the stimOn data to milisecond indices
        stim_ind = []
        for i in range(20):
            run_ind = []
            for cell in self.stimOn[i]:
                a = np.where(self.indices[i] == cell)
                out1 = a[0][0]
                run_ind.append(out1)
            stim_ind.append(run_ind)
        # creating an array of tuples with the start,end indices of the intertrial periods
        int_tuples = []
        for i in range(20):
            run_tup = []
            for x in range(0,39):
                tup = (runout_ind[i][x], stim_ind[i][x+1])
                run_tup.append(tup)
            int_tuples.append(run_tup)

        # creating an array of run averages of pupil diameter
        run_pavg = []
        for i in range(20):
            ind1 = stim_ind[i][1]
            ind2 = stim_ind[i][0]




            arry = zip(self.psize[i][ind2:ind1],self.ypos[i][ind2:ind1])
            arry = filter(lambda v: v[1]<1, arry)
            arry = filter(lambda v: v[1]>-1, arry)
            arry = [a[0] for a in arry]


            a = self.mean_val(arry)
            #a = self.mean_val(self.psize[i])
            run_pavg.append(a)

        # creating an array of arrays that are each a block of data from an intertrial period.
        pup_intbl = []
        for i in range(20):
            run_int = []
            run_intbl = []
            for x in range (0,39):
                strt = int_tuples[i][x][0] - 1
                end = int_tuples[i][x][1] - 1

                ##lag adjustor

                arry1 = self.psize[i][strt+220:end+220]
                arry = [(a / float(run_pavg[i])) - 1 for a in arry1]
                run_intbl.append(arry)
            pup_intbl.append(run_intbl)
        self.pblks = pup_intbl

        # same thing as above but for the x positions
        x_intbl = []
        for i in range(20):
            run_intbl = []
            for x in range (0,39):
                strt = int_tuples[i][x][0] - 1
                end = int_tuples[i][x][1] - 1

                ##lag adjustor

                arry = self.xpos[i][strt+220:end+220]
                run_intbl.append(arry)
            x_intbl.append(run_intbl)
        self.xblks = x_intbl

         # same thing as above but for the x positions
        y_intbl = []
        for i in range(20):
            run_intbl = []
            for x in range (0,39):
                strt = int_tuples[i][x][0] - 1
                end = int_tuples[i][x][1] - 1

                ##lag adjustor

                arry = self.ypos[i][strt+220:end+220]
                run_intbl.append(arry)
            y_intbl.append(run_intbl)
        self.yblks = y_intbl

    def run_extract2(self, run_num):
        inc = 1
        avg_pupil = []
        for i in range(0,39):
            block1 = self.pblks[run_num][i]
            num1 = len(block1) - (len(block1)%inc)
            block2 = block1[0:num1]
            num = num1 / inc
            for x in range(num):
                a = block2[ x*inc : (x*inc) + inc ]
                b = self.mean_val(a)
                avg_pupil.append(b)

        avg_dim = np.asarray(avg_pupil, dtype = float)

        x_avg = []
        for i in range(0,39):
            block1 = self.xblks[run_num][i]
            num1 = len(block1) - (len(block1)%inc)
            block2 = block1[0:num1]
            num = num1 / inc
            for x in range(num):
                a = block2[ x*inc : (x*inc) + inc ]
                b = self.mean_val(a)
                x_avg.append(b)
        xs = np.asarray(x_avg, dtype = float)

        y_avg = []
        for i in range(0,39):
            block1 = self.yblks[run_num][i]
            num1 = len(block1) - (len(block1)%inc)
            block2 = block1[0:num1]
            num = num1 / inc
            for x in range(num):
                a = block2[ x*inc : (x*inc) + inc ]
                b = self.mean_val(a)
                y_avg.append(b)
        y = np.asarray(y_avg, dtype = float)



        return xs, avg_dim, y

    def combine_runs(self):
        x_all1 = []
        p_all1 = []
        y_all1  = []
        for i in range(20):
            run_num = i
            x, avg_dim, y = self.run_extract2(run_num)


            x_all1.extend(x)
            p_all1.extend(avg_dim)
            y_all1.extend(y)

        x_all = np.asarray(x_all1, dtype = float)
        p_all = np.asarray(p_all1, dtype = float)
        y_all = np.asarray(y_all1, dtype = float)

        self.xall = x_all
        self.pall = p_all
        self.yall = y_all

        tup = zip(self.xall,self.pall,self.yall)
        sorted_tuple = sorted(tup, key=lambda tup: tup[0])
        sorted_tuple = filter(lambda v: v[0]==v[0], sorted_tuple)
        self.sorted_tuple = sorted(sorted_tuple, key=lambda tup: tup[0])
        self.sorted_tuple = filter(lambda v: v[2]<1, self.sorted_tuple)
        self.sorted_tuple = filter(lambda v: v[2]>-1, self.sorted_tuple)

    def bin_eql(self,arry,binnum):
        binmeans = []
        xerr = []
        sem = []
        bsize = 30/float(binnum)
        xbins = np.arange(-15,16.5, bsize, dtype= float)
        xspots = [(xbins[x]+xbins[x+1])/2 for x in range(0, len(xbins)-1)]
        bins = []
        N = []
        for x in range (binnum):
            beg = xbins[x]
            end = xbins[x+1]
            bin_ = []
            for e in arry:
                if e[0]>= beg and e[0]< end:
                    bin_.append(e)
            binout = [b[1] for b in bin_]

            N.append(len(binout))
            b = filter(lambda v: v==v, binout)
            bsem = np.std(b)
            sem.append(bsem)
            bmean = self.mean_val(binout)
            binmeans.append(bmean)
            bins.append(bin_)
        w1 = np.asarray(sem, dtype = float)
        w2 = np.divide(1, w1)
        infinity =  w2[3]
        w2 = np.array(map(lambda x: 0.05 if x == infinity else x, w2))
        #N = np.asarray(N, dtype = float)
        #N = [x/N.max() for x in N]
        #N = np.power(N, 2)

        #Nsqrt = [sqrt(x) for x in N]
        #Nsqrt = np.asarray(Nsqrt, dtype = float)
        #w3 = w2 * N
        w = np.asarray(w2, dtype = float)


        return xspots, binmeans, sem, w, N



    def get_data(self, tnum, binnum):
        self.Trial_data(tnum)
        self.get_intertrial()
        self.combine_runs()
        self.xspots_per, self.binmeans_per, self.sem, self.W, self.N= self.bin_eql(self.sorted_tuple, binnum)

    def func(self, x, a, b, c, d):
        return a + (b * (x**2)) + (c * (x**3)) + (d * (x**4))


    def spl_makr(self, tnum, binnum):
        self.get_data(tnum, binnum)

        x = np.asarray(self.xspots_per, dtype = float)
        y = np.asarray(self.binmeans_per, dtype = float)

        xyz = zip(x,y,self.W)
        xyz = filter(lambda v: v[1]==v[1],xyz)
        x = [a[0] for a in xyz]
        y = [a[1] for a in xyz]
        w = [a[2] for a in xyz]

        self.spl = splrep(x, y, w = w, k=3)

    def spl_evaluate(self, xpos):

        return splev(xpos, self.spl)
