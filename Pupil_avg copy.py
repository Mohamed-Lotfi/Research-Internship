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

class Pupil_Data(object):
    def __init__(self):
        #Instantiating the arrays
        self.psize = []
        self.xpos = []
        self.stimOn = []
        self.outcome = []
        self.indices = []
        self.xblks = []
        self.pblks = []
        self.xall = []
        self.pall = []
        self.sorted_tuple = []
        self.xspots_per = []
        self.binmeans_per = []
        self.xerr = []
        self.sem = []
        self.sem_inc = []
        self.pvtime_inc = []
        self.pvtime_c = []
        self.tarrys = []
        self.corrects = []
        self.incorrects = []

    def mean_val(self, arry):
        av = np.array(map(lambda x: np.nan if (x=='NaN' or x == None) else x, arry))
        av1 = np.asarray(av, dtype = float)
        return np.nanmean(av1)

    def Trial_data(self, tnum):
       # importing the pupil measurement data for the trial
        pruns = []
        name = "/Users/cdlab_admin/Documents/LRB/Mohamed/pupil/Corrected Pupil sizes/p"+str(tnum)+"_corrected.csv"
        with open(name) as x:
            mycsv = csv.reader(x)
            mycsv = list(mycsv)
            m = pd.DataFrame(mycsv)
            n =m[1:]
            pdim = n.drop(0,axis =1)
            pdim = pdim.values
            p = []
            for m in pdim:
                m3 = np.array(map(lambda x: np.nan if x=='' else x, m))
                p.append(m3)
            pdim = np.asarray(p, dtype = float)
            ##outlier handling
            oned_pdim = []
            for x in pdim:
                oned_pdim.extend(x)
            oned_pdim = filter(lambda v: v==v, oned_pdim)
            oned_pdim = sorted(oned_pdim)
            third = np.percentile(oned_pdim,75)
            first = np.percentile(oned_pdim,25)
            iqr = third - first
            extent = iqr * 1.5
            uplim = third + extent
            lowlim = first - extent

            new_pdim = []
            for r in pdim:
                new_r  = np.array(map(lambda x: np.nan if (x > uplim  or x < lowlim) else x, r))
                new_pdim.append(new_r)
            pdim = new_pdim

        self.psize = pdim


        # Importing the X gaze position data
        xpos = []
        for x in range(20):
            name = "/Users/cdlab_admin/Documents/LRB/Mohamed/AllXData-desacc/p"+str(tnum)+"b"+str(x)+"_xpos.csv"
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
            name = "/Users/cdlab_admin/Documents/LRB/Mohamed/AllStimData/p"+str(tnum)+"b"+str(x)+"_stimOn.csv"
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
            name = "/Users/cdlab_admin/Documents/LRB/Mohamed/AllOutcomeData/p"+str(tnum)+"b"+str(x)+"_outcome.csv"
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
            name = "/Users/cdlab_admin/Documents/LRB/Mohamed/AllSampleData/p"+str(tnum)+"b"+str(x)+"_sample.csv"
            with open(name) as x:
               mycsv = csv.reader(x)
               mycsv = list(mycsv)
               col = mycsv[0]
               samples1.append(col)
        samples1 = pd.DataFrame(samples1)
        sample2 = samples1.values
        sample3 = np.asarray(sample2, dtype=float)
        self.indices = sample3


        for i in range(20):
            name  = "/Users/cdlab_admin/Documents/LRB/Mohamed/AllTaskData/p"+str(tnum)+"b"+str(i)+"_task.csv"
            with open(name) as x:
                mycsv = csv.reader(x)
                mycsv = list(mycsv)
                mycsv = np.array(mycsv).T
                corr = mycsv[8]
                corr = corr[1:]
                corr = np.asarray(corr, dtype = int)
                correct = np.where(corr == 1)
                correct = correct[0]
                incorrect = np.where(corr == 0)
                incorrect = incorrect[0]
            self.corrects.append(correct)
            self.incorrects.append(incorrect)

    def get_intertrial(self):
        # converting the output data to milisecond indices
        runout_ind = []
        for i in range(0,20):
            run_ind = []
            for cell in self.outcome[i]:
                a = np.where(self.indices[i] == cell)
                out1 = a[0][0]
                run_ind.append(out1)
            runout_ind.append(run_ind)
        # converting the stimOn data to milisecond indices
        stim_ind = []
        for i in range(0,20):
            run_ind = []
            for cell in self.stimOn[i]:
                a = np.where(self.indices[i] == cell)
                out1 = a[0][0]
                run_ind.append(out1)
            stim_ind.append(run_ind)
        # creating an array of tuples with the start,end indices of the intertrial periods
        int_tuples = []
        for i in range(0,20):
            run_tup = []
            for x in range(0,39):
                tup = (runout_ind[i][x], stim_ind[i][x+1])
                run_tup.append(tup)
            int_tuples.append(run_tup)

        #creating the 1000ms intervals
        ext_int_tuples = []
        for i in range(0,20):
            run_tup = []
            for x in range(0,39):
                tup = (runout_ind[i][x]-500, runout_ind[i][x]+500)
                run_tup.append(tup)
            ext_int_tuples.append(run_tup)

        # creating an array of run averages of pupil diameter
        run_pavg = []
        for i in range(0,20):
            a = self.mean_val(self.psize[i])
            run_pavg.append(a)

        # creating an array of arrays that are each a block of data from an intertrial period.
        pup_intbl = []
        for i in range(0,20):
            run_int = []
            run_intbl = []
            for x in range (0,39):
                strt = ext_int_tuples[i][x][0] - 1
                end = ext_int_tuples[i][x][1] - 1
                arry1 = self.psize[i][strt:end]
                run_intbl.append(arry1)
            pup_intbl.append(run_intbl)
        self.pblks = pup_intbl


        # same thing as above but for the x positions
        x_intbl = []
        for i in range(0,20):
            run_intbl = []
            for x in range (0,39):
                strt = int_tuples[i][x][0] - 1
                end = int_tuples[i][x][1] - 1
                arry = self.xpos[i][strt:end]
                run_intbl.append(arry)
            x_intbl.append(run_intbl)
        self.xblks = x_intbl



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
        x = np.asarray(x_avg, dtype = float)

        return x, avg_dim


    def ptime_agg_c(self):
        avgs = []
        asem = []
        arrys = []
        for a in range(1000):
            arry = []
            for r in range(20):
                correct = self.corrects[r]
                for c in range (len(correct)):
                    blk = correct[c]
                    if blk != 39:
                        num = self.pblks[r][blk][a] if len(self.pblks[r][blk]) > a else np.nan
                        arry.append(num)
            avgs.append(self.mean_val(arry))
            farry = filter(lambda v: v==v, arry)
            asem.append(stats.sem(farry))
            arrys.append(arry)
        self.pvtime_c= avgs
        self.sem = asem
        self.tarrys = arrys

    def ptime_agg_inc(self):
        avgs = []
        asem = []
        arrys = []
        for a in range(1000):
            arry = []
            for r in range(20):
                incorrect = self.incorrects[r]
                for c in range (len(incorrect)):
                    blk = incorrect[c]
                    if blk != 39:
                        num = self.pblks[r][blk][a] if len(self.pblks[r][blk]) > a else np.nan
                        arry.append(num)
            avgs.append(self.mean_val(arry))
            farry = filter(lambda v: v==v, arry)
            asem.append(stats.sem(farry))
            arrys.append(arry)
        self.pvtime_inc= avgs
        self.sem_inc = asem
        self.tarrys = arrys


    def graph_data(self, tnum, xbar, ybar, fit):

        fig = plt.figure(figsize=(20,10))
        ax1 = fig.add_subplot(111)
        fig.suptitle('Pupil Diameter vs. Time', fontsize= 18)
        ax1.set_ylabel('Pupil Diameter(% Change)')
        ax1.set_xlabel('Time (ms)')

        horiz_line_data = np.array([0 for i in xrange(1000)])
        ax1.plot(np.arange(1000), horiz_line_data, 'k:')
        vert_line_data2 = np.array([220 for i in xrange(1000)])
        ax1.plot(vert_line_data2, np.arange(-0.1,0.1,0.0002), 'k:')

        colors = ['b','r','g','m']


        self.Trial_data(tnum)
        self.get_intertrial()
        self.ptime_agg_c()
        self.ptime_agg_inc()


        X = np.asarray(np.arange(1000), dtype = float)
        Y = np.asarray(self.pvtime_c, dtype = float)
        ax1.scatter(X,Y, c=colors[0], label = 'correct')


        X1 = np.asarray(np.arange(1000), dtype = float)
        Y1 = np.asarray(self.pvtime_inc, dtype = float)
        ax1.scatter(X1,Y1, c=colors[1], label = 'incorrect')

        ax1.axhline(y=max(max(Y),max(Y1))+0.02 , xmin=0, xmax=0.495, c= 'k', linewidth = 15.0, label = 'outcome On period')

        if xbar == 1 & ybar == 1:
            xerr = np.absolute(self.xerr)
            yerr = self.sem
            ax1.errorbar(X, Y, xerr =xerr, yerr=yerr ,fmt='o')
        elif xbar == 1:
            xerr = np.absolute(self.xerr)
            ax1.errorbar(X, Y, xerr =xerr,fmt='o')
        elif ybar == 1:
            yerr = self.sem
            yerr1 = self.sem_inc
            ax1.errorbar(X, Y, yerr =yerr,fmt='o', c = colors[0], alpha = 0.2)
            ax1.errorbar(X1, Y1, yerr =yerr1,fmt='o', c = colors[1], alpha = 0.2)
        if fit == 1:
            z = np.polyfit(X,Y,1)
            p = np.poly1d(z)
            x = np.arange(-600,600)
            w = p(x)
            ax1.plot(x,w,c='r')

        legend = ax1.legend(loc='upper right', shadow=True, fontsize = 'large')

    def graph_this(self):
        #plot of the x gaze position vs time with multiple run subplots
        fig = plt.figure()
        fig, axs = plt.subplots(4,4, figsize=(40, 45), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = 0.2, wspace=.2)
        fig.suptitle('All Correct Vs. Incorrect', fontsize= 25)
        axs = axs.ravel()

        horiz_line_data = np.array([0 for i in xrange(1000)])
        vert_line_data2 = np.array([220 for i in xrange(1000)])

        colors = ['b','r']


        for p in range(1,17):

            axs[p-1].set_title("Participant "+str(p))
            axs[p-1].set_ylabel('Pupil Diameter')
            axs[p-1].set_xlabel('Time')
            axs[p-1].plot(np.arange(1000), horiz_line_data, 'k:')
            axs[p-1].plot(vert_line_data2, np.arange(-0.1,0.1,0.0002), 'k:')

            self.Trial_data(p)
            self.get_intertrial()
            self.ptime_agg_c()
            self.ptime_agg_inc()

            X = np.asarray(np.arange(1000), dtype = float)
            Y = np.asarray(self.pvtime_c, dtype = float)
            axs[p-1].scatter(X,Y, c=colors[0], label = 'correct')


            X1 = np.asarray(np.arange(1000), dtype = float)
            Y1 = np.asarray(self.pvtime_inc, dtype = float)
            axs[p-1].scatter(X1,Y1, c=colors[1], label = 'incorrect')

            axs[p-1].axhline(y=max(max(Y),max(Y1))+0.02 , xmin=0, xmax=0.495, c= 'k', linewidth = 15.0, label = 'outcome On period')

            yerr = self.sem
            yerr1 = self.sem_inc
            axs[p-1].errorbar(X, Y, yerr =yerr,fmt='o', c = colors[0], alpha = 0.2)
            axs[p-1].errorbar(X1, Y1, yerr =yerr1,fmt='o', c = colors[1], alpha = 0.2)
