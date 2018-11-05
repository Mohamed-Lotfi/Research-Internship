# Summer-Research-Python
A collection of python files and classes created for my data analysis during my internship at Boston University's Cognition and Decision Lab. Here is a short description of each file:

1) Pupil_avg: A python class with functions to aggregate, graph and perform multiple other operations on a pupillometry data set from a spatial prediction psychology experiment.

2) Gaze_corrector: A python class to collect pupil diameter data, gaze position coordinates along with other data from the experiment's csv files outputed by the eye-tracker. The file also includes functions that splice and segment the data and put it into a usable format(mostly numpy arrays and pandas dataframes). This is essentially a collection of data preprocessing tools.

3) Correction_class: This file takes care of correcting the eye-tracker's data in order to account for the artifitial dialation and constriction of the pupil diameter that is can be attributed to the gaze position of the eye during the experiment. Multiple methods are included in the class including different degrees of polynomial regression(cubic, quadratatic linear, etc.). Ultimately, the method that was used was a cubic b-spline fitting algorithm that performed best on our dataset. 
