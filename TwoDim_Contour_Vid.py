import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.tri as tri
import scipy.stats as st
from scipy.interpolate import Rbf
from scipy import interpolate, optimize
from scipy.interpolate import griddata
import numpy as np
import pandas as pd
import math
from sigfig import round
from cycler import cycler
from statistics import mean
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# A) "TwoDim_ContourVid" : plots .gif video of stitched series of 2D contour plots, using a data in the form [A1, A2, A3, .. An] and [B1, B2, B3, .. Bn]. Also plots of all of the raw data, for comparison/check of the contours. For example, data may of calculations such as A = Mg / Mg + Fe = Mg#,  B = Si + Al / Si + Al + Mg + Fe = SiRatio.

#   A.1) "init_TwoDimVid" : makes a blank contour plot prior to looping through each 'window' of data. Required for video (.gif) making.

#   A.2) "animate_TwoDimVid" : steps through each window of data plotting a 2D contour plot for each.

def TwoDim_ContourVid(yData, xData, yMax, yMin, xMax, xMin, NumLevels, Colour, yLabel, xLabel, WindDispType, WindDispPositionData, WindowWidth, NumberOfSteps, fps, StartingIndex_FirstWindow, FigureSavePath, FileName):
    """
    ** yData : Data to be plotted in the y-axis, in the form of a list [A1, A2, A3, .. An]
    ** xData : Data to be plotted in the x-axis, in the form of a list [B1, B2, B3, .. Bn]
    ** yMax : Coordinate for y-axis maximum of plot.
    ** yMin : Coordinate for y-axis minimum of plot.
    ** xMax : Coordinate for x-axis maximum of plot.
    ** xMin : Coordinate for x-axis minimum of plot.
    ** yLabel : In quotations ('test' for example), the text to be displayed in the y-axis of the plot.
    ** xLabel : In quotations ('test' for example), the text to be displayed in the x-axis of the plot.
    ** NumLevels: Number of contour levels.
    ** Colour : In quotations (such as 'Blues') the colour of the contour plot. Options available according to 'cmap' of matplotlib (https://matplotlib.org/stable/tutorials/colors/colormaps.html).
    ** WindDispType : either 'index', 'distance' or 'none' to display the  position of the window for each frame of the video (continously plots the first and last index of the datapoints considered for videoframe's contour). User specifies the type of identifier of the position of the window (data index, or physical position, such as distance in 'nm' from a user-defined datum like the end of an EDS line-scan).
    ** WindDispPositionData : either a simple 1D array of position data [X1, X2, X3, ... Xn] where n = number of datapoints in 'dataset', or 'none' if WindDispType is set to either 'index' or 'none.
    ** WindowWidth : width (number of data points) to consider for each contour plot (each videoframe). Higher number means smoother contours but lower 'resolution' in window position (wider range of data considered to make each contour).
    ** NumberOfSteps : the number of data points ([Si+Al, Fe, Mg] triplicates) which to scan through. Can be the algebraic solution to the entire dataset (considering the user-defined value of the WindowWidth), or can be less if a subset of the data is only wanted to be considered.
    ** fps : frames per second of the video. Increase if it progresses too slow, decrease if it plays too fast.
    ** StartingIndex_FirstWindow : index of the first data point. Do you want to start at the first data point (=0), or somewhere offset from it? (=100 for example). Thus: StartingIndex_FirstWindow + WindowWidth * NumberOfSteps = Very last index of data considered (in final video frame).
    ** FigureSavePath : Path to folder location where figures should be saved. Must be in single quotations, example : '/Volumes/Samsung_T5/Experiment categories/Laser/Figures/'
    ** FileName : General name of files to be saved. Must be in single quotations, example : 'TEST_DataSet01'
    """
    
    FinalIndex_LastWindow = StartingIndex_FirstWindow +  WindowWidth + NumberOfSteps
      
    with plt.style.context(['Icarus_Base', 'Preferences_Base']):
        fig, ax = plt.subplots()
     
    def init_TwoDimVid():
        """ This is an initial contour plot, using blank data to initialize the 2D plot. Required to be run before real data is plotted.
        """
        with plt.style.context(['Icarus_Base', 'Preferences_Base']):
        # Peform the kernel density estimate
            xx, yy = np.mgrid[xMin:xMax:100j, yMin:yMax:100j]
    #        with plt.style.context(['Icarus_Base', 'Preferences_Base']):
            ax.contourf(xx, yy, np.zeros((100, 100)), NumLevels, cmap=Colour)
            ax.axis('off')
                
    def animate_TwoDimVid(i):
        with plt.style.context(['Icarus_Base', 'Preferences_Base']):
            ax.clear() # Clearing axis with every new frame. Will allow re-writing of window position, ect..
            StartingIndex = StartingIndex_FirstWindow + i # index of one end of the data window.
            EndingIndex = StartingIndex_FirstWindow +  WindowWidth + i # index of the second end of the data window.
                    
            # Defining temporary lists of data and the contour's gris for the particular video frame (i).
            data = data_list[i]
            xx = xx_list[i]
            yy = yy_list[i]
            
            # plotting contour for video frame i.
            ax.contourf(xx, yy, data, NumLevels, cmap=Colour)
            
            # Setting the figure's position of the text for the window position (in index # or distance, as defined by user). In same units as current x- and y-axis.
            WindowPosition_TextLoc_xAxis = xMin + 0.75*(xMax - xMin)
            WindowPosition_TextLoc_yAxis = yMin + 0.85*(yMax - yMin)
                    
            # displaying the position of the window in the units of data indexes.
            if WindDispType == 'index':
                ax.text(WindowPosition_TextLoc_xAxis, WindowPosition_TextLoc_yAxis, '[' + str(StartingIndex) + ' - ' + str(EndingIndex) + ']')
            elif WindDispType == 'distance':
                ax.text(WindowPosition_TextLoc_xAxis, WindowPosition_TextLoc_yAxis, '[' + str(int(WindDispPositionData.iloc[StartingIndex])) + ' - ' + str(int(WindDispPositionData.iloc[EndingIndex])) + 'nm]')
            ax.set_ylabel(yLabel)
            ax.set_xlabel(xLabel)
            ax.set_xlim(xMin, xMax)
            ax.set_ylim(yMin, yMax)
#            ax.axis('off')
            
    # Making 2D plot contour video.
    data_list = []
    xx_list = []
    yy_list = []
    for j in range(NumberOfSteps-1):
        StartingIndex = StartingIndex_FirstWindow + j # index of one end of the data window.
        EndingIndex = StartingIndex_FirstWindow +  WindowWidth + j # index of the second end of the data window.
        
        y = yData.iloc[StartingIndex: EndingIndex]
        x = xData.iloc[StartingIndex: EndingIndex]
        
        # Peform the kernel density estimate
        xx, yy = np.mgrid[xMin:xMax:200j, yMin:yMax:200j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = st.gaussian_kde(values)
        data = np.reshape(kernel(positions).T, xx.shape)
        data_list.append(data)
        xx_list.append(xx)
        yy_list.append(yy)


    anim = animation.FuncAnimation(fig, animate_TwoDimVid, init_func=init_TwoDimVid, frames=NumberOfSteps-1, repeat = False)
    ax.axis('off')
    pillowwriter = animation.PillowWriter(fps)
    anim.save(FigureSavePath + FileName +  '_TwoDimension_Contour_Video.gif', writer=pillowwriter)
        
