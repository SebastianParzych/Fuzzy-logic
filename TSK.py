import pandas as pd
import numpy as np
import math as math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statistics
import csv
#-----------------------------------------------------------------------------------------------------------------------------#
#------------------------------VIRTUAL FLOW SENSOR DESIGN USING TAKAGI-SUGENO-KANG METHOD----------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------#

df = pd.read_csv('data_zawor_2001-1.csv', sep=',')
data=np.array( [ df["X[%]"][0:1800],df["P1[kPa]"][0:1800]-df["P2[[kPa]"][0:1800], df["F[m^3/h]"][0:1800]]) # learn dataset
val_data=np.array( (df["X[%]"][1801:], (df["P1[kPa]"][1801:]-df["P2[[kPa]"][1801:]), df["F[m^3/h]"][1801:])) # validation dataset

template=["S0","CE","B0"] # names of triangular functions of membership
regions=3 # number of regions
N=1
peaks=np.zeros((3,regions)) # points in datasets of maximum value of membership functions
predictions= []

def allpeaks(data): #definition of all peaks of membership functions for a given input/output data
        for i in range (len(data)):
            max_v=max(data[i])
            min_v=min(data[i])
            for j in range (regions):
               peaks[i][j]=min_v+ j * (max_v-min_v)/(2*N)
def d_degree (data,number, implication): # functions of membership, blurring
    one_series=[]
    for record in data:
        membership = np.zeros((regions, 1))
        for i in range (regions-1):
            if (record >= peaks[number, i] and record <= peaks[number, i + 1]):
                membership[i][0] = (peaks[number, i + 1] - record) / (peaks[number, i + 1] - peaks[number, i])
                membership[i + 1][0] = (record - peaks[number, i]) / (peaks[number, i + 1] - peaks[number, i])
                break
        index = np.where(membership != 0)[0]

        if ( implication == True and len(index) == 2):   # Picking maximum value of function of membership
            if (membership[index[0]][0] > membership[index[1]][0]):
                membership[index[0]][0] = 0;
            else:
                membership[index[1]][0] = 0

        one_series.append(membership)
    return one_series
def all_degres(data,implication): #  for each input dataset genereting fuzzy values
    series=0
    all_membership=[]
    for i in data:
        all_membership.append(d_degree(i,series,implication))
        series+=1
    return all_membership
def set_rules (input, list_of_mem): # A set of rules, returning  sharp exits of specific premises
    y_all= []
    for i in range(len(data[0])):
        y= {}
        if (list_of_mem[0][i][0] != 0 and list_of_mem[1][i][0] != 0):
            y[0] = max(data[2]) - 0 * input[0][1] - 0.01 * input[1][i] ** 0.5
        if (list_of_mem[0][i][0] != 0 and list_of_mem[1][i][1] != 0):
            y[1] = max(data[2]) - 0.05 * input[0][1] - 0.1 * input[1][i] ** 0.5
        if (list_of_mem[0][i][0] != 0 and list_of_mem[1][i][2] != 0):
            y[2] = max(data[2]) - 0 * input[0][1] - 0 * input[1][i] ** 0.5
        if (list_of_mem[0][i][1] != 0 and list_of_mem[1][i][0] != 0):
            y[3] = max(data[2]) - 0.55 * input[0][1] - 0.5 * input[1][i] ** 0.5
        if (list_of_mem[0][i][1] != 0 and list_of_mem[1][i][1] != 0):
            y[4] = max(data[2]) - 0.5 * input[0][1] - 0.5 * input[1][i] ** 0.5
        if (list_of_mem[0][i][1] != 0 and list_of_mem[1][i][2] != 0):
            y[5] = max(data[2]) - 0.80 * input[0][1] - 0.8 * input[1][i] ** 0.5
        if (list_of_mem[0][i][2] != 0 and list_of_mem[1][i][0] != 0):
            y[6] = max(data[2]) - 1.15 * input[0][1] - 1.5 * input[1][i] ** 0.5
        if (list_of_mem[0][i][2] != 0 and list_of_mem[1][i][1] != 0):
            y[7] = max(data[2]) - 1 * input[0][1] - 0.71 * input[1][i] ** 0.5
        if (list_of_mem[0][i][2] != 0 and list_of_mem[1][i][2] != 0):
            y[8] = max(data[2]) - 1.3 * input[0][1] - 1.2 * input[1][i] ** 0.5
        y_all.append(y)

    print("--------")
    return y_all
def switchfun(i,memX1, memX2): # A function that returns the result of Mandami implication
     switcher={
        0: min(memX1[0],memX2[0]),
        1: min(memX1[0],memX2[1]),
        2: min(memX1[0],memX2[2]),
        3: min(memX1[1],memX2[0]),
        4: min(memX1[1],memX2[1]),
        5: min(memX1[1],memX2[2]),
        6: min(memX1[2],memX2[0]),
        7: min(memX1[2],memX2[1]),
        8: min(memX1[2],memX2[2]),
     }
     return switcher.get(i)
def deffuzying (output, memberships): # defuzzification
    for record in range (len(data[0])):
        counter_sum=0
        denom_sum = 0
        for i in output[record]:
          min_val= switchfun(i,memberships[0][record],memberships[1][record])
          counter_sum+=min_val*output[record][i]
          denom_sum+=min_val
        predictions.append(float(counter_sum/denom_sum))
def errocount (real): # Wyznaczanie statystyki błędów
    l = []
    l = (real[2] - predictions) / max(data[2]) * 100
    avarage_error = sum(abs(l)) / len(l)
    max_error = max(l)
    min_error = min(l)
    print('Minimal error: ' + str(min_error))
    print('Maximum error: ' + str(max_error))
    print('Avarage error: ' + str(avarage_error))
    print("Standard Deviation of sample: " + str(statistics.stdev(predictions)))
    print('Statistic harmonic mean:  ' + str(statistics.harmonic_mean(predictions)))
    return l
# -----------------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------PLOTS---------------------------------------------------------------#
# -----------------------------------------------------------------------------------------------------------------------------#
def controlPane_draw (data1,data2,data3, tittle, xname, yname, zname):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(data1, data2, data3, linewidth=0.2, antialiased=True, cmap='plasma')
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_zlabel(zname)
    fig.suptitle(tittle)
    plt.show()
def histogram_draw (data , tittle, xname, yname):
    x=data
    bins_num= 80
    plt.hist(x, bins= bins_num)
    plt.title(tittle)
    plt.ylabel(yname)
    plt.xlabel(xname)
    plt.show()
def one_plot (data1, data2 , tittle, yname, xname, plotname1, plotname2):
    x=np.linspace(0,1800, num = 1800)
    plt.figure(figsize=(12, 5))
    plt.plot(x, data1, label=plotname1)
    if data2 != 0:
        plt.plot(x, data2, label=plotname2)
    plt.xticks(np.arange(0, 1900, 100))
    plt.title(tittle)
    plt.ylabel(yname)
    plt.xlabel(xname)
    plt.legend(loc="lower right")
    plt.show()

def  allPlotsdraw (X1,X2,Y_real,Y_predictions,error): # funkcja rysująca wszystkie wykresy
    controlPane_draw(X1,X2,Y_real, "Plane of real control", "X[%]", "dP[[kPa]", "F[m^3/h]")
    controlPane_draw(X1,X2,Y_predictions, "Plane of predicted control", "X[%]", "dP[[kPa]", "F[m^3/h]")
    histogram_draw(Y_real, 'Histogram of Real values', 'Predictions', 'Frequency')
    histogram_draw(Y_predictions, 'Histogram of predictions', 'Predictions', 'Frequency')
    one_plot(Y_real, Y_predictions, 'Takagi-Sugeno-Kang results', 'F[m^3/h]', 'time[s]', 'Real', " Predictions")
    one_plot(error, 0, 'Relative error', '[%]', 'time[s]', 'error', '')

def runProject(): # Funkcja wykonująca wszystkie potrzebne do projektu operacje
    allpeaks(data)

    #test on learn dataset
    list_of_mem = all_degres(data, False)
    y = set_rules(data, list_of_mem)
    deffuzying(y, list_of_mem)
    error=errocount(data)
    allPlotsdraw(data[0],data[1],data[2],predictions,error)

    #Test on validation dataset
    predictions.clear()
    list_of_mem = all_degres(val_data, False)
    y = set_rules(val_data, list_of_mem)
    deffuzying(y, list_of_mem)
    error = errocount(val_data)
    allPlotsdraw(val_data[0], val_data[1], val_data[2], predictions, error)



runProject()
