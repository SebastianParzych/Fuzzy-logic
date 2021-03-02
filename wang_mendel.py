import pandas as pd
import numpy as np
import math as math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statistics
import csv

#-----------------------------------------------------------------------------------------------------------------------------#
#------------------------------VIRTUAL FLOW SENSOR DESIGN USING THE WANG-MENDEL METHOD----------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------#

P1_row = "P1[kPa]"
P2_row = "P2[[kPa]"
F_row = "F[m^3/h]"
X_row = "X[%]"
df = pd.read_csv('data_zawor_2001-1.csv', sep=',')
df_dP = df[P1_row][df.index < 1800] - df[P2_row][df.index < 1800]
df_X = df[X_row][df.index < 1800]
df_F = df[F_row][df.index < 1800]
data=np.array( [df_X, df_dP, df_F]) # learning dataset
val_data=np.array( [df[X_row][1801:], df[P1_row][1801:]-df[P2_row][1801:]]) # Validation dataset
val_F=df[F_row][1801:] # validation dataset of output
templateX = ["S2", "S1", "CE", "B1", "B2"," "] # Linguistic names for default number of regions
templateY = [" ","S2", "S1", "CE", "B1", "B2"]

regions_number=5  # number of region,
N=regions_number/2 -1


peaks = np.zeros((3,regions_number)) # points in datasets of maximum value of membership functions
rules = np.zeros((regions_number, regions_number)) # Rule base
rules_y =np.zeros((regions_number, regions_number))-1 # Mapping output values in rule base

predictions = []
function_of_region1 = [] # lists to create chart of all functions of membership
function_of_region2 = []

def peaks_def (data, num,  regions_number): #definition of all peaks of membership functions for a given input/output data
    max_v=max(data)
    min_v=min(data)
    for i in range( regions_number):
        peaks[num][i]=min_v+i*(max_v-min_v)/(N*2)
def member_def(rekord,num,con, regions_number): # blurring
    mem=np.zeros(( regions_number,1))
    for i in range ( regions_number):
        if(i+1<regions_number):
            if (rekord >= peaks[num,i] and rekord <= peaks[num,i+1]):
                mem[i][0] = (peaks[num, i + 1] - rekord) / (peaks[num, i + 1] - peaks[num, i])
                mem[i+1][0] = (rekord - peaks[num, i]) / (peaks[num, i + 1] - peaks[num,i])
                break
    if (rekord > peaks[num][regions_number - 1]):
        mem[regions_number -1][0] =1
    if (rekord < peaks[num][0]):
        mem[0][0] = 1
    index = np.where(mem != 0)[0]
    if(con == True and num == 1 ): # collecting values for chart
        if len(index)==2:
            function_of_region1.append(float(mem[index[0]]))
            function_of_region2.append(float(mem[index[1]]))
        else:
            function_of_region1.append(float(mem[index[0]]))
            function_of_region2.append(0)
    if ( con == True and len(index)==2):
            if (mem[index[0]][0] > mem[index[1]][0]):
                mem[index[1]][0] = 0;
            else:
                mem[index[0]][0] = 0
    return mem
def allpeaks_def(data,  regions_number): # definition of all characteristic points of fuzzy sets
    for i in range(len(data)):
        peaks_def(data[i],i,  regions_number)
def fuzzy_def (data,num, bool,  regions_number): # saving to list fuzzy values for each set
    all = []
    for i in data[num]:
        all.append(member_def(i,num,bool, regions_number))
    return np.array(all)
def power_def(data, regions_number): #  Making rule base
    allpeaks_def(data,  regions_number)
    fuzzy_X = fuzzy_def(data,0, True, regions_number)
    fuzzy_P = fuzzy_def(data,1, True, regions_number)
    fuzzy_F = fuzzy_def(data, 2, True,  regions_number)
    indexX = np.where(fuzzy_X != 0)[1]
    indexP = np.where(fuzzy_P != 0)[1]
    indexF= np.where(fuzzy_F != 0)[1]
    all_id=[]
    all_id.append(indexX)
    all_id.append(indexP)
    all_id.append(indexF)
    for record in range(len(data[0])):
        degree = fuzzy_X[record, all_id[0][record]] * fuzzy_P[record, all_id[1][record]] * fuzzy_F[record, all_id[2][record]]
        if degree > rules[all_id[0][record], all_id[1][record]]:
            rules[all_id[0][record], all_id[1][record]] = degree
            rules_y[all_id[0][record], all_id[1][record]] =int(all_id[2][record])

def deffuzing (validation_data,  regions_number): # defuzzification
    fuzzy_X = fuzzy_def(validation_data, 0, False,  regions_number) # declaration fuzzy set of data X
    fuzzy_P = fuzzy_def(validation_data, 1, False, regions_number)  # declaration fuzzy set of data P
    for record in range (len(val_data[0])):
        mem_sum=0
        prod_sum=0
        for i in range( regions_number):
            for j in  range( regions_number):
                prod_val=0
                mem_val=0
                if( rules_y[i][j] == -1):
                    continue
                else:
                    center=peaks[2,int(rules_y[i,j])]
                    mem_sum += float(fuzzy_X[record][i])*float(fuzzy_P[record][j])
                    prod_sum += float(center * float(fuzzy_X[record][i])*float(fuzzy_P[record][j]))

        sharp_prediction=prod_sum/mem_sum
        predictions.append(float(sharp_prediction))
def error_count (predictions, real_output): # basic calculus of  error statistic
    l = []
    l=(real_output-predictions)/max(data[2]) * 100
    avarage_error=sum(abs(l))/len(l)
    max_error = max(l)
    min_error = min (l)
    print('Minimal error: '+str(min_error))
    print('Maximum error: '+str(max_error))
    print('Avarage error: '+str(avarage_error))
    print("Standard Deviation of sample: "   + str(statistics.stdev(predictions)))
    print('Statistic harmonic mean:  '+str(statistics.harmonic_mean(predictions)))
    return l
#-----------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------PLOTS---------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------#
def one_plot ():
    x = np.linspace(0, 1800, num=1800)
    plt.figure(figsize=(12,5))
    plt.plot(x,val_F, label=" Real")
    plt.plot(x,predictions,label=" Predictions ")
    plt.xticks(np.arange(0, 1900, 100))
    plt.title('Mandani implication results')
    plt.ylabel('F[m^3/h]')
    plt.xlabel('time[s]')
    plt.legend(loc="lower right")
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
def histogram_draw (data , tittle, xname, yname):
    x=data
    bins_num= 80
    plt.hist(x, bins= bins_num)
    plt.title(tittle)
    plt.ylabel(yname)
    plt.xlabel(xname)
    plt.show()
def controlPane_draw (data1,data2,data3, tittle, xname, yname, zname):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(data1, data2, data3, linewidth=0.2, antialiased=True, cmap='plasma')
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_zlabel(zname)
    fig.suptitle(tittle)
    plt.show()
def draw_function_ofregion():
    plt.figure(figsize=(12, 5))
    plt.plot(data[1], function_of_region1, 'o', color="tab:blue")
    plt.plot(data[1], function_of_region2, 'o', color="tab:blue")
    plt.title("Functions of Region")
    plt.xlabel('dP[[kPa]')
    plt.ylabel('u(X)')
    plt.show()

def  allPlotsdraw (validation, error): #This function calls above functions to create all needed charts
    if validation:
        controlPane_draw(val_data[0],val_data[1],val_F, "Plane of Control", "X[%]", "dP[[kPa]", "F[m^3/h]")
        controlPane_draw(val_data[0],val_data[1],predictions, "Plane of Control", "X[%]", "dP[[kPa]", "F[m^3/h]")
        histogram_draw(predictions, 'Histogram of predictions', 'Predictions', 'Frequency')
        one_plot(val_F, predictions, 'Wang-Mendel results ', 'F[m^3/h]', 'time[s]', 'Real', " Predictions")
        one_plot(error, 0, 'Relative error', '[%]', 'time[s]', 'error', '')
    else:
        controlPane_draw(data[0], data[1], data[2], "Plane of Control", "X[%]", "dP[[kPa]", "F[m^3/h]")
        controlPane_draw(data[0], data[1], predictions, "Plane of Control", "X[%]", "dP[[kPa]", "F[m^3/h]")
        histogram_draw(predictions, 'Histogram of predictions', 'Predictions', 'Frequency')
        one_plot(data[2], predictions, 'Wang-Mendel results', 'F[m^3/h]', 'time[s]', 'Real', " Predictions")
        one_plot(error, 0, 'Relative error', '[%]', 'time[s]', 'error', '')
def Mangami(validation):
    if validation:
        power_def(data, regions_number)
        deffuzing(val_data,regions_number)
        error = error_count(predictions,val_F)
    else:
        power_def(data, regions_number)
        deffuzing(data, regions_number)
        error = error_count(predictions, data[2])
    draw_function_ofregion()

    print(rules)
    print(rules_y)
    allPlotsdraw(validation, error)

Mangami(True) # True- test model on val dataset, False- test model on learn dataset



