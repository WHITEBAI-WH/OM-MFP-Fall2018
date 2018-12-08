# The PredictFail class takes as input time-between-failure data
# from over 60 opthalmic manufacturing machines.  The data is used to
# train/derive MLE parameters for all machine-failure combinations.  
# To calculate the RUL % for each machine-failure type, MLE parameters are added to
# the last failure date of each machine-failure type and then divided by the current date.
# Author: Daniel McAdams
# Date: 12/08/2018

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from scipy.stats import norm
import math

class PredictFail:

    def __init__(self):
        self.MLE_Params = []
        self.RUL = []
        self.ST = []
        self.outbl = []
        self.fail_cat = []
        self.mach_cat = []
        self.df1 = pd.DataFrame()
        self.df2 = pd.DataFrame()
        self.fail_cat = ["Other",
                    "Cal./Diamond",
                    "Polish",
                    "Axis",
                    "Suction",
                    "Missing/Lost Lens",
                    "Sensor",
                    "Pump",
                    "Gripper",
                    "Leak",
                    "Freezing",
                    "Dropping Lens"]
        self.mach_cat = ["AR_Robotics",
                    "Generator",
                    "Polisher",
                    "Blocker",
                    "Laser",
                    "Deblocker",
                    "Lens Washer",
                    "Backside Coater",
                    "Vacuum Coater",
                    "Tape Stripper",
                    "Generator Separator",
                    "Surface Taper"]
        self.read_data()

    #reads training data and test data into pandas dataframe
    def read_data(self):
        file = "C:\\Users\danielmcadams\Desktop\Training_Data.csv"
        data = open(file)
        #MLE training data set
        self.df1 = pd.read_csv(data, delimiter=',',
                  dtype=None, converters={ 1: mdates.strpdate2num('%m/%d/%Y %H:%M')})
        data.close()

        #Failure prediction data set
        file2 = "C:\\Users\danielmcadams\Desktop\Test_Data.csv"
        data2 = open(file2)
        self.df2 = pd.read_csv(data2, delimiter=',',
                dtype=None,date_parser = self.parse_dates, parse_dates=[1])
        data2.close()

    #calculates the remaining useful life from MLE parameters 
    def calc_rul(self):
        for x in range(len(self.df2)):
            #gets mach_cat from df2 to locate index range of MLE_Params
            m1 = self.df2.iat[x,2]*12  
            #maximum likelihood estimate index - adds machine category and failure type
            index = m1 + self.df2.iat[x,3]
            #difference between current datetime and the last failure datetime in hours (Observed!)
            delta = (datetime.now() - self.df2.iat[x,1]).total_seconds()/3600
            #remaining useful life 
            newRul =  1 - (delta / self.MLE_Params[index])
            #store remaining useful life as probability
            self.RUL.append(newRul)

    #utilizes training data to establish MLE parameters
    def build_MLE(self):
        mach = []
        ft = []
        for m in range(12):  #each machine category
            for i in range(12):  #each failure type
                ftype = pd.DataFrame()
                ftype = self.df1.drop(self.df1[((self.df1.mach_cat != m) & (self.df1.failure_type != i))].index)
                if ftype.empty or len(ftype)<30:
                    self.MLE_Params.append(0)
                    #creates lists for machine types and failure types (for creating table)
                    mach.append(self.mach_cat[i])
                    ft.append(self.fail_cat[i])
                elif not ftype.empty and len(ftype)>=30:
                    maxLL = self.getMaxLik(ftype)
                    self.MLE_Params.append(maxLL)
                    #creates lists for machine types and failure types (for creating table)
                    mach.append(self.mach_cat[i])
                    ft.append(self.fail_cat[i])
                del ftype

        data = {'Machine Cat.': mach, 'Failure Type': ft, 'MLE': self.MLE_Params}
        tbl = pd.DataFrame(data=data)
        #print(tbl)
        #print("MLE_PARAMS: ", self.MLE_Params) 
          
    #utility function to establish MLE parameters         
    def getMaxLik(self,failType):
        mean = failType.hours_bf.mean()
        stdev = failType.hours_bf.std()
        max = 0
        logMax = -100000000
        for x in failType.hours_bf:
            lik = np.log(norm.pdf(x,mean,stdev))
            if(logMax < lik):
                logMax = lik
                max = x
        return max
    
    #prints remaining life until failure estimate by machine and failure type
    def output_summary_table(self):
        a = []
        rl = []
        m = []
        f = []
        lf = []
        stp = []
        mle = []
        for i in range(len(self.df2)):
            a.append(self.df2.iat[i,0])
            lf.append(self.df2.iat[i,1])
            #gets mach_cat from df2 to locate index range of MLE_Params
            m1 = self.df2.iat[i,2]*12  
            #maximum likelihood estimate index - adds machine category and failure type
            index = m1 + self.df2.iat[i,3]

            m.append(self.mach_cat[self.df2.iat[i,2]])
            f.append(self.fail_cat[self.df2.iat[i,3]])            

            if self.RUL[i] <= 0:
                rl.append(0)
            else:
                rl.append(round(self.RUL[i]*100,2))
 
            stp.append(round(self.ST[i],4))
            mle.append(self.MLE_Params[index])
                
        data = {'Mach #': a, 'Last Fail': lf, 'Mach Cat.': m, 'Fail Type': f, 'MLE(hours)': mle,
                'RUL(%)': rl, 'State Prob.': stp}
        self.outbl = pd.DataFrame(data=data).sort_values(['State Prob.'],ascending=False)
        np.savetxt("rul_output.txt", self.outbl.values, delimiter='\t',
                   header='%s%25s%15s%20s%15s%15s%15s'%('Mach#','Last Failure','Mach Cat','Fail Type','MLE(hours)','RUL(%)','State Prob.'),
                   fmt='%i%25s%15s%20s%15f%15f%15f', comments='')
        print("RUL summary table compiled...")

    #calculates probabilities for all machines and failure types to establish priority
    def create_state_probs(self):
        total = 0
        #print(self.RUL)
        for i in self.RUL:
                total += 1-i
        for i in self.RUL:
                self.ST.append((1-i)/total)
        
    #converts string to date with format m/d/y H:M
    def parse_dates(self, x):
        return datetime.strptime(x,'%m/%d/%Y %H:%M')

    #outputs a plot of RUL% to last failure date
    def create_plot(self):
        #Define the date format
        myFmt = DateFormatter("%m/%d/%Y %H:%M")
        
        x = self.outbl.drop(['Mach #','State Prob.','Mach Cat.','Fail Type','MLE(hours)','RUL(%)'], axis=1)
        y = self.outbl.drop(['Mach #', 'State Prob.','Last Fail','Mach Cat.','Fail Type','MLE(hours)'], axis=1)

        # plot the data
        fig, ax1 = plt.subplots()
        #ax1.plot(x, y, 'ro', color='blue', s=3)
        plt.plot(x, y, 'ro')
        plt.xlabel('Failure Date', fontsize=18)
        plt.ylabel('RUL %', fontsize=16)
        plt.title('Remaining Useful Life - Current Date 12/09/2018', fontsize=16)
        plt.show()      


    
    


