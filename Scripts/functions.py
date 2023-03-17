# Functions for the BOOM TephraDataSet exploration
from re import X
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
warnings.simplefilter("ignore")

#--------------------- General functions --------------------------------------------------------------
def simbologia(volcano, event):

    simbología = pd.read_csv('../../Scripts/Simbologia.csv', encoding = 'latin1', low_memory=False)
    Event = simbología.loc[simbología['Volcano'] == volcano]
    Event = Event.loc[Event['Event'] == event]
    coloR = Event.values[0, 2]
    markeR = Event.values[0, 3]
    return coloR, markeR

#--------------------- Functions for CheckNormalizations notebook --------------------------------------
def renormalizing(BOOM_dataset):

    BOOM_dataset_renormalized = BOOM_dataset.copy()
    BOOM_dataset_renormalized['MnO'] = BOOM_dataset_renormalized['MnO'].replace('-',-1).astype(float)
    BOOM_dataset_renormalized['P2O5'] = BOOM_dataset_renormalized['P2O5'].replace('-',-1).astype(float)
    BOOM_dataset_renormalized['Cl'] = BOOM_dataset_renormalized['Cl'].replace('-',-1).astype(float)
    BOOM_dataset_renormalized['LOI'] = BOOM_dataset_renormalized['LOI'].replace('-',-1).astype(float)
    BOOM_dataset_renormalized['FeO'] = BOOM_dataset_renormalized['FeO'].replace(np.nan,-1)
    BOOM_dataset_renormalized['Fe2O3'] = BOOM_dataset_renormalized['Fe2O3'].replace(np.nan,-1)
    BOOM_dataset_renormalized['FeOT'] = BOOM_dataset_renormalized['FeOT'].replace(np.nan,-1)
    BOOM_dataset_renormalized['Fe2O3T'] = BOOM_dataset_renormalized['Fe2O3T'].replace(np.nan,-1)

    #Defining some variables which we will plot later to understand the variability of the re normalized data 
    BOOM_dataset_renormalized['MnO + P2O5 + Cl'] = 'default'
    BOOM_dataset_renormalized['Analytical Total without LOI'] = 'default'

    for i in range(0,len(BOOM_dataset_renormalized['Total'])):
        
        if (BOOM_dataset_renormalized['FeOT'][i] != -1)&(BOOM_dataset_renormalized['Fe2O3T'][i] == -1)&(BOOM_dataset_renormalized['FeO'][i] == -1)&(BOOM_dataset_renormalized['Fe2O3'][i] == -1):
            sum_ = np.nansum([BOOM_dataset_renormalized['SiO2'][i],
                      BOOM_dataset_renormalized['TiO2'][i],
                      BOOM_dataset_renormalized['Al2O3'][i],
                      BOOM_dataset_renormalized['FeOT'][i], #the samples tested have been analyzed by EMP, thus FeO corresponds to FeOT
                      BOOM_dataset_renormalized['MgO'][i],
                      BOOM_dataset_renormalized['CaO'][i],
                      BOOM_dataset_renormalized['Na2O'][i],
                      BOOM_dataset_renormalized['K2O'][i]])
            
            BOOM_dataset_renormalized.loc[i,'SiO2_normalized'] = BOOM_dataset_renormalized['SiO2'][i]*100/sum_
            BOOM_dataset_renormalized.loc[i,'TiO2_normalized'] = BOOM_dataset_renormalized['TiO2'][i]*100/sum_
            BOOM_dataset_renormalized.loc[i,'Al2O3_normalized'] = BOOM_dataset_renormalized['Al2O3'][i]*100/sum_
            BOOM_dataset_renormalized.loc[i,'FeOT_normalized'] = BOOM_dataset_renormalized['FeOT'][i]*100/sum_
            BOOM_dataset_renormalized.loc[i,'MgO_normalized'] = BOOM_dataset_renormalized['MgO'][i]*100/sum_
            BOOM_dataset_renormalized.loc[i,'CaO_normalized'] = BOOM_dataset_renormalized['CaO'][i]*100/sum_
            BOOM_dataset_renormalized.loc[i,'Na2O_normalized'] = BOOM_dataset_renormalized['Na2O'][i]*100/sum_
            BOOM_dataset_renormalized.loc[i,'K2O_normalized'] = BOOM_dataset_renormalized['K2O'][i]*100/sum_

            BOOM_dataset_renormalized.loc[i,'Total_normalization'] = sum_
                

        if (BOOM_dataset_renormalized['FeOT'][i] == -1)&(BOOM_dataset_renormalized['Fe2O3T'][i] != -1)&(BOOM_dataset_renormalized['FeO'][i] == -1)&(BOOM_dataset_renormalized['Fe2O3'][i] == -1):
            sum_ = np.nansum([BOOM_dataset_renormalized['SiO2'][i],
                      BOOM_dataset_renormalized['TiO2'][i],
                      BOOM_dataset_renormalized['Al2O3'][i],
                      BOOM_dataset_renormalized['Fe2O3T'][i]*0.899, 
                      BOOM_dataset_renormalized['MgO'][i],
                      BOOM_dataset_renormalized['CaO'][i],
                      BOOM_dataset_renormalized['Na2O'][i],
                      BOOM_dataset_renormalized['K2O'][i]])

            BOOM_dataset_renormalized.loc[i,'SiO2_normalized'] = BOOM_dataset_renormalized['SiO2'][i]*100/sum_
            BOOM_dataset_renormalized.loc[i,'TiO2_normalized'] = BOOM_dataset_renormalized['TiO2'][i]*100/sum_
            BOOM_dataset_renormalized.loc[i,'Al2O3_normalized'] = BOOM_dataset_renormalized['Al2O3'][i]*100/sum_
            BOOM_dataset_renormalized.loc[i,'Fe2O3T_normalized'] = BOOM_dataset_renormalized['Fe2O3T'][i]*100/sum_
            BOOM_dataset_renormalized.loc[i,'MgO_normalized'] = BOOM_dataset_renormalized['MgO'][i]*100/sum_
            BOOM_dataset_renormalized.loc[i,'CaO_normalized'] = BOOM_dataset_renormalized['CaO'][i]*100/sum_
            BOOM_dataset_renormalized.loc[i,'Na2O_normalized'] = BOOM_dataset_renormalized['Na2O'][i]*100/sum_
            BOOM_dataset_renormalized.loc[i,'K2O_normalized'] = BOOM_dataset_renormalized['K2O'][i]*100/sum_
            BOOM_dataset_renormalized.loc[i,'Total_normalization'] = sum_
                                      

        if ((BOOM_dataset_renormalized['FeO'][i] != -1)&(BOOM_dataset_renormalized['Fe2O3'][i] != -1))&(BOOM_dataset_renormalized['FeOT'][i] == -1):
            sum_ = np.nansum([BOOM_dataset_renormalized['SiO2'][i],
                      BOOM_dataset_renormalized['TiO2'][i],
                      BOOM_dataset_renormalized['Al2O3'][i],
                      BOOM_dataset_renormalized['FeO'][i], 
                      BOOM_dataset_renormalized['Fe2O3'][i]*0.899, 
                      BOOM_dataset_renormalized['MgO'][i],
                      BOOM_dataset_renormalized['CaO'][i],
                      BOOM_dataset_renormalized['Na2O'][i],
                      BOOM_dataset_renormalized['K2O'][i]]) 

            BOOM_dataset_renormalized.loc[i,'SiO2_normalized'] = BOOM_dataset_renormalized['SiO2'][i]*100/sum_
            BOOM_dataset_renormalized.loc[i,'TiO2_normalized'] = BOOM_dataset_renormalized['TiO2'][i]*100/sum_
            BOOM_dataset_renormalized.loc[i,'Al2O3_normalized'] = BOOM_dataset_renormalized['Al2O3'][i]*100/sum_
            BOOM_dataset_renormalized.loc[i,'FeO_normalized'] = BOOM_dataset_renormalized['FeO'][i]*100/sum_
            BOOM_dataset_renormalized.loc[i,'Fe2O3_normalized'] = BOOM_dataset_renormalized['Fe2O3'][i]*100/sum_
            BOOM_dataset_renormalized.loc[i,'MgO_normalized'] = BOOM_dataset_renormalized['MgO'][i]*100/sum_
            BOOM_dataset_renormalized.loc[i,'CaO_normalized'] = BOOM_dataset_renormalized['CaO'][i]*100/sum_
            BOOM_dataset_renormalized.loc[i,'Na2O_normalized'] = BOOM_dataset_renormalized['Na2O'][i]*100/sum_
            BOOM_dataset_renormalized.loc[i,'K2O_normalized'] = BOOM_dataset_renormalized['K2O'][i]*100/sum_
            BOOM_dataset_renormalized.loc[i,'Total_normalization'] = sum_                     

        if ((BOOM_dataset_renormalized['FeO'][i] != -1)&(BOOM_dataset_renormalized['Fe2O3'][i] != -1))&(BOOM_dataset_renormalized['FeOT'][i] != -1):
            sum_ = np.nansum([BOOM_dataset_renormalized['SiO2'][i],
                      BOOM_dataset_renormalized['TiO2'][i],
                      BOOM_dataset_renormalized['Al2O3'][i],
                      BOOM_dataset_renormalized['FeO'][i], 
                      BOOM_dataset_renormalized['Fe2O3'][i]*0.899, 
                      BOOM_dataset_renormalized['MgO'][i],
                      BOOM_dataset_renormalized['CaO'][i],
                      BOOM_dataset_renormalized['Na2O'][i],
                      BOOM_dataset_renormalized['K2O'][i]])
                      
            BOOM_dataset_renormalized.loc[i,'SiO2_normalized'] = BOOM_dataset_renormalized['SiO2'][i]*100/sum_
            BOOM_dataset_renormalized.loc[i,'TiO2_normalized'] = BOOM_dataset_renormalized['TiO2'][i]*100/sum_
            BOOM_dataset_renormalized.loc[i,'Al2O3_normalized'] = BOOM_dataset_renormalized['Al2O3'][i]*100/sum_
            BOOM_dataset_renormalized.loc[i,'FeO_normalized'] = BOOM_dataset_renormalized['FeO'][i]*100/sum_
            BOOM_dataset_renormalized.loc[i,'FeOT_normalized'] = BOOM_dataset_renormalized['FeOT'][i]*100/sum_
            BOOM_dataset_renormalized.loc[i,'Fe2O3_normalized'] = BOOM_dataset_renormalized['Fe2O3'][i]*100/sum_
            BOOM_dataset_renormalized.loc[i,'MgO_normalized'] = BOOM_dataset_renormalized['MgO'][i]*100/sum_
            BOOM_dataset_renormalized.loc[i,'CaO_normalized'] = BOOM_dataset_renormalized['CaO'][i]*100/sum_
            BOOM_dataset_renormalized.loc[i,'Na2O_normalized'] = BOOM_dataset_renormalized['Na2O'][i]*100/sum_
            BOOM_dataset_renormalized.loc[i,'K2O_normalized'] = BOOM_dataset_renormalized['K2O'][i]*100/sum_
            BOOM_dataset_renormalized.loc[i,'Total_normalization'] = sum_                     

        BOOM_dataset_renormalized.loc[i,'MnO + P2O5 + Cl'] = np.nansum([BOOM_dataset_renormalized['MnO'][i],
                                                              BOOM_dataset_renormalized['Cl'][i],
                                                              BOOM_dataset_renormalized['P2O5'][i]])
        BOOM_dataset_renormalized.loc[i,'Analytical Total without LOI'] = np.nansum([BOOM_dataset_renormalized['Total'][i],
                                                                           - BOOM_dataset_renormalized['LOI'][i]])
     
    return BOOM_dataset_renormalized

#--------------------- Functions for UncertaintyAndGeostandards notebook -------------------------------
def simbología_std(std):
    simbología = pd.read_csv('../assets/Data/Standards_Reference.csv', encoding = 'UTF-8', low_memory =False)
    temp = simbología.loc[simbología['StandardID'] == std]
    coloR = temp.values[0,1]
    return coloR

def estimating_accuracy(BOOM_geostandards,BOOM_geostandards_ref):
# Estimating Accuracy: Measured Average/ Certified Value for each analyzed element for each secondary standard
    MeasuredVsRef = pd.DataFrame(0, index = np.arange(len(BOOM_geostandards.Standard)) ,columns = ['MeasurementRun','Standard','SiO2','TiO2','Al2O3','MnO','MgO','Fe2O3T','FeOT','CaO','Na2O','K2O','P2O5','Cl','Rb','Sr','Y','Zr','Nb','Cs','Ba','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','Pb','Th','U'])
    elementos = ['SiO2','TiO2','Al2O3','MnO','MgO','FeOT','CaO','Na2O','K2O','P2O5','Cl','Rb','Sr','Y','Zr','Nb','Cs','Ba','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','Pb','Th','U']
    #'Fe2O3T',
    BOOM_geostandards.replace(to_replace='n.a.', value=np.nan, inplace=True)
    BOOM_geostandards.replace(to_replace='<0.01', value=np.nan, inplace=True)
    BOOM_geostandards.replace(to_replace='<0.002', value=np.nan, inplace=True)
    BOOM_geostandards.replace(to_replace='> 10000', value=np.nan, inplace=True)
    BOOM_geostandards.replace(to_replace='<5', value=np.nan, inplace=True)
    BOOM_geostandards.replace(to_replace='Over range', value=np.nan, inplace=True)
    BOOM_geostandards.loc[:, 'SiO2':'RSD_U'] = BOOM_geostandards.loc[:, 'SiO2':'RSD_U'].astype('float')
    MeasuredVsRef.loc[:, 'SiO2':'U'] = MeasuredVsRef.loc[:, 'SiO2':'U'].astype('float64')

   
    # filtering measurement runs where certified geostandards have been analyzed
    BOOM_geostandards_certified = BOOM_geostandards[BOOM_geostandards.Standard.isin(BOOM_geostandards_ref[BOOM_geostandards_ref.ErrorType.isin(["95%CL","SD"])].StandardID.unique().tolist())].copy()
    BOOM_geostandards_ref.replace(to_replace='<10', value=np.nan, inplace=True)
    BOOM_geostandards_ref.loc[:, 'SiO2':'U'] = BOOM_geostandards_ref.loc[:, 'SiO2':'U'].astype('float')
 
    
    i=0
    for run in BOOM_geostandards_certified.MeasurementRun.unique():
        #print(run)
        temp = BOOM_geostandards_certified[BOOM_geostandards_certified.MeasurementRun==run]
        for std in temp.Standard.unique():
            #print(std)
            MeasuredVsRef.loc[i,'MeasurementRun'] = run
            MeasuredVsRef.loc[i,'Standard'] = std
        
            temp1 = temp[temp.Standard == std]
            index1 = temp1.first_valid_index()
            
            temp2 = BOOM_geostandards_ref[BOOM_geostandards_ref.StandardID == std]
            index2 = temp2.first_valid_index()
        
            for elemento in elementos:
                
                if (temp1[elemento][index1] != -1) & (temp2[elemento][index2] != -1):
                    #print(type(temp1[elemento][index1]))
                    #print(type(temp2[elemento][index2]))   
                    #print(temp2[elemento][index2])             
                    MeasuredVsRef.loc[i,elemento] = temp1[elemento][index1]/temp2[elemento][index2]
                    #print(type(Standards_Color[elemento][i]))
                    #print(temp1[elemento][index1]/temp2[elemento][index2])
            i=i+1            
             
    MeasuredVsRef = MeasuredVsRef.replace(0,np.nan)
    MeasuredVsRef = MeasuredVsRef.dropna(subset = ['Standard'],axis=0)
    MeasuredVsRef.loc[:,'MeasurementRun'] = MeasuredVsRef.loc[:,'MeasurementRun'].astype('str')
    
    return MeasuredVsRef

def plot_accuracy_MeasurementRun(Accuracy_data,save=False,ymin=0.4,ymax=1.6):
# Plot the accuracy for all the elements analyzed for each Standards in each MeasurementRun

    # Here we choose which set of elements we want to analyze
    elementos = ['SiO2','TiO2','Al2O3','FeOT','MnO','MgO','CaO','Na2O','K2O','P2O5','Cl','Rb','Sr','Y','Zr','Nb','Cs','Ba','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Pb', 'Th', 'U']
    #elementos = ['SiO2','TiO2','Al2O3','MnO','MgO','FeOT','CaO','Na2O','K2O','P2O5','Cl']
    linea1 = np.empty(len(elementos))
    linea1.fill(1.05)
    linea2 = np.empty(len(elementos))
    linea2.fill(0.95)

    for run in Accuracy_data.MeasurementRun.unique():
        #print(run)
        plt.figure(figsize=(12,5))
        ax = plt.axes()        
        temp = Accuracy_data[Accuracy_data.MeasurementRun==run]
        for std in temp.Standard.unique():
            temp2 = temp[temp.Standard == std]
            temp2 = temp2.reset_index(drop=True)
            index2 = temp2.first_valid_index()
            #print(index2)
            #print(temp2)
            Color = simbología_std(std)
            plt.plot(elementos, linea1,color = 'lightgrey')
            plt.plot(elementos, linea2,color = 'lightgrey')
            plt.plot(elementos, temp2[elementos].iloc[0,:],marker = 'o',color = Color,label = std)
        
        leg=plt.legend(fancybox=True, bbox_to_anchor=(1,1),ncol=1,fontsize=14, title="Analyzed Standards")
        plt.ylim(ymin,ymax)
        ax.set_title('Measurement Run: '+ run ,fontsize=16)
        ax.tick_params(labelsize = 14,direction='in',axis='x',rotation=45)
        plt.ylabel("Measured/Certified", fontsize = 16)
        ax.grid(axis ='y')
        
        if save:
            plt.savefig('../Plots/Accuracy_'+run+'.pdf',dpi = 300,bbox_inches='tight')#,bbox_extra_artists=(leg,)
        plt.show()

def plot_accuracy_BOOM(Accuracy_data,save=False,ymin=0.4,ymax=1.6):
# Plot the accuracy for all the elements analyzed for each Standards in each MeasurementRun grafico
    # Here we choose which set of elements we want to analyze
    elementos = ['SiO2','TiO2','Al2O3','FeOT','MnO','MgO','CaO','Na2O','K2O','P2O5','Cl','Rb','Sr','Y','Zr','Nb','Cs','Ba','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Pb', 'Th', 'U']
    #elementos = ['SiO2','TiO2','Al2O3','MnO','MgO','FeOT','CaO','Na2O','K2O','P2O5']
    linea1 = np.empty(len(elementos))
    linea1.fill(1.05)
    linea2 = np.empty(len(elementos))
    linea2.fill(0.95)
    Accuracy_data = Accuracy_data.sort_values(by=['Standard'])
    plt.figure(figsize=(12,5))
    ax = plt.axes()        

    for std in Accuracy_data.Standard.unique():
        temp = Accuracy_data[Accuracy_data.Standard==std]
        #print(std)
        #print(len(temp))
        Color = simbología_std(std)
        if len(temp.SiO2) > 3:
            for elemento in elementos:
                #print(elemento)
                temp2 = temp.dropna(axis = 'rows',subset=([elemento]))
                temp2 = temp2.reset_index(drop=True)
                index2 = temp2.first_valid_index()
                if temp2[elemento].notnull().sum()>1:
                    ax.vlines(elemento,temp2[elemento].mean()-temp2[elemento].std(),temp2[elemento].mean()+temp2[elemento].std(),colors=Color,linewidth=3.5)
            ax.vlines(elemento,temp2[elemento].mean()-temp2[elemento].std(),temp2[elemento].mean()+temp2[elemento].std(),colors=Color,linewidth=3.5,label = std +' (' + str(len(temp))+' MRs)')
        
        if len(temp.SiO2) <= 3:
            plt.plot(elementos, temp[elementos].iloc[0,:],marker = 'o',linestyle='None',ms=4,color = Color,label = std)    
    
    plt.plot(elementos, linea1,color = 'grey')
    plt.plot(elementos, linea2,color = 'grey')            
    leg=plt.legend(fancybox=True, bbox_to_anchor=(1,1.1),ncol=1,fontsize=13, title="Analyzed Standards")
    plt.ylim(ymin,ymax)
    ax.tick_params(labelsize = 15,direction='in',axis='x',rotation=75)
    ax.tick_params(labelsize = 15,direction='in',axis='y')
    plt.ylabel("Analyzed/Certified: Accuracy", fontsize = 16)
    ax.grid(axis ='y')
    if save:
        plt.savefig('../Plots/AccuracyTDS.pdf',dpi = 300,bbox_inches='tight')#,bbox_extra_artists=(leg,)
    plt.show()

def plot_RSD_MeasurementRun(BOOM_geostandards,save=False,ymin=0,ymax=50):
###### Plot the presicion for all the elements analyzed for each Standards in each MeasurementRun
    #first filter the data for which n, SD and thus RSD have not been reported:
    BOOM_geostandards = BOOM_geostandards[BOOM_geostandards.n != 'Not reported']
    
    elements = [ 'SiO2', 'TiO2', 'Al2O3', 'FeOT', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'Cl', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd','Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Pb', 'Th', 'U']
#    elementos = ['SiO2','TiO2','Al2O3','MnO','MgO','FeO*','CaO','Na2O','K2O','P2O5']
    elements_RSD = ['RSD_SiO2','RSD_TiO2','RSD_Al2O3','RSD_FeOT','RSD_MnO','RSD_MgO','RSD_CaO','RSD_Na2O','RSD_K2O','RSD_P2O5','RSD_Cl','RSD_Rb','RSD_Sr','RSD_Y','RSD_Zr','RSD_Nb','RSD_Cs','RSD_Ba','RSD_La','RSD_Ce','RSD_Pr','RSD_Nd','RSD_Sm','RSD_Eu','RSD_Gd','RSD_Tb','RSD_Dy','RSD_Ho','RSD_Er','RSD_Tm','RSD_Yb','RSD_Lu','RSD_Hf','RSD_Ta','RSD_Pb','RSD_Th','RSD_U']
    
    linea1 = np.empty(len(elements))
    linea1.fill(5)
    linea2 = np.empty(len(elements))
    linea2.fill(10)

    for run in BOOM_geostandards.MeasurementRun.unique():
        plt.figure(figsize=(12,5))
        ax = plt.axes()
        temp = BOOM_geostandards[BOOM_geostandards.MeasurementRun==run]
        for std in temp.Standard.unique():
            temp2 = temp[temp.Standard == std]
            temp2 = temp2.reset_index(drop=True)
            index2 = temp2.first_valid_index()
            Color = simbología_std(std)
            
            plt.plot(elements, linea1,color = 'lightgrey')
            plt.plot(elements, linea2,color = 'lightgrey')
            plt.plot(elements, temp2[elements_RSD].iloc[0,:],marker = 'o',color = Color,label = std)
        
        leg=plt.legend(fancybox=True, bbox_to_anchor=(1,1),ncol=1,fontsize=12, title="Analyzed Standards")
        plt.ylim(0,100)
        ax.set_title('Measurement Run: '+ run ,fontsize=16)
        ax.tick_params(labelsize = 13,direction='in',axis='x',rotation=45)
        ax.tick_params(labelsize = 13,direction='in',axis='y')
        plt.ylabel("RSD (%)", fontsize = 16)
        ax.grid(axis ='y')
    
        if save:
            plt.savefig('../Plots/RSD_'+run+'.pdf',dpi = 300,bbox_inches='tight')#,bbox_extra_artists=(leg,)
        plt.show()
    
def plot_RSD_BOOM(BOOM_geostandards,save=False,ymin=0,ymax=50):

# Plot the presición for all the elements analyzed for each Standards in each MeasurementRun

    #first filter the data for which n, SD and thus RSD have not been reported:
    BOOM_geostandards = BOOM_geostandards[BOOM_geostandards.n != 'Not reported']
    
    # Here we choose which set of elements we want to analyze
    elements = ['SiO2','TiO2','Al2O3','FeOT','MnO','MgO','CaO','Na2O','K2O','P2O5','Cl',
                 'Rb','Sr','Y','Zr','Nb','Cs','Ba','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Pb', 'Th', 'U']
    elements_RSD = ['RSD_SiO2','RSD_TiO2','RSD_Al2O3','RSD_FeOT','RSD_MnO','RSD_MgO','RSD_CaO','RSD_Na2O','RSD_K2O','RSD_P2O5',
                     'RSD_Cl','RSD_Rb','RSD_Sr','RSD_Y','RSD_Zr','RSD_Nb','RSD_Cs','RSD_Ba','RSD_La','RSD_Ce','RSD_Pr','RSD_Nd','RSD_Sm','RSD_Eu','RSD_Gd','RSD_Tb','RSD_Dy','RSD_Ho','RSD_Er','RSD_Tm','RSD_Yb','RSD_Lu','RSD_Hf','RSD_Ta','RSD_Pb','RSD_Th','RSD_U']
    linea1 = np.empty(len(elements_RSD))
    linea1.fill(5)
    linea2 = np.empty(len(elements_RSD))
    linea2.fill(10)
    BOOM_geostandards = BOOM_geostandards.sort_values(by=['Standard'])
    plt.figure(figsize=(12,5)) 
    ax = plt.axes()

    for std in BOOM_geostandards.Standard.unique():
        temp = BOOM_geostandards[BOOM_geostandards.Standard==std]
        #print(std)
        #print(len(temp))
        Color = simbología_std(std)
        if len(temp.SiO2) > 3:
            for elemento in elements_RSD:
                #print(elemento)
                temp2 = temp.dropna(axis = 'rows',subset=([elemento]))
                temp2 = temp2.reset_index(drop=True)
                index2 = temp2.first_valid_index()
                if temp2[elemento].notnull().sum()>1:
                    ax.vlines(elemento,temp2[elemento].mean()-temp2[elemento].std() ,temp2[elemento].mean()+temp2[elemento].std(),colors=Color,linewidth=3.5)
            ax.vlines(elemento,temp2[elemento].mean()-temp2[elemento].std() ,temp2[elemento].mean()+temp2[elemento].std(),colors=Color,linewidth=3.5,label = std +' (' + str(len(temp))+' MRs)')        
                    
        if len(temp.SiO2) <= 3:
            plt.plot(elements_RSD, temp[elements_RSD].iloc[0,:],marker = 'o',linestyle='None',ms=4,color = Color,label = std)    
       
    plt.plot(elements_RSD, linea1,color = 'grey')
    plt.plot(elements_RSD, linea2,color = 'grey')            
    leg=plt.legend(fancybox=True, bbox_to_anchor=(1,1),ncol=1,fontsize=13, title="Analyzed Standards")
    plt.ylim(ymin,ymax)
    ax.tick_params(labelsize = 15,direction='in',axis='x',rotation=75)
    ax.tick_params(labelsize = 15,direction='in',axis='y')
    
    ax.set_xticklabels(elements)
    plt.ylabel("RSD (%)", fontsize = 16)
    ax.grid(axis ='y')
    if save:
        plt.savefig('../Plots/RSD_TDS.pdf',dpi = 300,bbox_inches='tight')#,bbox_extra_artists=(leg,)
    plt.show()

#-------------------- Functions for Correlations notebook ----------------------------------------------
def color_volcan(volcan):
    simbología = pd.read_csv('../../Scripts/Simbologia.csv',encoding = 'latin1', low_memory=False)
    temp = simbología.loc[simbología['Volcano'] == volcan]
    coloR = simbología.loc[temp.first_valid_index(),'Color']
    return coloR

def colores(Y,type):
    Dpal = {}
    if type == 'volcano':
        for i, volcan in enumerate(Y.unique()):
            color, marker = simbologia(volcan,'Unknown')
            Dpal[volcan] = color

    if type == 'event':
        simbología = pd.read_csv('../../Scripts/Simbologia.csv', encoding = 'latin1', low_memory =False)
        for event in Y.unique():
            #print(event)
            color, marker = simbologia(simbología[simbología.Event == event].Volcano.values[0],event)
            Dpal[event] = color
            
    return Dpal

def plot_map(BOOM_geodf,unknown):

    import folium

    volcanoes_by_latitude = ["Llaima",
                         "Sollipulli",
                         "Caburgua-Huelemolles-Relicura",
                         "Villarrica",
                         "Quetrupillán",
                         "Lanín",
                         "Huanquihue Group",
                         "Mocho-Choshuenco",
                         "Carrán-Los Venados",
                         "Puyehue-Cordón Caulle",
                         "Grupo Antillanca",
                         "Osorno",
                         "Calbuco",
                         "Yate",
                         'Cordón Cabrera',
                         "Apagado o Hualaihué",
                         "Hornopirén",
                         "Huequi",
                         "Michinmahuida",
                         'Subsidiary Vcha dome',
                         "Chaitén",
                         "Corcovado",
                         "Yanteles",
                         "Melimoyu",
                         "Mentolat",
                         "Cay",
                         "Macá",
                         "Hudson",
                         "Lautaro",
                         "Aguilera",
                         "Reclus",
                         "Monte Burney",
                         "Unknown"]

    base_map = unknown.explore(color = 'grey',
                           name = 'Unknown',
                           marker_type = 'marker',
                           legend = True,
                           tiles='https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.png',
                           attr= "Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL")

    df_correlation = BOOM_geodf[(BOOM_geodf['properties.Volcano']!='Unknown')]

    for volcan in df_correlation['properties.Volcano'].unique():

        temp = BOOM_geodf[BOOM_geodf['properties.Volcano']==volcan]
        temp.explore(m=base_map,
                 color = color_volcan(volcan),
                 tooltip = ['properties.Volcano','properties.Event','properties.TypeOfSection','properties.SectionID','properties.SubSectionID',
                            'properties.SampleID','properties.TypeOfRegister','properties.AnalyzedMaterial',
                            'properties.DepositThickness_cm','properties.DepositColor','properties.Flag','properties.FlagDescription'],
                 name = volcan,
                 marker_kwds = {'radius' : 4},
                 tiles='https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.png',
                 attr= "Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL")
    

    unknown.explore(m= base_map,
                color = 'grey',
                tooltip = ['properties.Volcano','properties.Event','properties.TypeOfSection','properties.SectionID','properties.SubSectionID',
                           'properties.SampleID','properties.TypeOfRegister','properties.AnalyzedMaterial',
                           'properties.DepositThickness_cm','properties.DepositColor'],
                name = 'Unknown',
                legend = True,
                 marker_kwds = {'radius' : 5},
                tiles='https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.png',
                attr= "Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL")

    folium.LayerControl().add_to(base_map)
    return base_map  

def plot_geochemistry(BOOM_geodf, unknown, element1, element2):
    import warnings
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FutureWarning)
    import plotly.express as px

    unknown_row = {'Volcan': 'Unknown', 'Latitud': unknown.centroid.map(lambda p: p.y).unique()[0]}
    
    # the following is done to sort the volcanoes by latitude in the legend, together with the unnown sample, that way is easier to compare
    #  the unknown smaples with nearby volcanic centers.  
    volcanoes_by_latitude = ["Llaima",
                         "Sollipulli",
                         "Caburgua-Huelemolles-Relicura",
                         "Villarrica",
                         "Quetrupillán",
                         "Lanín",
                         "Huanquihue Group",
                         "Mocho-Choshuenco",
                         "Carrán-Los Venados",
                         "Puyehue-Cordón Caulle",
                         "Grupo Antillanca",
                         "Osorno",
                         "Calbuco",
                         "Yate",
                         'Cordón Cabrera',
                         "Apagado o Hualaihué",
                         "Hornopirén",
                         "Huequi",
                         "Michinmahuida",
                         'Subsidiary Vcha dome',
                         "Chaitén",
                         "Corcovado",
                         "Yanteles",
                         "Melimoyu",
                         "Mentolat",
                         "Cay",
                         "Macá",
                         "Hudson",
                         "Lautaro",
                         "Aguilera",
                         "Reclus",
                         "Monte Burney",
                         "Unknown"]

    #defininf a new dataset including the unknown sample and the known samples
    df_correlation = BOOM_geodf[(BOOM_geodf['properties.Volcano']!='Unknown')]
    temp = df_correlation.dropna(axis = 'rows',subset=(['properties.SiO2']))
    temp = pd.concat([temp,unknown]) 
    temp['properties.Volcano'] = pd.Categorical(temp['properties.Volcano'],
                                                   categories=volcanoes_by_latitude,
                                                  ordered = True)
    temp.sort_values('properties.Volcano', inplace=True)

    # plot
    fig = px.scatter(temp, element1, element2 , color = 'properties.Volcano',
                  color_discrete_map = colores(temp['properties.Volcano'],type="volcano"), 
                  hover_data = ['properties.SampleID','properties.SampleObservationID',
                                'properties.Volcano','properties.Event','properties.Authors',
                                'properties.AnalyzedMaterial','properties.Flag'],
                  labels = volcanoes_by_latitude,
                  width=800, height=500)

    fig.show()

def plot_age(BOOM_geodf, unknown):

    import warnings
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FutureWarning)
    import plotly.express as px
    unknown_row = {'properties.Volcan': 'Unknown', 'Latitud': unknown.centroid.map(lambda p: p.y).unique()[0]}
    
    volcanoes_by_latitude = ["Llaima",
                         "Sollipulli",
                         "Caburgua-Huelemolles-Relicura",
                         "Villarrica",
                         "Quetrupillán",
                         "Lanín",
                         "Huanquihue Group",
                         "Mocho-Choshuenco",
                         "Carrán-Los Venados",
                         "Puyehue-Cordón Caulle",
                         "Grupo Antillanca",
                         "Osorno",
                         "Calbuco",
                         "Yate",
                         'Cordón Cabrera',
                         "Apagado o Hualaihué",
                         "Hornopirén",
                         "Huequi",
                         "Michinmahuida",
                         'Subsidiary Vcha dome',
                         "Chaitén",
                         "Corcovado",
                         "Yanteles",
                         "Melimoyu",
                         "Mentolat",
                         "Cay",
                         "Macá",
                         "Hudson",
                         "Lautaro",
                         "Aguilera",
                         "Reclus",
                         "Monte Burney",
                         "Unknown"]

    df_correlation = BOOM_geodf[(BOOM_geodf['properties.Volcano']!='Unknown')]
    temp = df_correlation.append(unknown)    
    temp['properties.Volcano'] = pd.Categorical(temp['properties.Volcano'],
                                                   categories=volcanoes_by_latitude,
                                                  ordered = True)
    temp.sort_values('properties.Volcano', inplace=True)

    fig = px.violin(temp, y="properties.RadiocarbonAge", x = 'properties.Volcano',
               color="properties.Event", color_discrete_map = colores(temp['properties.Event'],"event"), 
               violinmode ='overlay', box =True, points = 'all',
                #violinmode='overlay', # draw violins on top of each other
                # default violinmode is 'group' as in example above
                hover_data=['properties.Event','properties.SampleID','properties.AnalyzedMaterial','properties.StratigraphicPosition'])
    
    fig.update_yaxes(autorange="reversed");fig.update_xaxes(categoryorder='array',categoryarray=volcanoes_by_latitude)
    fig.show()

def unknown_info(BOOM_unknown_volcano,maxlines):

    counter = 0

    for section in BOOM_unknown_volcano['properties.SubSectionID'].unique():

        if section != 'NaN': 
            print( 'SubSection: '+ '\033[1m' + section + '\033[0m')
            temp = BOOM_unknown_volcano[BOOM_unknown_volcano['properties.SubSectionID'] == section]
            print('Samples: '+  str(temp['properties.SampleID'].unique().tolist()))
            #print( 'Samples: '+ '\033[1m' + temp['properties.SubSectionID'].unique().tolist() + '\033[0m')
            print(temp['properties.Authors'].unique()[0])
            #print('Number of sample observations: '+ str(len(temp['properties.SampleObservationID'].unique())))
            print("Number of analysis by type of register:")    
            print(temp['properties.TypeOfRegister'].value_counts())
            print(" ")
        
        else: 
            for sample in BOOM_unknown_volcano[BOOM_unknown_volcano['properties.SubSectionID'] == 'NaN']['properties.SampleID'].unique():
                print( 'Sample: '+ '\033[1m' + sample + '\033[0m')
                temp = BOOM_unknown_volcano[BOOM_unknown_volcano['properties.SampleID'] == sample]
                #print( 'Samples: '+ '\033[1m' + temp['properties.SubSectionID'].unique().tolist() + '\033[0m')
                print(temp['properties.Authors'].unique()[0])
               #print('Number of sample observations: '+ str(len(temp['properties.SampleObservationID'].unique())))
                print(temp['properties.TypeOfRegister'].value_counts())
                print(" ")

        if counter == maxlines+1:
            break
        
        counter=counter+1

#----------------------------- Functions for Machine Learning notebook ---------------------------------
def compare_accuracies(mo_file, mt_file, mo_bal_file, mt_bal_file, pdf_output=False):

    mo_accuracies_df = pd.read_csv(mo_file)
    mo_accuracies_df.drop(columns=['Unnamed: 0'],inplace=True)
    mt_accuracies_df = pd.read_csv(mt_file)
    mt_accuracies_df.drop(columns=['Unnamed: 0'],inplace=True)
    mo_bal_accuracies_df = pd.read_csv(mo_bal_file)
    mo_bal_accuracies_df.drop(columns=['Unnamed: 0'],inplace=True)
    mt_bal_accuracies_df = pd.read_csv(mt_bal_file)
    mt_bal_accuracies_df.drop(columns=['Unnamed: 0'],inplace=True)

    x = mo_accuracies_df.mean(axis=0)
    std_x = mo_accuracies_df.std(axis=0)
    max_err_x = mo_accuracies_df.max(axis=0) - mo_accuracies_df.min(axis=0)

    y = mt_accuracies_df[mo_accuracies_df.columns].mean(axis=0)
    std_y = mt_accuracies_df[mo_accuracies_df.columns].std(axis=0)
    max_err_y = mt_accuracies_df[mo_accuracies_df.columns].max(axis=0) - \
                mt_accuracies_df[mo_accuracies_df.columns].min(axis=0)

    x_bal = mo_bal_accuracies_df.mean(axis=0)
    std_x_bal = mo_bal_accuracies_df.std(axis=0)
    max_err_x_bal = mo_bal_accuracies_df.max(axis=0) - mo_bal_accuracies_df.min(axis=0)

    y_bal = mt_bal_accuracies_df[mo_bal_accuracies_df.columns].mean(axis=0)
    std_y_bal = mt_bal_accuracies_df[mo_bal_accuracies_df.columns].std(axis=0)
    max_err_y_bal = mt_bal_accuracies_df[mo_bal_accuracies_df.columns].max(axis=0) - \
                    mt_bal_accuracies_df[mo_bal_accuracies_df.columns].min(axis=0)

    fig1, ax = plt.subplots(ncols=2,sharey=True)

    colors = ['cyan', 'purple', 'orange', 'blue']
    labels = ['KNN - mean imputer', 'Logistic Regression - mean imputer', 'Random Forest - mean imputer', 'Gradient Boosting']

    ax[0].set_xlim(0.55, 0.95)
    ax[0].set_ylim(0.55, 0.95)
    ax[0].set_box_aspect(1)
    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("majors only")
    ax[0].set_ylabel("majors + traces")
    ax[0].scatter(x, y, c=colors, s=50, zorder=3)
    ax[0].errorbar(x,y,xerr=std_x,yerr=std_y,fmt='o',linewidth=2)
    ax[0].errorbar(x,y,xerr=max_err_x,yerr=max_err_y,fmt='o',linewidth=0.5)

    ax[1].set_xlim(0.55, 0.95)
    ax[1].set_ylim(0.55, 0.95)
    ax[1].set_box_aspect(1)
    ax[1].set_title("Balanced Accuracy")
    ax[1].set_xlabel("majors only")
    ax[1].scatter(x_bal, y_bal, c=colors, s=50, zorder=3)
    ax[1].errorbar(x_bal,y_bal,xerr=std_x_bal,yerr=std_y_bal,fmt='o',linewidth=2)
    ax[1].errorbar(x_bal,y_bal,xerr=max_err_x_bal,yerr=max_err_y_bal,fmt='o',linewidth=0.5)

    dots = []
    for (color, label) in zip(colors,labels):
        dot = ptc.Circle([], radius=2, color=color, label=label)
        dots.append(dot)
    legend = ax[1].legend(handles=dots, loc = (1.1,0.1), title="models")
    ax[1].add_artist(legend)

    if pdf_output:
        plt.savefig('comparison_accuracy.pdf',
                    dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()