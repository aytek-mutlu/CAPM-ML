#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 16:11:11 2019

@author: Aytek
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
readcapm.py

Purpose:
    Read CAPM data and apply ML with normal distribution and 
    student t-distribution assumptions 


Date:
    2017/9/14

Author:
    Aytek Mutlu
"""
###########################################################
### Imports
import numpy as np
import pandas as pd
import scipy.stats as st
import scipy.optimize as opt
from lib.grad import *
import math
###########################################################
def ReadCAPM(asStocks, sIndex, sTB):
    """
    Purpose:
        Read CAPM data

    Inputs:
        asStocks    list, with list of stocks,
        sIndex      string with sp500,
        sTB         string with TB indicator

    Return value:
        df      dataframe, data
    """
    df= pd.DataFrame()
    # Get SP500 data
    df[sIndex[0]]= pd.read_csv("data/"+sIndex[0]+".csv", index_col="Date")["Adj Close"]

    # Add in TB
    df[sTB[0]]= pd.read_csv("data/"+sTB[0]+".csv", index_col="DATE", na_values=".")

    for sStock in asStocks:
        df[sStock]= pd.read_csv("data/"+sStock+".csv", index_col="Date")["Adj Close"]

    df.index.name= "Date"
    df.index= pd.to_datetime(df.index)

    # For simplicity, drop full rows with nans;
    df= df.dropna(axis=0, how='all')
    df= df.dropna(subset=sTB)

    
    return df

def LognormalDailyExcessReturn(sIndex,asStocks,df,sTB):
    """
    Purpose:
        Compute log-normal daily excess return of data

    Inputs:
        sIndex      string with sp500
        asStocks    list, with list of stocks,
        df          dataframe, data
        sTB         string with TB indicator

    Return value:
        df          dataframe, log-normal excess return data
    """   
    #apply log return formula to index and stocks
    cols= sIndex + asStocks
    df[cols]= 100*(np.log(df[cols]) - np.log(df[cols].shift(1)))
    
    #calculate daily risk-free rate from T-bills
    df[sTB[0]+'_d'] = df[sTB[0]]/(250)    
    
    for sStock in cols:    
        df[sStock+'_d'] = df[sStock] - df[sTB[0]+'_d']
    
    #drop first row of nans occured due to return calculation
    df.dropna(how='any',inplace=True)
        
    return df

    
def LnLRegr(vP, vY, mX,stud_t):
    """
    Purpose:
        Compute loglikelihood of regression model

    Inputs:
        vP      iK+1 1D-vector of parameters, with sigma and beta
        vY      iN 1D-vector of data
        mX      iN x iK matrix of regressors
        stud_t  boolean variable for Student-t distribution
        

    Return value:
        vLL     double, loglikelihood
    """
    (iN, iK)= mX.shape
    if (np.size(vP) != iK+1) & (stud_t==False):         # Check if vP is as expected
        print ("Warning: wrong size vP= ", vP)
        
    if stud_t==True:
        (dSigma, vBeta,dNu)= GetPars(vP,bStud = stud_t)
    else:
        (dSigma, vBeta)= GetPars(vP,bStud = stud_t)
    if (dSigma <= 0):
        print ("x", end="")
        return -math.inf


    vE= vY - (mX @ vBeta).reshape(iN,1)
    

    if stud_t:
        vLL = st.t.logpdf(vE,df= dNu, scale = np.sqrt(dSigma))
    else:
        vLL = st.norm.logpdf (vE , scale = np. sqrt (dSigma))


    print (".", end="")             # Give sign of life

    return vLL.flatten()

def EstimateRegr(vY, mX,robust,stud_t):
    """
    Purpose:
      Estimate the regression model

    Inputs:
      vY        iN vector of data
      mX        iN x iK matrix of regressors
      robust    boolean variable for robust sandwich form
      stud_t    boolean variable for Student-t distribution

    Return value:
      vP        iK+1 (iK+2 in case of Student-t distribution) vector of optimal parameters sigma and beta's
      vS        iK+1 (iK+2 in case of Student-t distribution) vector of standard deviations
      dLL       double, loglikelihood
      sMess     string, output of optimization
    """
    
    (iN, iK)= mX.shape
    if stud_t==True:
        vP0= np.ones(iK+2)       
    else:
        vP0= np.ones(iK+1)  
        
    vLL= LnLRegr(vP0, vY, mX,stud_t=stud_t)

    # Create lambda function returning NEGATIVE AVERAGE LL, as function of vP only
    AvgNLnLRegr = lambda vP: -np. mean(LnLRegr(vP, vY, mX,stud_t=stud_t))
    


    res= opt.minimize(AvgNLnLRegr, vP0, args=(), method="BFGS")

    vP= res.x
    sMess= res.message
    dLL= -res.fun

    mH= -hessian_2sided(AvgNLnLRegr, vP)    
    mHI = np.linalg.inv(mH)/iN
     
    if robust == True:
        mG= jacobian_2sided(LnLRegr ,vP,vY,mX,stud_t)
        mS2 = mHI @ (mG.T@mG) @ mHI
    else:
        mS2 = -mHI
        
    vS= np.sqrt(np.diag(mS2))    
    
    print ("\nML RESUlTS:\nRobust Sandwich Form: ", robust, "\nStudent t-distribution: ", stud_t,
           "\n\nInitial LL= ", np.mean(vLL), "\nvP0=", vP0,"\n\nBFGS results in ", sMess,"\nParameters: ", vP, "\nOptimized LL: ", dLL,"\nStandard Deviations: ", vS,"\nVariance-Covariance Matrix: ", mS2,
           "\nf-eval= ", res.nfev)

    
    return (vP, vS, dLL, mS2,sMess)
    
def GetPars(vP,bStud):
    """
    Purpose:
      Read out the parameters from the vector

    Inputs:
      vP        iK+1 vector with sigma and beta's
      bStud     boolean variable for Student-t distribution

    Return value:
      dS        double, sigma
      vBeta     iK vector, beta's
      dNu       degrees of freedom for Student-t distribution
    """
    iK= np.size(vP)-1
    vP= vP.reshape(iK+1,)
    dS= vP[0]   
    vBeta= vP[1:3]
    if bStud==True:
        dNu = vP[3:]
        return (dS, vBeta,dNu)
    else:
        return (dS, vBeta)

###########################################################
### main
def main():
    # Magic numbers
    asStocks= ["MSFT"]
    sIndex= ["^GSPC"]
    sTB= ["DTB3"]      # Stocks, index and TB


    # Initialisation
    df= ReadCAPM(asStocks, sIndex, sTB)
    
    df_daily= LognormalDailyExcessReturn(sIndex,asStocks,df,sTB)    
    
    # mX is 
    mX = np.hstack([np.ones((len(df_daily['^GSPC_d']), 1)), np.array(df_daily['^GSPC_d']).reshape(len(df_daily['MSFT_d']),1)])
    
    vY=np.array(df_daily['MSFT_d']).reshape(len(df_daily['MSFT_d']),1)


    iSeed= 1234
    np.random.seed(iSeed)
    
    results = dict()
    results['normal,non_robust'] = EstimateRegr(vY, mX,robust=False,stud_t=False)
    results['normal,robust'] = EstimateRegr(vY, mX,robust=True,stud_t=False)
    results['stud_t,non_robust'] = EstimateRegr(vY, mX,robust=False,stud_t=True)
    results['stud_t,robust'] = EstimateRegr(vY, mX,robust=True,stud_t=True)
    
            
###########################################################
### start main
if __name__ == "__main__":
    main()
