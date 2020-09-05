import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import scipy.optimize as sco 

# ---- IMPORT DATA FROM CSV (NO QUANDL API) 
#data pulled from aYhFinanc
rbsc = pd.read_csv('/Users/phuongd/Desktop/BRSC.L (1).csv')
rbsc['Date'] = pd.to_datetime(rbsc['Date'])
smt = pd.read_csv('/Users/phuongd/Desktop/SMT.L.csv')
smt['Date'] = pd.to_datetime(smt['Date'])
sls = pd.read_csv('/Users/phuongd/Desktop/SLS.L.csv')
sls['Date'] = pd.to_datetime(sls['Date'])

data = pd.DataFrame() #create new dataframe which concerns only close price 
data['rbsc'] = rbsc['Close']
data['smt'] = smt['Close'] 
data['sls'] = sls['Close']
data['date'] = smt['Date']
data = data.set_index('date', drop=True)  #set date as index for better processing 
print(data)

def portoflio_annualised_performance(weights,mean_returns, cov_maxtrix): #this function returns annualised performance summary based on historical data 
    returns = np.sum(mean_returns * weights) * 252 #sum of mean log returns * weightage or portfolio * # of days in a year
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252) #standard deivation of annualised returns 
    return returns, std 

def random_portoflios(num_portfolios, mean_returns, cov_matrix, risk_free_rate): 
    results = np.zeros((3, num_portfolios)) #return zero-filled data to store 
    weights_record = [] #empty list to store weights data 
    for i in range(num_portfolios): 
        weights = np.random.random(3) #generate random weight (max sum is 1)
        weights /= np.sum(weights) #conditional that indv weight cannot be equal to 1 
        weights_record.append(weights) #append list 
        portfolio_returns, portfolio_std = portoflio_annualised_performance(weights, mean_returns, cov_matrix) #declaring portfolio returns and portfolio std
        results[0,i] = portfolio_std #append results, std or volatility 
        results[1,i] = portfolio_returns #append result, returns 
        results[2,i] = (portfolio_returns - risk_free_rate) / portfolio_std #sharpe ratio 
    return results, weights_record

returns = np.log(data/data.shift(1)) 
mean_returns = returns.mean() 
cov_matrix = returns.cov() 
num_portfolios = 75000
risk_free_rate = 0.0027800000

def display_ef(mean_returns, cov_matrix, num_portfolios, risk_free_rate): 
    results, weights = random_portoflios(num_portfolios, mean_returns, cov_matrix, risk_free_rate) 

    #getting information of maximised sharpe ratio datapoint 
    max_sharpe_idx = np.argmax(results[2])
    sd, r = results[0, max_sharpe_idx], results[1, max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx], index=data.columns, columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i, 5) for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T 

    min_vol_idx = np.argmin(results[0]) #minimum volatility 
    sd_min, r_min = results[0, min_vol_idx], results[1, min_vol_idx] 
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx], index=data.columns, columns=['allocation'])
    min_vol_allocation.allocation = [round(i, 5) for i in min_vol_allocation.allocation] 
    min_vol_allocation = min_vol_allocation.T 

    print('-'*80) 
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualised Return:", r)
    print("Annualised Volatility:", sd)
    print('\n') 
    print(max_sharpe_allocation) 

    print('-'*80) 
    print('Minimum Volatility Ratio Portfolio Allocation\n') 
    print("Annualised Return:", r_min) 
    print("Annualised Volatility:", sd_min) 
    print('\n') 
    print(min_vol_allocation) 

    sns.set() 
    plt.figure(figsize = (10,7)) 
    plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='viridis', marker='o', s=19, alpha=0.2) 
    plt.colorbar() 
    plt.scatter(sd, r, marker='*', color='r', s=500, label='Maximum Sharpe Ratio') 
    plt.scatter(sd_min, r_min, marker='*', color='g', s=500, label='Minimum Volatility') 
    plt.title('Simulated Portoflio Optimisation based on Efficient Frontier') 
    plt.xlabel('Annualised Volatility') 
    plt.ylabel('Annualised Returns') 
    plt.legend(labelspacing = 0.8) 
    plt.show()

display_ef(mean_returns, cov_matrix, num_portfolios, risk_free_rate)