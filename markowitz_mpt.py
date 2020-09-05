import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import quandl
import scipy.optimize as sco 
import yfinance as yf

# --- PART 1: get and clean data 
quandl.ApiConfig.api_key = 'R-u__oVA8a6YMzCajW3M'
tickers = ['AAPL', 'TSLA', 'MSFT', 'AMZN']
data = quandl.get_table('WIKI/PRICES', ticker = tickers, qopts = {'columns': ['date', 'ticker', 'adj_close']}, date = {'gte': '2018-1-1', 'lte':'2020-7-8'}, paginate=True)

df = data.set_index('date') 
table = df.pivot(columns='ticker') 
table.columns = [col[1] for col in table.columns]
print(table.head())

# --- PART 2: random portfolio generation 
#calculate yearly returns and volatility 
def portoflio_annualised_performance(weights,mean_returns, cov_maxtrix): 
    returns = np.sum(mean_returns * weights) * 252 
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252) 
    return returns, std 

def random_portoflios(num_portfolios, mean_returns, cov_matrix, risk_free_rate): 
    results = np.zeros((3, num_portfolios)) 
    weights_record = [] 
    for i in range(num_portfolios): 
        weights = np.random.random(4) 
        weights /= np.sum(weights) 
        weights_record.append(weights) 
        portfolio_returns, portfolio_std = portoflio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std 
        results[1,i] = portfolio_returns 
        results[2,i] = (portfolio_returns - risk_free_rate) / portfolio_std #sharpe ratio 
    return results, weights_record

returns = np.log(table/table.shift(1)) 
mean_returns = returns.mean() 
cov_matrix = returns.cov() 
num_portfolios = 50000
risk_free_rate = 0.0027800000

def display_ef(mean_returns, cov_matrix, num_portfolios, risk_free_rate): 
    results, weights = random_portoflios(num_portfolios, mean_returns, cov_matrix, risk_free_rate) 

    #getting information of maximised sharpe ratio datapoint 
    max_sharpe_idx = np.argmax(results[2])
    sd, r = results[0, max_sharpe_idx], results[1, max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx], index=table.columns, columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100, 2) for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T 

    min_vol_idx = np.argmin(results[0]) #minimum volatility 
    sd_min, r_min = results[0, min_vol_idx], results[1, min_vol_idx] 
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx], index=table.columns, columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100, 2) for i in min_vol_allocation.allocation] 
    min_vol_allocation = min_vol_allocation.T 

    print('-'*80) 
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualised Return:", round(r,5))
    print("Annualised Volatility:", round(sd, 5))
    print('\n') 
    print(max_sharpe_allocation) 

    print('-'*80) 
    print('Minimum Volatility Ratio Portfolio Allocation\n') 
    print("Annualised Return:", round(r_min,5)) 
    print("Annualised Volatility:", round(sd_min, 5)) 
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