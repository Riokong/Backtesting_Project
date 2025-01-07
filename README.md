1. Basic Version
   In the Basic version, I Simulated trade execution using a TWAP strategy.
   Generate synthetic data for prices, volumes, and timestamps.
   Calculate and output the following metrics:
   Execution Cost: Difference between executed price an benchmark price (e.g., VWAP).
   Slippage: Difference between expected execution price and actualprice.
   
1.1 In the Basic_version, There are mainly four files, Backtesting_Proj.pdf, backtest.py, and Implementation_Report.pdf, Report of my Advanced Version of Backtesting.pdf
1.2 THe Backtesting_Proj  is a report intends to outline my proposed back-testing framework for Smart Order Routing (SOR). 
1.3 backtest.py is the python code to evaluate the performance of a Time-Weighted Average Price (TWAP) strategy for executing an order of 10,000 shares over a trading session. 
1.4 Implementation_Report is a concise explanation of my implementation approach in the backtest.py and the results of my backtesting simulation of the basic version.

3. Advanced Version
   The advanced version of the backtesting framework integrates the adaptive trading strategies, synthetic data generation, and multi-asset marketsimulation.
   It models realistic market behavior using correlated returns and evaluates portfolio performance with metrics like Total Return, Sharpe Ratio, and Max Drawdown.
   By combining real and GAN-generated synthetic data, the framework aims to stress-test strategies under diverse market conditions.

   the code files organize in the following datafile structure:
   run.py, an empty _init_.py and setup.py are above the folder Advance_version,
   in the advanced_version folder, there are models.py, synthetic_data.py, market_components.py, simulator.py, main.py and another empty _init_.py

   In this advanced version: I have implemented an advanced simulation framework for financial trading, with the following key components:

3.1 Market Simulation: 
Simulates price movements and correlations among multiple assets using a Cholesky decomposition for correlated returns.
3.2 Adaptive Strategies:
Incorporates machine learning (Random Forest) for predictive modeling and adaptive position sizing based on historical data.
3.3 Synthetic Data Generation: 
Uses a Generative Adversarial Network (GAN) to create synthetic financial data, enabling simulations with both real and synthetic datasets.
3.4 Evaluation Metrics:
Measures performance using key metrics like total return, annualized return, Sharpe ratio, and maximum drawdown.

However, Currently, the synthetic data results (all metrics at 0.00) highlight challenges in generating realistic data or effectively integrating it into the
simulation, limiting the ability to test strategies beyond historical scenarios. Improvements to the project are ongoing.
   
