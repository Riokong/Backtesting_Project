import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class MarketSimulator:
    def __init__(self, start_price, volatility, num_minutes):
        """
        Initialize market simulator with starting parameters.
        
        Args:
            start_price (float): Initial price of the asset
            volatility (float): Daily volatility for price generation
            num_minutes (int): Number of minutes to simulate
        """
        self.start_price = start_price
        self.volatility = volatility
        self.num_minutes = num_minutes
        
    def generate_market_data(self):
        """Generate synthetic market data with minute-level timestamps."""
        # Generate timestamps
        eastern = pytz.timezone("US/Eastern")
        base_timestamp = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
        timestamps = [base_timestamp + timedelta(minutes=i) for i in range(self.num_minutes)]
        
        # Generate prices using geometric Brownian motion
        #daily_returns = np.random.normal(0, self.volatility, self.num_minutes) / np.sqrt(252 * 390)
        #Add a positive drift, mu is the annualized expected return, and sigma is the volatility.
        daily_returns = (mu - 0.5 * sigma**2) / (252 * 390) + np.random.normal(0, sigma, self.num_minutes) / np.sqrt(252 * 390)

        prices = self.start_price * np.exp(np.cumsum(daily_returns))
        
        # Generate volumes with U-shaped pattern
        time_of_day = np.array(range(self.num_minutes)) / self.num_minutes
        volume_pattern = 1 + np.sin(np.pi * time_of_day) * 0.5
        volumes = np.random.lognormal(mean=8, sigma=0.5, size=self.num_minutes) * volume_pattern
        
        # Create DataFrame
        market_data = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': volumes.astype(int)
        })
        
        return market_data

class TWAPStrategy:
    def __init__(self, order_size, num_slices):
        """
        Initialize TWAP strategy parameters.
        
        Args:
            order_size (int): Total size to execute
            num_slices (int): Number of time slices for execution
        """
        self.order_size = order_size
        self.num_slices = num_slices
        self.slice_size = order_size // num_slices
        
    def execute(self, market_data):
        """
        Execute TWAP strategy on market data.
        
        Args:
            market_data (pd.DataFrame): Market data with price and volume
        
        Returns:
            pd.DataFrame: Execution results
        """
        # Calculate time slice points
        slice_points = np.linspace(0, len(market_data)-1, self.num_slices+1).astype(int)
        
        executions = []
        remaining_shares = self.order_size
        
        for i in range(len(slice_points)-1):
            start_idx = slice_points[i]
            end_idx = slice_points[i+1]
            
            # Calculate slice metrics
            slice_data = market_data.iloc[start_idx:end_idx+1]
            vwap_price = np.average(slice_data['price'], weights=slice_data['volume'])
            
            # Add random slippage
            execution_price = vwap_price * (1 + np.random.normal(0, 0.0005))
            
            # Calculate execution size
            execution_size = min(self.slice_size, remaining_shares)
            remaining_shares -= execution_size
            
            executions.append({
                'timestamp': slice_data.iloc[0]['timestamp'],
                'size': execution_size,
                'price': execution_price,
                'vwap': vwap_price
            })
            
        return pd.DataFrame(executions)

class BacktestAnalyzer:
    @staticmethod
    def calculate_metrics(executions, market_data):
        """
        Calculate execution metrics.
        
        Args:
            executions (pd.DataFrame): Execution results
            market_data (pd.DataFrame): Market data
            
        Returns:
            dict: Dictionary containing execution metrics
        """
        # Calculate VWAP of entire period
        benchmark_vwap = np.average(market_data['price'], weights=market_data['volume'])
        
        # Calculate volume-weighted execution price
        execution_vwap = np.average(executions['price'], weights=executions['size'])
        
        # Calculate metrics
        total_executed = executions['size'].sum()
        execution_cost_bps = (execution_vwap - benchmark_vwap) / benchmark_vwap * 10000
        avg_slippage_bps = ((executions['price'] - executions['vwap']) / executions['vwap'] * 10000).mean()
        
        return {
            'benchmark_vwap': benchmark_vwap,
            'execution_vwap': execution_vwap,
            'total_executed': total_executed,
            'execution_cost_bps': execution_cost_bps,
            'avg_slippage_bps': avg_slippage_bps
        }

def run_backtest(order_size=10000, num_slices=10, start_price=100, volatility=0.2, num_minutes=390):
    """
    Run a complete backtest simulation.
    
    Args:
        order_size (int): Total size to execute
        num_slices (int): Number of TWAP slices
        start_price (float): Initial price
        volatility (float): Daily volatility
        num_minutes (int): Trading minutes to simulate
    
    Returns:
        tuple: (execution results, market data, metrics)
    """
    # Generate market data
    simulator = MarketSimulator(start_price, volatility, num_minutes)
    market_data = simulator.generate_market_data()
    
    # Execute strategy
    strategy = TWAPStrategy(order_size, num_slices)
    executions = strategy.execute(market_data)
    
    # Calculate metrics
    metrics = BacktestAnalyzer.calculate_metrics(executions, market_data)
    
    return executions, market_data, metrics

# Example usage
if __name__ == "__main__":
    # Run backtest
    executions, market_data, metrics = run_backtest()
    
    # Print results
    print("\nBacktest Results:")
    print(f"Benchmark VWAP: ${metrics['benchmark_vwap']:.4f}")
    print(f"Execution VWAP: ${metrics['execution_vwap']:.4f}")
    print(f"Total Executed: {metrics['total_executed']:,} shares")
    print(f"Execution Cost: {metrics['execution_cost_bps']:.2f} bps")
    print(f"Average Slippage: {metrics['avg_slippage_bps']:.2f} bps")
    
    # Display sample of executions
    print("\nSample Executions:")
    print(executions.head())