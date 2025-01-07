import numpy as np
from Advanced_version.market_components import Asset, AdaptiveStrategy
class MarketSimulator:
    def __init__(self, assets, time_steps, dt=1/252):
        self.assets = assets
        self.time_steps = time_steps
        self.dt = dt
        self.strategies = {}
        self.portfolio_value_history = []
        self.initial_portfolio_value = 1000000
        
    def add_strategy(self, asset_name, strategy):
        self.strategies[asset_name] = strategy
        
    def simulate_correlated_returns(self):
        n_assets = len(self.assets)
        correlation_matrix = np.ones((n_assets, n_assets))
        
        for i, asset_i in enumerate(self.assets):
            for j, asset_j in enumerate(self.assets):
                if i != j and asset_i.correlation_matrix is not None:
                    correlation_matrix[i][j] = asset_i.correlation_matrix.get(asset_j.name, 0)
                    correlation_matrix[j][i] = correlation_matrix[i][j]
        
        volatilities = np.array([asset.volatility for asset in self.assets])
        uncorrelated_returns = np.random.normal(0, 1, n_assets)
        cholesky_matrix = np.linalg.cholesky(correlation_matrix)
        correlated_returns = np.dot(cholesky_matrix, uncorrelated_returns)
        
        return correlated_returns * volatilities * np.sqrt(self.dt)
        
    def run_simulation(self):
        portfolio_value = self.initial_portfolio_value
        self.portfolio_value_history = [portfolio_value]
        
        for _ in range(self.time_steps):
            correlated_returns = self.simulate_correlated_returns()
            
            for i, asset in enumerate(self.assets):
                return_t = correlated_returns[i]
                new_price = asset.price * (1 + return_t)
                asset.returns_history.append(return_t)
                asset.price_history.append(new_price)
                asset.price = new_price
                
                if asset.name in self.strategies:
                    strategy = self.strategies[asset.name]
                    
                    if len(asset.returns_history) > strategy.lookback_period:
                        strategy.update_model(
                            asset.price_history,
                            asset.returns_history[-strategy.lookback_period:]
                        )
                    
                    new_position = strategy.generate_signal(asset.price_history)
                    strategy.position = new_position
                    strategy.positions_history.append(new_position)
                    
                    if len(strategy.positions_history) > 1:
                        position_return = strategy.positions_history[-2] * return_t
                        portfolio_value *= (1 + position_return)
                        
            self.portfolio_value_history.append(portfolio_value)
            
    def get_performance_metrics(self):
        portfolio_returns = np.diff(self.portfolio_value_history) / self.portfolio_value_history[:-1]
        
        sharpe_ratio = (
        np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)
        if np.std(portfolio_returns) > 0 else 0
    )
        metrics = {
            'Total Return': (self.portfolio_value_history[-1] / self.portfolio_value_history[0] - 1) * 100,
            'Annualized Return': np.mean(portfolio_returns) * 252 * 100 if len(portfolio_returns) > 0 else 0,
            'Annualized Volatility': np.std(portfolio_returns) * np.sqrt(252) * 100 if len(portfolio_returns) > 0 else 0,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': self._calculate_max_drawdown() * 100
        }
        
        return metrics
        
    def _calculate_max_drawdown(self):
        peak = self.portfolio_value_history[0]
        max_drawdown = 0
        
        for value in self.portfolio_value_history[1:]:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
            
        return max_drawdown
   


