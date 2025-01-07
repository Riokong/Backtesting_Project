import numpy as np
from Advanced_version.market_components import Asset, AdaptiveStrategy
from Advanced_version.simulator import MarketSimulator
from Advanced_version.synthetic_data import SyntheticDataGenerator


def run_simulation():
    # Define asset correlations
    spy_correlations = {'QQQ': 0.8, 'TLT': -0.2}
    qqq_correlations = {'SPY': 0.8, 'TLT': -0.3}
    tlt_correlations = {'SPY': -0.2, 'QQQ': -0.3}
    
    # Create assets
    assets = [
        Asset('SPY', 400, 0.15, spy_correlations),
        Asset('QQQ', 300, 0.20, qqq_correlations),
        Asset('TLT', 100, 0.10, tlt_correlations)
    ]
    
    # Initialize simulator
    simulator = MarketSimulator(assets, time_steps=252)
    
    # Add adaptive strategies
    for asset in assets:
        simulator.add_strategy(asset.name, AdaptiveStrategy(lookback_period=30))
    
    # Run real data simulation
    simulator.run_simulation()
    real_metrics = simulator.get_performance_metrics()
    
    # Generate and run synthetic data simulation
    historical_returns = np.array([asset.returns_history for asset in assets]).T
    synthetic_generator = SyntheticDataGenerator(historical_returns)
    synthetic_generator.train(epochs=1000)
    synthetic_data = synthetic_generator.generate_synthetic_data(252)
    
    synthetic_simulator = MarketSimulator(assets, time_steps=252)
    for asset, synthetic_returns in zip(assets, synthetic_data.T):
        asset.returns_history = list(synthetic_returns)
    synthetic_simulator.run_simulation()
    synthetic_metrics = synthetic_simulator.get_performance_metrics()
    
    return real_metrics, synthetic_metrics

if __name__ == "__main__":
    real_metrics, synthetic_metrics = run_simulation()
    
    print("\nReal Data Metrics:")
    for metric, value in real_metrics.items():
        print(f"{metric}: {value:.2f}")
        
    print("\nSynthetic Data Metrics:")
    for metric, value in synthetic_metrics.items():
        print(f"{metric}: {value:.2f}")