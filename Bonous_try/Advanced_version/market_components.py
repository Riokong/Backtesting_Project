import numpy as np
from sklearn.ensemble import RandomForestRegressor

class Asset:
    def __init__(self, name, initial_price, volatility, correlation_matrix=None):
        self.name = name
        self.price = initial_price
        self.volatility = volatility
        self.price_history = [initial_price]
        self.returns_history = []
        self.correlation_matrix = correlation_matrix

class AdaptiveStrategy:
    def __init__(self, lookback_period=30):
        self.lookback_period = lookback_period
        self.model = RandomForestRegressor(n_estimators=100)
        self.position = 0
        self.positions_history = []
        #Initial fit with dummy data 
        X = np.random.random((10, 3))
        y = np.random.random(10)
        self.model.fit(X, y)

    def calculate_features(self, price_history):
        if len(price_history) < self.lookback_period:
            return None
            
        returns = np.diff(price_history) / price_history[:-1]
        features = []
        
        sma = np.mean(price_history[-self.lookback_period:])
        volatility = np.std(returns[-self.lookback_period:])
        momentum = returns[-self.lookback_period:].sum()
        
        features.extend([sma, volatility, momentum])
        return np.array(features).reshape(1, -1)

    def update_model(self, price_history, future_returns):
        if len(price_history) < self.lookback_period:
            return
            
        X = self.calculate_features(price_history[:-1])
        if X is None or len(future_returns) != X.shape[0]:
            return
        self.model.fit(X, future_returns[:X.shape[0]])
      
    

        

    def generate_signal(self, price_history):
        features = self.calculate_features(price_history)
        if features is None:
            return 0
            
        prediction = self.model.predict(features)[0]
        
        if prediction > 0.01:
            return 1
        elif prediction < -0.01:
            return -1
        return 0