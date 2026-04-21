import numpy as np

def sharpe_ratio(returns, risk_free=0.0):
    excess = returns - risk_free
    return np.mean(excess) / (np.std(excess) + 1e-8) * np.sqrt(252)

def sortino_ratio(returns, risk_free=0.0):
    downside = returns[returns < risk_free]
    downside_std = np.std(downside) if len(downside) > 0 else 1e-8
    excess = returns - risk_free
    return np.mean(excess) / (downside_std + 1e-8) * np.sqrt(252)

def max_drawdown(nav):
    nav = np.array(nav)
    peak = np.maximum.accumulate(nav)
    drawdown = (nav - peak) / (peak + 1e-8)
    return np.min(drawdown) 