from .metrics import sharpe_ratio, sortino_ratio, max_drawdown
import numpy as np

def compare_portfolios(nav_dict, risk_free=0.0):
    '''
    nav_dict: {name: nav_array}
    返回对比表格（dict of dict）
    '''
    result = {}
    for name, nav in nav_dict.items():
        nav = np.array(nav)
        returns = np.diff(nav) / nav[:-1]
        sr = sharpe_ratio(returns, risk_free)
        so = sortino_ratio(returns, risk_free)
        mdd = max_drawdown(nav)
        result[name] = {
            'Sharpe': sr,
            'Sortino': so,
            'MaxDrawdown': mdd
        }
    return result 