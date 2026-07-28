from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import itertools
import numpy as np
import pandas as pd

from algo_trading.backtest.vectorized import run_backtest as run_vector_backtest, BacktestConfig, RiskConfig
from algo_trading.backtest.event_driven import run_event_backtest, EventConfig


def _evaluate(df: pd.DataFrame, strategy_cls, params: Dict[str, Any],
              mode: str = 'vectorized', backtest_kwargs: Optional[Dict[str, Any]] = None,
              risk: Optional[RiskConfig] = None, metric: str = 'Sharpe') -> Tuple[float, Dict[str, Any]]:
    backtest_kwargs = backtest_kwargs or {}
    try:
        strat = strategy_cls(**params)
        res_sig = strat.generate_signals(df)
        signals = res_sig.signals
        if mode == 'vectorized':
            cfg = backtest_kwargs.get('cfg') or BacktestConfig(**backtest_kwargs.get('cfg_kwargs', {}))
            res = run_vector_backtest(df, signals, cfg=cfg, risk=risk)
        elif mode == 'event':
            cfg = backtest_kwargs.get('cfg') or EventConfig(**backtest_kwargs.get('cfg_kwargs', {}))
            res = run_event_backtest(df, signals, cfg=cfg, risk=risk)
        else:
            raise ValueError("mode phải là 'vectorized' hoặc 'event'")
        score = res['summary'].get(metric, np.nan)
        if score is None or np.isnan(score):
            score = -np.inf
        return float(score), res
    except Exception as e:
        return -np.inf, {'error': str(e), 'params': params}


def param_product(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combos = []
    for vals in itertools.product(*values):
        combos.append({k: v for k, v in zip(keys, vals)})
    return combos


def grid_search(df: pd.DataFrame, strategy_cls, param_grid: Dict[str, List[Any]],
                mode: str = 'vectorized', backtest_kwargs: Optional[Dict[str, Any]] = None,
                risk: Optional[RiskConfig] = None, metric: str = 'Sharpe', maximize: bool = True) -> Dict[str, Any]:
    rows = []
    best_score = -np.inf if maximize else np.inf
    best_params = None
    best_res = None
    for params in param_product(param_grid):
        score, res = _evaluate(df, strategy_cls, params, mode, backtest_kwargs, risk, metric)
        rows.append({**params, 'score': score})
        if maximize:
            if score > best_score:
                best_score, best_params, best_res = score, params, res
        else:
            if score < best_score:
                best_score, best_params, best_res = score, params, res
    results_df = pd.DataFrame(rows).sort_values('score', ascending=not maximize)
    return {
        'results': results_df,
        'best_params': best_params,
        'best_score': best_score,
        'best_result': best_res,
        'metric': metric,
        'mode': mode,
    }

