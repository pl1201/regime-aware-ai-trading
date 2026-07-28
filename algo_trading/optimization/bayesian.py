from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd

from algo_trading.backtest.vectorized import run_backtest as run_vector_backtest, BacktestConfig, RiskConfig
from algo_trading.backtest.event_driven import run_event_backtest, EventConfig


# Không bắt buộc cài đặt. Nếu không có skopt -> fallback sang random search.
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    SKOPT_AVAILABLE = True
except Exception:
    SKOPT_AVAILABLE = False


# Mô tả không gian tham số giống genetic.param_space
# {
#   'fast': {'type':'int','low':5,'high':50},
#   'slow': {'type':'int','low':20,'high':200},
#   'ma_type': {'type':'choice','values':['sma','ema']},
#   'k': {'type':'float','low':1.0,'high':3.0}
# }

def _to_skopt_space(space: Dict[str, Dict[str, Any]]):
    dims = []
    keys = []
    for k, spec in space.items():
        t = spec.get('type')
        if t == 'int':
            dims.append(Integer(int(spec['low']), int(spec['high']), name=k))
        elif t == 'float':
            prior = spec.get('prior', 'uniform')  # 'uniform' hoặc 'log-uniform'
            dims.append(Real(float(spec['low']), float(spec['high']), name=k, prior=prior))
        elif t == 'choice':
            dims.append(Categorical(list(spec['values']), name=k))
        else:
            raise ValueError(f"Loại tham số không hỗ trợ: {t}")
        keys.append(k)
    return dims, keys


def _sample_one(spec: Dict[str, Any]) -> Any:
    t = spec.get('type')
    if t == 'int':
        return np.random.randint(int(spec['low']), int(spec['high'])+1)
    if t == 'float':
        low, high = float(spec['low']), float(spec['high'])
        if spec.get('prior') == 'log-uniform':
            # sample log-uniform
            u = np.random.uniform(np.log(low+1e-12), np.log(high+1e-12))
            return float(np.exp(u))
        else:
            return float(np.random.uniform(low, high))
    if t == 'choice':
        vals = list(spec['values'])
        return vals[int(np.random.randint(0, len(vals)))]
    raise ValueError(f"Unsupported spec: {spec}")


def _random_search(df: pd.DataFrame, strategy_cls, space: Dict[str, Dict[str, Any]], n_calls: int,
                   mode: str, backtest_kwargs: Optional[Dict[str, Any]], risk: Optional[RiskConfig],
                   metric: str, maximize: bool):
    rows = []
    best_score = -np.inf if maximize else np.inf
    best_params = None
    best_res = None
    for _ in range(n_calls):
        params = {k: _sample_one(spec) for k, spec in space.items()}
        try:
            strat = strategy_cls(**params)
            sig = strat.generate_signals(df).signals
            if mode == 'vectorized':
                cfg = backtest_kwargs.get('cfg') or BacktestConfig(**backtest_kwargs.get('cfg_kwargs', {}))
                res = run_vector_backtest(df, sig, cfg=cfg, risk=risk)
            else:
                cfg = backtest_kwargs.get('cfg') or EventConfig(**backtest_kwargs.get('cfg_kwargs', {}))
                res = run_event_backtest(df, sig, cfg=cfg, risk=risk)
            score = res['summary'].get(metric, -np.inf)
        except Exception as e:
            res = {'error': str(e)}
            score = -np.inf
        rows.append({**params, 'score': score})
        if maximize:
            if score > best_score:
                best_score, best_params, best_res = score, params, res
        else:
            if score < best_score:
                best_score, best_params, best_res = score, params, res
    return {
        'history': pd.DataFrame(rows).sort_values('score', ascending=not maximize),
        'best_params': best_params,
        'best_score': best_score,
        'best_result': best_res,
        'metric': metric,
        'mode': mode,
        'optimizer': 'random_fallback',
    }


def bayesian_optimize(
    df: pd.DataFrame,
    strategy_cls,
    space: Dict[str, Dict[str, Any]],
    n_calls: int = 50,
    n_initial_points: int = 10,
    acq_func: str = 'EI',  # Expected Improvement
    mode: str = 'vectorized',
    backtest_kwargs: Optional[Dict[str, Any]] = None,
    risk: Optional[RiskConfig] = None,
    metric: str = 'Sharpe',
    maximize: bool = True,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Bayesian Optimization sử dụng skopt nếu có, nếu không fallback sang random search.
    """
    backtest_kwargs = backtest_kwargs or {}
    if random_state is not None:
        np.random.seed(random_state)

    if not SKOPT_AVAILABLE:
        return _random_search(df, strategy_cls, space, n_calls, mode, backtest_kwargs, risk, metric, maximize)

    dims, keys = _to_skopt_space(space)

    def objective(x_list):
        params = {k: v for k, v in zip(keys, x_list)}
        try:
            strat = strategy_cls(**params)
            sig = strat.generate_signals(df).signals
            if mode == 'vectorized':
                cfg = backtest_kwargs.get('cfg') or BacktestConfig(**backtest_kwargs.get('cfg_kwargs', {}))
                res = run_vector_backtest(df, sig, cfg=cfg, risk=risk)
            elif mode == 'event':
                cfg = backtest_kwargs.get('cfg') or EventConfig(**backtest_kwargs.get('cfg_kwargs', {}))
                res = run_event_backtest(df, sig, cfg=cfg, risk=risk)
            else:
                return 1e9  # invalid
            score = res['summary'].get(metric, -np.inf)
            if score is None or np.isnan(score):
                score = -np.inf
        except Exception:
            score = -np.inf
        # skopt minimization -> trả về -score để tối đa hóa
        return -float(score) if maximize else float(score)

    result = gp_minimize(
        objective,
        dimensions=dims,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        acq_func=acq_func,
        random_state=random_state,
        verbose=False,
    )

    rows = []
    for x, y in zip(result.x_iters, result.func_vals):
        params = {k: v for k, v in zip(keys, x)}
        # objective trả về -metric nếu maximize=True, ngược lại trả về metric
        score = -y if maximize else y
        rows.append({**params, 'score': score})
    hist = pd.DataFrame(rows).sort_values('score', ascending=not maximize)
    best_params = {k: v for k, v in zip(keys, result.x)}

    # chạy lại với best_params để lấy kết quả đầy đủ
    strat = strategy_cls(**best_params)
    sig = strat.generate_signals(df).signals
    if mode == 'vectorized':
        cfg = backtest_kwargs.get('cfg') or BacktestConfig(**backtest_kwargs.get('cfg_kwargs', {}))
        best_res = run_vector_backtest(df, sig, cfg=cfg, risk=risk)
    else:
        cfg = backtest_kwargs.get('cfg') or EventConfig(**backtest_kwargs.get('cfg_kwargs', {}))
        best_res = run_event_backtest(df, sig, cfg=cfg, risk=risk)

    return {
        'history': hist,
        'best_params': best_params,
        'best_score': best_res['summary'].get(metric),
        'best_result': best_res,
        'metric': metric,
        'mode': mode,
        'optimizer': 'skopt_gp_minimize',
    }

