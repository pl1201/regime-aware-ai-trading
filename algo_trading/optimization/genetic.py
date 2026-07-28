from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import random
import numpy as np
import pandas as pd

from algo_trading.backtest.vectorized import run_backtest as run_vector_backtest, BacktestConfig, RiskConfig
from algo_trading.backtest.event_driven import run_event_backtest, EventConfig


# --------------------------
# Tham số & sampling
# --------------------------
# Định dạng param_space cho GA:
# {
#   'fast': {'type':'int','low':5,'high':50},
#   'slow': {'type':'int','low':20,'high':200},
#   'ma_type': {'type':'choice','values':['sma','ema']},
#   'k': {'type':'float','low':1.0,'high':3.0}
# }


def _sample_one(spec: Dict[str, Any]) -> Any:
    t = spec.get('type')
    if t == 'int':
        return random.randint(int(spec['low']), int(spec['high']))
    if t == 'float':
        return random.uniform(float(spec['low']), float(spec['high']))
    if t == 'choice':
        return random.choice(list(spec['values']))
    raise ValueError(f"Unsupported spec: {spec}")


def sample_params(space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return {k: _sample_one(v) for k, v in space.items()}


def crossover(p1: Dict[str, Any], p2: Dict[str, Any], cx_prob: float = 0.5) -> Dict[str, Any]:
    child = {}
    for k in p1.keys():
        child[k] = p1[k] if random.random() < cx_prob else p2[k]
    return child


def mutate(params: Dict[str, Any], space: Dict[str, Dict[str, Any]], mut_rate: float = 0.2) -> Dict[str, Any]:
    child = params.copy()
    for k, spec in space.items():
        if random.random() < mut_rate:
            child[k] = _sample_one(spec)
    return child


# --------------------------
# Đánh giá
# --------------------------

def _evaluate(df: pd.DataFrame, strategy_cls, params: Dict[str, Any],
              mode: str = 'vectorized', backtest_kwargs: Optional[Dict[str, Any]] = None,
              risk: Optional[RiskConfig] = None, metric: str = 'Sharpe') -> Tuple[float, Dict[str, Any]]:
    backtest_kwargs = backtest_kwargs or {}
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
            raise ValueError("mode phải là 'vectorized' hoặc 'event'")
        score = res['summary'].get(metric, -np.inf)
        if score is None or (isinstance(score, float) and (np.isnan(score))):
            score = -np.inf
        return float(score), res
    except Exception as e:
        return -np.inf, {'error': str(e), 'params': params}


# --------------------------
# GA
# --------------------------

def genetic_search(
    df: pd.DataFrame,
    strategy_cls,
    param_space: Dict[str, Dict[str, Any]],
    pop_size: int = 30,
    generations: int = 20,
    elite: int = 2,
    tournament_k: int = 3,
    mode: str = 'vectorized',
    backtest_kwargs: Optional[Dict[str, Any]] = None,
    risk: Optional[RiskConfig] = None,
    metric: str = 'Sharpe',
    maximize: bool = True,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
    backtest_kwargs = backtest_kwargs or {}

    # khởi tạo quần thể
    population = [sample_params(param_space) for _ in range(pop_size)]
    history_rows: List[Dict[str, Any]] = []
    best_score = -np.inf
    best_params = None
    best_res = None

    def tournament_select(scores: List[float], k: int) -> int:
        idxs = np.random.choice(len(scores), size=min(k, len(scores)), replace=False)
        if maximize:
            return int(idxs[np.argmax([scores[i] for i in idxs])])
        else:
            return int(idxs[np.argmin([scores[i] for i in idxs])])

    for gen in range(generations):
        # đánh giá
        scored: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
        for p in population:
            score, res = _evaluate(df, strategy_cls, p, mode, backtest_kwargs, risk, metric)
            scored.append((score, p, res))
            history_rows.append({'gen': gen, **p, 'score': score})
        # sắp xếp
        scored.sort(key=lambda x: x[0], reverse=maximize)
        if (maximize and scored[0][0] > best_score) or ((not maximize) and (best_params is None or scored[0][0] < best_score)):
            best_score, best_params, best_res = scored[0]
        # tạo thế hệ mới
        new_population: List[Dict[str, Any]] = [sp for _, sp, _ in scored[:elite]]  # elitism
        scores_only = [s for s, _, _ in scored]
        while len(new_population) < pop_size:
            i = tournament_select(scores_only, tournament_k)
            j = tournament_select(scores_only, tournament_k)
            p1 = scored[i][1]
            p2 = scored[j][1]
            child = crossover(p1, p2, cx_prob=0.5)
            child = mutate(child, param_space, mut_rate=0.2)
            new_population.append(child)
        population = new_population

    results_df = pd.DataFrame(history_rows).sort_values(['gen','score'], ascending=[True, not maximize])
    return {
        'history': results_df,
        'best_params': best_params,
        'best_score': best_score,
        'best_result': best_res,
        'metric': metric,
        'mode': mode,
    }

