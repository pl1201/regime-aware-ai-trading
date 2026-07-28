from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from .grid_search import grid_search
from .genetic import genetic_search
from .bayesian import bayesian_optimize
from algo_trading.backtest.vectorized import run_backtest as run_vector_backtest, BacktestConfig, RiskConfig
from algo_trading.backtest.event_driven import run_event_backtest, EventConfig


def make_walk_forward_splits(index: pd.DatetimeIndex,
                             train_size: int,
                             test_size: int,
                             step: Optional[int] = None,
                             expanding: bool = False) -> List[Tuple[slice, slice]]:
    """
    Trả về danh sách các (train_slice, test_slice) theo chỉ số integer của index.
    - train_size/test_size: số lượng mẫu cho train/test mỗi split
    - step: nếu None -> step = test_size
    - expanding: nếu True, train mở rộng từ đầu; nếu False, dùng cửa sổ trượt cố định train_size
    """
    n = len(index)
    step = test_size if step is None else int(step)
    splits: List[Tuple[slice, slice]] = []
    start_train = 0
    while True:
        end_train = max(train_size, (len(splits)+1)*step + train_size) if expanding else start_train + train_size
        start_test = end_train
        end_test = start_test + test_size
        if expanding:
            # mở rộng train từ 0 tới end_train
            tr_slice = slice(0, min(end_train, n))
        else:
            tr_slice = slice(start_train, min(end_train, n))
        te_slice = slice(start_test, min(end_test, n))
        # dừng nếu test rỗng hoặc train chưa đủ
        if te_slice.start >= n or te_slice.start >= te_slice.stop:
            break
        if tr_slice.start >= tr_slice.stop:
            break
        splits.append((tr_slice, te_slice))
        if end_test >= n:
            break
        if expanding:
            # chỉ tăng theo step
            start_train = 0
        else:
            start_train += step
    return splits


def walk_forward_optimize(
    df: pd.DataFrame,
    strategy_cls,
    method: str = 'grid',  # 'grid' | 'genetic' | 'bayesian'
    param_grid: Optional[Dict[str, List[Any]]] = None,   # cho grid
    param_space: Optional[Dict[str, Dict[str, Any]]] = None,  # cho genetic/bayesian
    train_size: int = 1000,
    test_size: int = 250,
    step: Optional[int] = None,
    expanding: bool = False,
    mode: str = 'vectorized',
    backtest_kwargs: Optional[Dict[str, Any]] = None,
    risk: Optional[RiskConfig] = None,
    metric: str = 'Sharpe',
    maximize: bool = True,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Thực hiện walk-forward optimization:
    - Với mỗi split (train/test), tối ưu tham số trên train theo 'method'
    - Chạy backtest trên test với tham số tốt nhất và ghép returns lại thành một chuỗi tổng
    Trả về:
    {
      'splits': [ {split_idx, train_range, test_range, best_params, train_score, test_score}, ...],
      'combined_returns': Series,
      'combined_equity': Series,
      'metric': metric,
      'method': method,
      'mode': mode
    }
    """
    backtest_kwargs = backtest_kwargs or {}

    idx = df.index
    splits = make_walk_forward_splits(idx, train_size, test_size, step, expanding)
    if len(splits) == 0:
        raise ValueError("Không tạo được splits walk-forward (kiểm tra train_size/test_size/step)")

    combined_returns = pd.Series(0.0, index=idx)
    split_rows: List[Dict[str, Any]] = []

    for k, (tr_slice, te_slice) in enumerate(splits):
        df_train = df.iloc[tr_slice]
        df_test = df.iloc[te_slice]
        # tối ưu trên train
        if method == 'grid':
            if param_grid is None:
                raise ValueError("Cần param_grid cho method='grid'")
            opt_res = grid_search(df_train, strategy_cls, param_grid, mode=mode, backtest_kwargs=backtest_kwargs, risk=risk, metric=metric, maximize=maximize)
        elif method == 'genetic':
            if param_space is None:
                raise ValueError("Cần param_space cho method='genetic'")
            opt_res = genetic_search(df_train, strategy_cls, param_space, mode=mode, backtest_kwargs=backtest_kwargs, risk=risk, metric=metric, maximize=maximize, random_state=random_state)
        elif method == 'bayesian':
            if param_space is None:
                raise ValueError("Cần param_space cho method='bayesian'")
            opt_res = bayesian_optimize(df_train, strategy_cls, param_space, mode=mode, backtest_kwargs=backtest_kwargs, risk=risk, metric=metric, maximize=maximize, random_state=random_state)
        else:
            raise ValueError("method không hợp lệ: grid/genetic/bayesian")

        best_params = opt_res['best_params']
        train_best_score = opt_res['best_score']

        # chạy test với params tối ưu
        strat = strategy_cls(**best_params)
        sig_test = strat.generate_signals(df_test).signals
        if mode == 'vectorized':
            cfg = backtest_kwargs.get('cfg') or BacktestConfig(**backtest_kwargs.get('cfg_kwargs', {}))
            res_test = run_vector_backtest(df_test, sig_test, cfg=cfg, risk=risk)
        else:
            cfg = backtest_kwargs.get('cfg') or EventConfig(**backtest_kwargs.get('cfg_kwargs', {}))
            res_test = run_event_backtest(df_test, sig_test, cfg=cfg, risk=risk)
        test_score = res_test['summary'].get(metric, np.nan)

        # ghép returns
        combined_returns.iloc[te_slice] = res_test['returns'].reindex(df_test.index).fillna(0.0)

        split_rows.append({
            'split_idx': k,
            'train_start': df_train.index[0],
            'train_end': df_train.index[-1],
            'test_start': df_test.index[0],
            'test_end': df_test.index[-1],
            'best_params': best_params,
            'train_score': train_best_score,
            'test_score': test_score,
        })

    combined_equity = (1.0 + combined_returns.fillna(0.0)).cumprod()

    return {
        'splits': pd.DataFrame(split_rows),
        'combined_returns': combined_returns,
        'combined_equity': combined_equity,
        'metric': metric,
        'method': method,
        'mode': mode,
    }

