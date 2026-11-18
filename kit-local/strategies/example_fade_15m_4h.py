"""示例策略（也是第 1 层审计里看起来最好的那个）：反着 15 分钟预测做，持有 4 小时。只用来演示接口，不代表有效。"""
import pandas as pd

NAME = "example_fade_15m_4h"
DESCRIPTION = "每个 15m 预测在看板上可见时：若它 4 小时后的中位数高于现价就做空，低于就做多，持有到那根 K 线收盘。"
PARAMS = {"scale": "15m", "hold_h": 4.0, "fade": True}


def signals(ds, scale="15m", hold_h=4.0, fade=True):
    rows = []
    F, P = ds.forecasts, ds.pairs
    for f in F[F.scale == scale].itertuples(index=False):
        now = f.visible_at                                    # 决策时刻：这个预测在看板上可见的时间
        fut = P[(P.fid == f.fid) & (P.t >= now + pd.Timedelta(hours=hold_h))]
        if not len(fut):
            continue
        tgt = fut.iloc[0]
        px = ds.at(now).last_price(f.asset)                   # 只用决策时刻之前的价格
        side = 1 if tgt.q50 > px else -1
        rows.append(dict(asset=f.asset, side=-side if fade else side, entry_time=now, exit_time=tgt.t, tag=f.fid))
    return pd.DataFrame(rows)
