#!/usr/bin/env bash
# 定时刷新「BTC·SOL 行情预测」看板的数据。
#   bash dashboard_refresh.sh            # 过期的预测才重跑，然后更新实际行情和看板数据
#   bash dashboard_refresh.sh actuals    # 只更新实际行情（几秒钟）
#   bash dashboard_refresh.sh force      # 不管新旧，全部预测重跑一遍
# 数据源：OKX 永续合约（BTC-USDT-SWAP、SOL-USDT-SWAP），和 OKX App 里看到的一致。
# 规则：15分钟/1小时/2小时线的预测超过 4 小时就重跑，日线超过 22 小时重跑；
#       同一时间只跑一个模型任务（CPU 只有 4 核），CPU 上用 bf16 提速。
# 结果写在 dashboard/*.json，由 Claude 上传到看板。
set -u
cd "$(dirname "$0")"
PY=KiT/.venv/bin/python
export PYTHONWARNINGS=ignore
MODE=${1:-auto}

if [ ! -x "$PY" ]; then echo "KiT 还没部署，先运行 bash setup.sh"; exit 1; fi

age_hours() {   # 某个品种 × 周期最新预测文件的年龄（小时），没有则返回 9999
  local f
  f=$(ls -t outputs/"$1"_"$2"_*.npz 2>/dev/null | head -1)
  if [ -z "$f" ]; then echo 9999; else echo $(( ($(date +%s) - $(stat -c %Y "$f")) / 3600 )); fi
}

if [ "$MODE" = "auto" ] || [ "$MODE" = "force" ]; then
  exec 9>/tmp/kit-dashboard.lock
  flock -n 9 || { echo "上一次刷新还在运行，这次跳过"; exit 0; }
  # 等其他模型任务（回测、手动预测）结束，避免抢 CPU
  while pgrep -f "python (backtest|forecast)\.py" >/dev/null; do sleep 20; done
  for code in SOLUSDT BTCUSDT; do
    for spec in "15m 16 4" "1h 16 4" "2h 16 4" "1d 32 22"; do
      set -- $spec
      scale=$1; paths=$2; max_age=$3
      age=$(age_hours "$code" "$scale")
      if [ "$MODE" = "force" ] || [ "$age" -ge "$max_age" ]; then
        echo "== 重跑 $code $scale（上次预测 ${age} 小时前）"
        $PY fetch_crypto.py --exchange okx --swap --symbol "$code" --interval "$scale" >/dev/null &&
        $PY forecast.py --market crypto --code "$code" --scale "$scale" --file "data_crypto/${code}_${scale}_okx_swap.csv" \
            --paths "$paths" --precision bf16 --ctx-show 120 | grep -E "中位数|历史截至" || echo "   $code $scale 预测失败"
      fi
    done
  done
fi
$PY dashboard_data.py
