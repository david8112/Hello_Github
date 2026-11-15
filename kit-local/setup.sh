#!/usr/bin/env bash
# KiT 一键本地部署（Linux / macOS）
#   bash setup.sh            # 默认：代码 + 依赖 + 官方模型(step_market_284M) + 100 只股票示例数据
#   bash setup.sh --both     # 额外下载 A 股专用的 step_market_284M_train260410 模型（再 +1.1GB）
#   bash setup.sh --no-data  # 不下载示例数据（只用自己的数据时）
# 国内网络下载 Hugging Face 慢，可先: export HF_ENDPOINT=https://hf-mirror.com
set -euo pipefail

KIT_REPO="https://github.com/Luciferbobo/KiT.git"
KIT_COMMIT="78acec5f9de5787672f0f9cc201f73c6e3526c43"   # 2026-11-13，本部署验证过的版本
HERE="$(cd "$(dirname "$0")" && pwd)"
KIT="$HERE/KiT"
WANT_DATA=1; WANT_BOTH=0
for a in "$@"; do
  case "$a" in
    --no-data) WANT_DATA=0 ;;
    --both) WANT_BOTH=1 ;;
    *) echo "未知参数: $a"; exit 2 ;;
  esac
done

PY="${PYTHON:-python3}"
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), sys.version' \
  || { echo "需要 Python >= 3.10（可用 PYTHON=/path/to/python3.12 bash setup.sh 指定）"; exit 1; }
command -v git >/dev/null || { echo "需要 git"; exit 1; }

echo "==> 1/4 获取 KiT 代码 -> $KIT"
if [ ! -d "$KIT/.git" ]; then
  git clone "$KIT_REPO" "$KIT"
fi
git -C "$KIT" fetch -q origin "$KIT_COMMIT" 2>/dev/null || true
git -C "$KIT" checkout -q "$KIT_COMMIT"

echo "==> 2/4 创建虚拟环境并安装依赖"
[ -x "$KIT/.venv/bin/python" ] || "$PY" -m venv "$KIT/.venv"
VPY="$KIT/.venv/bin/python"
"$VPY" -m pip install -q --upgrade pip
if "$VPY" -c 'import torch' 2>/dev/null; then
  echo "    已有 torch，跳过"
elif [ "$(uname -s)" = "Linux" ] && ! command -v nvidia-smi >/dev/null; then
  echo "    Linux 无 NVIDIA 显卡：安装 CPU 版 PyTorch（约 200MB）"
  "$VPY" -m pip install "torch>=2.4" --index-url https://download.pytorch.org/whl/cpu \
    || "$VPY" -m pip install "torch>=2.4"
else
  # Linux + NVIDIA：PyPI 默认即 CUDA 版；macOS：CPU/MPS 版
  "$VPY" -m pip install "torch>=2.4"
fi
"$VPY" -m pip install -q numpy pandas pyarrow matplotlib huggingface_hub

echo "==> 3/4 下载模型和数据（Hugging Face）"
# 默认用普通 HTTPS 下载，不走 Xet 协议（cas-server.xethub.hf.co）：镜像站和公司代理下更稳。
# 想用 Xet 加速可以: HF_HUB_DISABLE_XET=0 bash setup.sh
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
cd "$KIT"
WANT_DATA=$WANT_DATA WANT_BOTH=$WANT_BOTH "$VPY" - <<'PY'
import os
from huggingface_hub import snapshot_download
pats = ["config.json", "step_market_284M/*"]
if os.environ["WANT_BOTH"] == "1":
    pats.append("step_market_284M_train260410/*")
snapshot_download("Lucifer744/KiT_model", allow_patterns=pats, local_dir="ckpt")
if os.environ["WANT_DATA"] == "1":
    snapshot_download("Lucifer744/KiT_data_demo", repo_type="dataset", local_dir="data")
PY
# 官方脚本默认找 ckpt/step_market_284M.pt，而 HF 上的文件在子目录里；建软链接让官方命令直接可用
ln -sf step_market_284M/step_market_284M.pt ckpt/step_market_284M.pt

echo "==> 4/4 自检"
"$VPY" - <<'PY'
import torch
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"    torch {torch.__version__}  可用设备: {dev}")
if dev == "cuda":
    print(f"    GPU: {torch.cuda.get_device_name(0)}  显存 {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
else:
    print("    没有 CUDA：会用 CPU 推理（fp32），能跑但慢（4 核实测：日线约 5 分钟/案例，30 分钟线约 15 分钟/案例）")
PY
echo
echo "部署完成。试试："
echo "  $VPY $KIT/eval/infer_for_pred.py --case 300260.SZ,30m,2026-01-26 --paths 16"
echo "  $VPY $HERE/forecast.py --code 300260.SZ --scale 30m --anchor 2026-01-26"
