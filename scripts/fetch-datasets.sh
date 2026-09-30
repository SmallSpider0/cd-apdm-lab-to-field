#!/usr/bin/env bash
# 下载基准数据集到 $AGRI_WORKSPACE/data（仓库外，非 iCloud）。
#
# 服务器无外网，数据须在本机下载后传输过去，因此本脚本是必经步骤。
# PlantVillage 仓库含 color/grayscale/segmented 三套，本方案只用 color，
# 故用 sparse-checkout 只取 raw/color，避免多下约 2/3 的体积。
set -euo pipefail

WS="${AGRI_WORKSPACE:-$HOME/agri-cnz-workspace}"
DATA="$WS/data"
LOG="$WS/data/fetch.log"
case "$WS" in *"Mobile Documents"*|*CloudDocs*) echo "✗ 不得下载到云同步目录" >&2; exit 1;; esac
mkdir -p "$DATA"
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date '+%F %T') 开始 ==="

fetch_plantvillage(){
  local d="$DATA/plantvillage"
  if [[ -d "$d/raw/color" ]]; then echo "→ PlantVillage 已存在，跳过"; return; fi
  echo "→ PlantVillage（sparse: 仅 raw/color）"
  rm -rf "$d"
  git clone --depth 1 --filter=blob:none --sparse \
      https://github.com/spMohanty/PlantVillage-Dataset.git "$d"
  git -C "$d" sparse-checkout set raw/color
  echo "  完成"
}

fetch_plantdoc(){
  local d="$DATA/plantdoc"
  if [[ -d "$d/train" ]]; then echo "→ PlantDoc 已存在，跳过"; return; fi
  echo "→ PlantDoc"
  rm -rf "$d"
  git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset.git "$d"
  echo "  完成"
}

verify(){
  echo "=== 校验（对照 EXP-1a 实测计数）==="
  local pv pd
  pv=$(find "$DATA/plantvillage/raw/color" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) 2>/dev/null | wc -l | tr -d ' ')
  pd=$(find "$DATA/plantdoc" -path '*/train/*' -o -path '*/test/*' 2>/dev/null | grep -icE '\.(jpg|jpeg|png)$' || true)
  printf "  PlantVillage raw/color : %s  (期望 54305) %s\n" "$pv" "$([[ $pv == 54305 ]] && echo ✓ || echo ⚠)"
  printf "  PlantDoc train+test    : %s  (期望 2578)  %s\n" "$pd" "$([[ $pd == 2578 ]] && echo ✓ || echo ⚠)"
  du -sh "$DATA"/* 2>/dev/null | sed 's|^|  |'
}

fetch_plantvillage
fetch_plantdoc
verify
echo "=== $(date '+%F %T') 结束 ==="
