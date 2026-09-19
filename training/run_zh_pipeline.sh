#!/bin/bash
# 中文判官训练全自动流水线：等额度重置 → 生成 → 审计 → 烟测 → 训练 → 换checkpoint → judge-econ 验收
# 日志：training/pipeline.log （各阶段失败即停，日志里查 FAIL）
set -u
LOG="E:/Projects/MyGitHub/jev-judge-projects/training/pipeline.log"
NJ="C:/Users/14970/AppData/Local/Temp/NanoJev/scripts"
CKPT="D:/Models/NanoJev"
OUTDIR="D:/Models/NanoJev-zh/run1"
log(){ echo "[$(date +%m-%d\ %T)] $*" >> "$LOG"; }

log "=== pipeline start ==="

# 阶段0：等 GLM 5小时额度重置（08:19:51）
WAIT=$(python -c "import time; print(max(0, time.mktime((2026,9,20,8,21,30,0,0,-1)) - time.time()))")
log "waiting ${WAIT}s for quota reset"
sleep "$WAIT"
log "quota window open, starting generation"

# 阶段1：数据生成（GLM 订阅，并发1，2000样本×k3）
export ZHIPU_CODING_KEY=3c246d0321f546a79a5e26ee69b92957.v4oL8tAKK2DyhWwP
cd /e/Projects/MyGitHub/jev-judge-projects
python training/gen_zh_data.py --meta-batches 28 --samples 500 --k 3 --workers 1 >> "$LOG" 2>&1
N=$(wc -l < training/zh_records.jsonl)
log "generation done: $N records"
if [ "$N" -lt 400 ]; then log "FAIL: too few records ($N), abort"; exit 1; fi

# 阶段2：schema 审计
cd "$NJ" && python train_pipeline_decisions.py --input E:/Projects/MyGitHub/jev-judge-projects/training/zh_records.jsonl --validate-only >> "$LOG" 2>&1 \
  || { log "FAIL: validate-only"; exit 1; }
log "validate-only passed"

# 阶段3：腾显存（杀旧 NanoJev 服务）
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object {\$_.CommandLine -like '*serve_decisions.py*'} | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" >> "$LOG" 2>&1
sleep 5

# 阶段4：训练烟测（3 步，防 OOM）
python train_pipeline_decisions.py --input E:/Projects/MyGitHub/jev-judge-projects/training/zh_records.jsonl \
  --init-checkpoint "$CKPT" --objective gold_distribution --loss ce \
  --steps 3 --head-steps 6 --batch-questions 8 --microbatch-questions 2 \
  --max-microbatch-tokens 8192 --gradient-checkpointing --max-length 512 \
  --output-dir "$OUTDIR-smoke" >> "$LOG" 2>&1 \
  || { log "FAIL: training smoke (OOM? try --microbatch-questions 1 / lower max-microbatch-tokens)"; exit 1; }
log "training smoke passed"

# 阶段5：正式训练（150 步）
python train_pipeline_decisions.py --input E:/Projects/MyGitHub/jev-judge-projects/training/zh_records.jsonl \
  --init-checkpoint "$CKPT" --objective gold_distribution --loss ce \
  --steps 150 --head-steps 12 --batch-questions 8 --microbatch-questions 2 \
  --max-microbatch-tokens 8192 --gradient-checkpointing --max-length 512 --seed 17 \
  --output-dir "$OUTDIR" >> "$LOG" 2>&1 \
  || { log "FAIL: full training"; exit 1; }
log "training done -> $OUTDIR"

# 阶段6：补齐 tokenizer/config（若训练产物缺失）并启动新 checkpoint 服务
cp -r "$CKPT/tokenizer" "$OUTDIR/" 2>>"$LOG"; cp "$CKPT/config.json" "$OUTDIR/" 2>>"$LOG"; cp -r "$CKPT/backbone_config" "$OUTDIR/" 2>>"$LOG"
cd "$NJ" && python serve_decisions.py --checkpoint-dir "$OUTDIR" --precision bf16 --port 8765 >> "$LOG" 2>&1 &
sleep 90
curl -s -m 10 -x "" http://127.0.0.1:8765/api/health >> "$LOG" 2>&1 || { log "FAIL: new server not healthy"; exit 1; }
log "new checkpoint server up on 8765"

# 阶段7：judge-econ 验收（130 条，本地零成本）
cd /e/Projects/MyGitHub/jev-judge-projects
python benchmarks/run_bench.py --providers-file benchmarks/models.yaml --models nanojev-local \
  --datasets intent_zh,sentiment_zh,spam_zh,urgency_zh --limit 0 --tag zhft >> "$LOG" 2>&1
log "=== pipeline complete ==="
