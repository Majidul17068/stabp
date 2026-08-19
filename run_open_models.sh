#!/bin/bash
# Open-model landscape: run TenantBound relay (k=40) on several local Ollama
# models, judged by gpt-4o. Free generation (local), small judge cost.
cd /Users/majidmurad/Desktop/research-lab/stabp

echo "=== pulling models ==="
ollama pull llama3.2:1b 2>&1 | tail -1 || echo "pull llama3.2:1b failed"
ollama pull mistral:7b  2>&1 | tail -1 || echo "pull mistral:7b failed"

# llama3.2:3b already present; run the full spread
for m in llama3.2:1b llama3.2:3b mistral:7b; do
  safe=$(echo "$m" | tr ':/.' '___')
  echo "=== running $m (relay, k=40, gpt-4o judge) ==="
  python3 tenantbound.py --provider openai --model "$m" \
    --base-url http://localhost:11434/v1 --judge-model gpt-4o \
    --scenario relay --repeat 40 --out "results_${safe}.json" 2>&1 \
    | grep -E "policy|naive|prompt|typed|leak|error|LLM error" || echo "FAILED $m"
  echo "--- done $m ---"
done
echo "=== ALL OPEN-MODEL RUNS COMPLETE ==="