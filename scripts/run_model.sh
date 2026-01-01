#!/usr/bin/env bash
set -e

# Go to project root
echo "Moving to proj root"
cd ~/local-llm

echo "Updating repo"
old_rev=$(git rev-parse HEAD 2>/dev/null || true)
git pull
new_rev=$(git rev-parse HEAD 2>/dev/null || true)

if [ -n "$old_rev" ] && [ -n "$new_rev" ] && [ "$old_rev" != "$new_rev" ]; then
  if git diff --name-only "$old_rev" "$new_rev" | grep -qx "requirements.txt"; then
    echo "Requirements changed; installing deps"
    pip install -r requirements.txt
  fi
fi

echo "Activating venv"
source venv/bin/activate

echo "Moving model install location"
export HF_HOME=/tmp/hf_cache
export HF_HUB_DISABLE_TELEMETRY=1

echo "Launching app"
streamlit run src/app.py
