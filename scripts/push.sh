#!/usr/bin/env bash
# Commit this version and push it to GitHub (public repo, created on first push).
# Override the target with INKSOLVE_REPO=owner/name.
set -euo pipefail
cd "$(dirname "$0")/.."
VER="$(cat VERSION)"
REPO="${INKSOLVE_REPO:-merolaagi/inksolve}"
[ -d .git ] || git init -q -b main
git add -A
git commit -q -m "InkSolve v$VER" || echo "Nothing new to commit."
git tag -f "v$VER" >/dev/null
if ! git remote get-url origin >/dev/null 2>&1; then
  if command -v gh >/dev/null 2>&1; then
    gh repo view "$REPO" >/dev/null 2>&1 || gh repo create "$REPO" --public \
      --description "Handwritten maths, read with context and solved step by step: draw, photograph or type it."
    git remote add origin "https://github.com/$REPO.git"
  else
    echo "GitHub CLI not found. Create https://github.com/$REPO (public, empty), then re-run this script."
    git remote add origin "git@github.com:$REPO.git"
  fi
fi
git push -u origin main
git push -f origin "v$VER"
echo "Pushed v$VER to https://github.com/$REPO"
