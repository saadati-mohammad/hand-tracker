#!/bin/bash

# Switch to the script's directory (the repo root)
cd "$(dirname "$0")" || exit

# Generate commit message with timestamp
MSG="Auto-commit: $(date '+%Y-%m-%d %H:%M:%S')"

# Stage everything
git add -A

# Only commit if there are actually changes
if ! git diff --cached --quiet; then
    git commit -m "$MSG"
    git push origin main
else
    echo "No changes to commit at $MSG"
fi
