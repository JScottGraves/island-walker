#!/bin/bash
# Turns compiler errors and XCTest failures in a log into GitHub annotations, so they
# show on the PR (and through the checks API) without opening the raw job log.
log="$1"
grep -E '^/[^:]+:[0-9]+:[0-9]+: error: |error: ' "$log" | sort -u | head -40 | while IFS= read -r line; do
  if [[ "$line" =~ ^(/[^:]+):([0-9]+):([0-9]+):\ error:\ (.*)$ ]]; then
    file="${BASH_REMATCH[1]#"$GITHUB_WORKSPACE/"}"
    echo "::error file=$file,line=${BASH_REMATCH[2]},col=${BASH_REMATCH[3]}::${BASH_REMATCH[4]}"
  else
    echo "::error::$line"
  fi
done
# If nothing matched, surface the last lines of the log instead.
if ! grep -qE 'error: ' "$log"; then
  tail -n 15 "$log" | while IFS= read -r line; do echo "::error::$line"; done
fi
