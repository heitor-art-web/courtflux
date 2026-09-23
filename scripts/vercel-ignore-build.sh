#!/usr/bin/env bash

previous_sha="${VERCEL_GIT_PREVIOUS_SHA:-}"

if [ -z "$previous_sha" ]; then
  echo "Vercel previous deployment SHA unavailable; building."
  exit 1
fi

if git diff --quiet "$previous_sha" HEAD -- \
  web \
  package.json \
  package-lock.json \
  vercel.json \
  .vercelignore \
  scripts/vercel-ignore-build.sh; then
  echo "No web deployment inputs changed; skipping Vercel build."
  exit 0
fi

status=$?
if [ "$status" -eq 1 ]; then
  echo "Web deployment inputs changed; building."
else
  echo "Could not compare against previous deployment; building."
fi
exit 1
