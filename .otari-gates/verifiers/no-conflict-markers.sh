#!/usr/bin/env bash
# check_passed verifier: fails when a tracked file still carries a leftover
# Git merge-conflict marker (a line starting with <<<<<<<, |||||||,
# =======, or >>>>>>>; the second is diff3's base section, which survives a
# resolution that deleted only the other three). Run with cwd at the repo
# root, per docs/agent-gates.md's check_passed exit-code contract: 0 is
# pass, 1 is fail, anything else is error. Uses `git grep`, not the system
# `grep` binary, so this behaves the same on every platform `git` itself
# does, rather than depending on which grep flavor happens to be on PATH.
set -u

pattern='^(<<<<<<<|\|\|\|\|\|\|\||=======|>>>>>>>)( .*)?$'
matches=$(git grep -In -E "$pattern" -- . 2>&1)
status=$?

if [ "$status" -eq 1 ]; then
  exit 0
fi
echo "$matches"
if [ "$status" -eq 0 ]; then
  exit 1
fi
exit 2
