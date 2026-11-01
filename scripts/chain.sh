#!/usr/bin/env bash
# wait for kernel $1 to finish, then run push_and_wait.sh $2
cd "$(dirname "$0")/.."
while true; do
  ST=$(kaggle kernels status "shamimahossain/dna-grpo-$1" 2>&1 | tr '[:upper:]' '[:lower:]')
  case "$ST" in *complete*|*error*|*cancel*) break ;; esac
  sleep 30
done
mkdir -p "outputs/$1"; kaggle kernels output "shamimahossain/dna-grpo-$1" -p "outputs/$1" -o >/dev/null 2>&1
echo "$1 finished: $ST"
until scripts/push_and_wait.sh "$2"; do sleep 30; done
