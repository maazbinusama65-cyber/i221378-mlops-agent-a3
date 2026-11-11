#!/usr/bin/env bash
# Recreate the DVC objects of both dataset versions on a fresh clone.
#
# The Git history already contains the .dvc pointers (tags data-v1, data-v2) but the data objects live in the
# DVC remote, which is not part of Git. The generator is deterministic, so regenerating each version must reproduce
# the recorded md5 exactly - this script verifies that, then pushes the objects to the configured remote.
set -euo pipefail
cd "$(dirname "$0")/.."

for version in v1 v2; do
  git checkout "data-${version}" -- data/churn.csv.dvc
  recorded=$(grep -m1 "md5:" data/churn.csv.dvc | awk '{print $3}')
  python data/make_dataset.py --version "${version}" --out data/churn.csv
  dvc add data/churn.csv >/dev/null
  actual=$(grep -m1 "md5:" data/churn.csv.dvc | awk '{print $3}')
  if [ "${recorded}" != "${actual}" ]; then
    echo "ERROR: dataset ${version} md5 mismatch (recorded ${recorded}, regenerated ${actual})" >&2
    exit 1
  fi
  echo "dataset ${version}: md5 ${actual} reproduced"
  dvc push
done

git checkout HEAD -- data/churn.csv.dvc
dvc checkout data/churn.csv.dvc --force
echo "DVC remote populated; working tree restored to the current dataset version"
