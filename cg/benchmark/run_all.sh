#!/bin/bash
# Tune every representation on one set (default pp48) at once; resumable.
#   FPOCKET_TMP=/fast/tmp ./run_all.sh [set]
source ~/miniforge3/etc/profile.d/conda.sh
conda activate boonza
cd "$(dirname "$0")"
set_=${1:-pp48}
mkdir -p runs/$set_
for spec in "sirah 250 150 4" "martini3 250 150 3" "martini2 250 150 3" "aa 100 50 2"; do
    set -- $spec
    python tune.py --set $set_ --model $1 --random $2 --local $3 -j $4 >> runs/$set_/$1.log 2>&1 &
done
wait
python report.py --train $set_
python calibrate_density.py --set $set_
