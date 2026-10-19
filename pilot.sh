#!/bin/bash
# Pilot: does merge-consistency training recover FPET's accuracy drop at early merge layers?
# 4 datasets x {PET baseline, FPET r=2, MC-FPET r=2, FPET r=4, MC-FPET r=4} = 20 runs.
# Finished runs (summary.json present) are skipped, so the script can be re-run after a disconnect.
# Usage: VTAB_ROOT=/path/to/vtab-1k sh pilot.sh

DATASETS=${DATASETS:-"cifar dtd dsprites_loc smallnorb_azi"}
SEED=${SEED:-0}
OUT=${OUT:-./result/pilot}
COMMON="--method adaptformer --dim 8 --bit 32 --load_config --seed $SEED"

run() {
    NAME=$1; DATASET=$2; shift 2
    DIR=$OUT/$NAME/seed$SEED
    if [ -f "$DIR/$DATASET/summary.json" ]; then
        echo "skip $NAME $DATASET (done)"
        return
    fi
    mkdir -p $DIR/$DATASET
    python main.py --dataset $DATASET $COMMON --model_path $DIR "$@" 2>&1 | tee $DIR/$DATASET/stdout.log
}

for DATASET in $DATASETS
    do
        run adaptformer $DATASET
        for R in 2 4
            do
                run fpet-r$R $DATASET --r_layer $R
                run mcpet-r$R $DATASET --r_layer $R --consistency
            done
    done

python collect_results.py --root $OUT --pairs fpet-r2:mcpet-r2 fpet-r4:mcpet-r4 adaptformer:mcpet-r4
