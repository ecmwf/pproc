#!/bin/bash
#SBATCH --job-name=ecpoint_012_0_12_0_12
#SBATCH --qos=np
#SBATCH --nodes=1
#SBATCH --output=ecpoint_012_0_12.1

export ASTREAM="enfo"
export BASETIME="${BASETIME:=$(module load eclib; newdate -D 20250921 1)00}"
export BP_LOCATION="/ec/res4/scratch/mawj/50r1/epproc/data/ecpoint12h/BP.csv"
export CLASS="od"
export EMOS_BASE="00"
export ENS_BATCH_SIZE="51"
export FDB_DIR="/ec/res4/scratch/mawj/50r1/epproc/fdb"
export FDB_ENABLE_GRIBJUMP="0"
export FER_LOCATION="/ec/res4/scratch/mawj/50r1/epproc/data/ecpoint12h/FER.csv"
export INPUT_EXPVER="0001"
export IN_FC="fdb:"
export MIR_CACHE="/ec/res4/scratch/mawj/50r1/epproc/mir_cache"
export N_PAR_READ="1"
export OUTPUT_EXPVER="0001"
export OUTPUT_ROOT="/ec/res4/scratch/mawj/50r1/epproc"
export OUT_BS="fileset:bs.grib"
export OUT_PERC="fileset:perc.grib"
export OUT_WT="fileset:wt.grib"
export PPROC_LOG="INFO"
export PPROC_SCHEMA="/ec/res4/scratch/mawj/50r1/epproc/data/pproc/schema"
export USE_CHECKPOINT="1"
export WT_BATCH_SIZE="10"



trap - 15

# load tools and activate environment

export MIR_CACHE_PATH=${MIR_CACHE:-$OUTPUT_ROOT/local/mir_cache}:${MIR_CACHE_PATH:-}

export ECCODES_PYTHON_USE_FINDLIBS=1

export GRIBJUMP_HOME=${GRIBJUMP_HOME:-}

export FDB_ENABLE_GRIBJUMP=${FDB_ENABLE_GRIBJUMP:-0}

export FDB_HOME=$FDB_DIR

module use /usr/local/apps/pproc/modulefiles

set +ux
module unload pproc || true
module load pproc/unstable/cgen-6
set -ux
export OMP_NUM_THREADS=1
cat > outputs.yaml <<EOF
- class: "$CLASS"
  stream: "$ASTREAM"
  expver: "$OUTPUT_EXPVER"
  levtype: sfc
  domain: g
  date: "${BASETIME:0:8}"
  time: "$EMOS_BASE"
  type: pfc
  model: ecPoint
  quantile:
  - 1:100
  - 2:100
  - 3:100
  - 4:100
  - 5:100
  - 6:100
  - 7:100
  - 8:100
  - 9:100
  - 10:100
  - 11:100
  - 12:100
  - 13:100
  - 14:100
  - 15:100
  - 16:100
  - 17:100
  - 18:100
  - 19:100
  - 20:100
  - 21:100
  - 22:100
  - 23:100
  - 24:100
  - 25:100
  - 26:100
  - 27:100
  - 28:100
  - 29:100
  - 30:100
  - 31:100
  - 32:100
  - 33:100
  - 34:100
  - 35:100
  - 36:100
  - 37:100
  - 38:100
  - 39:100
  - 40:100
  - 41:100
  - 42:100
  - 43:100
  - 44:100
  - 45:100
  - 46:100
  - 47:100
  - 48:100
  - 49:100
  - 50:100
  - 51:100
  - 52:100
  - 53:100
  - 54:100
  - 55:100
  - 56:100
  - 57:100
  - 58:100
  - 59:100
  - 60:100
  - 61:100
  - 62:100
  - 63:100
  - 64:100
  - 65:100
  - 66:100
  - 67:100
  - 68:100
  - 69:100
  - 70:100
  - 71:100
  - 72:100
  - 73:100
  - 74:100
  - 75:100
  - 76:100
  - 77:100
  - 78:100
  - 79:100
  - 80:100
  - 81:100
  - 82:100
  - 83:100
  - 84:100
  - 85:100
  - 86:100
  - 87:100
  - 88:100
  - 89:100
  - 90:100
  - 91:100
  - 92:100
  - 93:100
  - 94:100
  - 95:100
  - 96:100
  - 97:100
  - 98:100
  - 99:100
  step: 0-12
  target_grid: "O640"
  param: 228

EOF
cat > template.yaml <<EOF
recovery:
  from_checkpoint: !!int '$USE_CHECKPOINT'
bp_location: $BP_LOCATION
fer_location: $FER_LOCATION
inputs:
  fc:
    source: "$IN_FC"
outputs:
  bs:
    target: "$OUT_BS"
    metadata:
      expver: "$OUTPUT_EXPVER"
      backgroundProcess: 4
  wt:
    target: "$OUT_WT"
    metadata:
      expver: "$OUTPUT_EXPVER"
      backgroundProcess: 4
  perc:
    target: "$OUT_PERC"
    metadata:
      expver: "$OUTPUT_EXPVER"
      backgroundProcess: 4
parameters:
  default:
    inputs:
      fc:
        request:
          expver: "$INPUT_EXPVER"
parallelisation:
  wt_batch_size: $WT_BATCH_SIZE
  ens_batch_size: $ENS_BATCH_SIZE
  n_par_read: $N_PAR_READ

EOF
pproc-config --config ecpoint-config.yaml from_outputs --outputs outputs.yaml --schema $PPROC_SCHEMA --overrides template.yaml

RECOVER=""
[[ ${USE_CHECKPOINT:-0} -eq 1 ]] && RECOVER="--recover"

pproc-ecpoint --config ecpoint-config.yaml --log ${PPROC_LOG:-INFO} $RECOVER
