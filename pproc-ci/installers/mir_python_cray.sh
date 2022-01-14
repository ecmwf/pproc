
 # switch to gnu environment
set +xv
module switch cdt/18.12
set -xv
source /usr/local/etc/ksh_functions/prgenvswitchto
prgenvswitchto gnu

rm -rf .git # required to avoid pip errors... :/

pip uninstall mir-python -y

export MIR_LIB_DIR=$LIB_DIR/$PPROC_ENV/lib
export MIR_INCLUDE_DIRS=$LIB_DIR/$PPROC_ENV/include

pip install .

python3 setup.py --version > version.txt
