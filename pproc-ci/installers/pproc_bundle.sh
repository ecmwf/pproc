
 # switch to gnu environment
set +xv
module switch cdt/18.12
set -xv
source /usr/local/etc/ksh_functions/prgenvswitchto
prgenvswitchto gnu

./bundle-create
./bundle-build --install-dir $LIB_DIR/pproc_env
build/install.sh --fast 
