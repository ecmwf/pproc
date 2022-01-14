
templates_fdb=$DATA_DIR/fdb_templates
fdb_conf=$LIB_DIR/$PPOP_ENV/etc/fdb

mkdir -p $FDB_DIR
mkdir -p $fdb_conf

cp $templates_fdb/schema $fdb_conf/
cp $templates_fdb/config.yaml $fdb_conf/config.yaml
sed -i "s:FDB_SCHEMA:$fdb_conf/schema:" $fdb_conf/config.yaml
sed -i "s:FDB_ROOT:$FDB_DIR:" $fdb_conf/config.yaml
