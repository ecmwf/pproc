
templates_fdb=$DATA_DIR/fdb_templates

mkdir -p $FDB_DIR

cp $templates_fdb/config.yaml $FDB_DIR/config.yaml
sed -i "s:FDB_SCHEMA:$templates_fdb/schema:" $FDB_DIR/config.yaml
sed -i "s:FDB_ROOT:$FDB_DIR:" $FDB_DIR/config.yaml
