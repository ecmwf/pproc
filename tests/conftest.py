import os
import pytest
import shutil
import tempfile

import pyfdb

TEST_DIR = os.path.dirname(os.path.realpath(__file__))
NEXUS = "https://get.ecmwf.int/test-data/pproc/test-data"


@pytest.fixture(scope="module")
def fdb() -> pyfdb.FDB:
    tmpdir = tempfile.mkdtemp()
    print("Using temporary directory", tmpdir)
    os.makedirs(f"{tmpdir}/etc/fdb")
    os.mkdir(f"{tmpdir}/fdb")
    shutil.copyfile(f"{TEST_DIR}/templates/fdb/schema", f"{tmpdir}/etc/fdb/schema")
    with open(f"{tmpdir}/etc/fdb/config.yaml", "w") as f:
        f.write(
            f"""
---
type: local
engine: toc
schema: "{tmpdir}/etc/fdb/schema"
spaces:
- roots:
    - path: {tmpdir}/fdb
"""
        )
    os.environ["FDB_HOME"] = str(tmpdir)
    os.environ["FDB_HANDLE_LUSTRE_STRIPE"] = "0"
    temp_fdb = pyfdb.FDB()
    yield temp_fdb
    shutil.rmtree(tmpdir)
