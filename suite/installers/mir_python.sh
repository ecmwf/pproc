
rm -rf .git # required to avoid pip errors... :/

pip uninstall mir-python -y

export MIR_LIB_DIR=$LIB_DIR/$PPOP_ENV/lib
export MIR_INCLUDE_DIRS=$LIB_DIR/$PPOP_ENV/include

pip install .

python3 setup.py --version > version.txt
