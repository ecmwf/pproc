
rm -rf .git # required to avoid pip errors... :/

pip uninstall pyfdb -y

pip install .

python3 setup.py --version > version.txt
