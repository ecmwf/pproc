
rm -rf .git # required to avoid pip errors... :/

pip uninstall eccodeshl -y

pip install .

python3 setup.py --version > version.txt
