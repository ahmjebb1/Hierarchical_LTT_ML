### Modules

```
module load Python/3.11.3-GCCcore-12.3.0
```

### Virtual Environment
#### Loading
```
source venv/bin/activate
```
#### Creating
This assumes that [~/fastdata is mapped to the parscratch share](https://docs.hpc.shef.ac.uk/en/latest/hpc/filestore.html#fastdata-areas)
```
mkdir ~/fastdata/ltt-ml-venv
ln -s ~/fastdata/ltt-ml-venv venv
cd venv
python3.11 -m venv .
cd ..
source /venv/bin/activate
pip install -r requirements.txt
pip install --upgrade pip
```

