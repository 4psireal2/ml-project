## Ideas
- [] Predict pDOS of HEA such as CoCrFeMnNi

## Technical details
Required tools: `conda`, `conda-lock`, `poetry`

### Do this once (create and activate the environment)

```
conda-lock install --name YOURENV linux-64.conda.lock
conda activate YOURENV
make init
```

### Do this happily ever after (Update the environment)

```
# Re-generate Conda lock file(s) based on environment.yml
conda-lock -f environment.yml -p linux-64 -k explicit --filename-template "linux-64.conda.lock"
# Update Conda packages based on re-generated lock file
conda update --file linux-64.conda.lock
# Update Poetry packages and re-generate poetry.lock
poetry update
```