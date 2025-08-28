

# Help on DNAnexus platfrom


## set-up

```
#!/usr/bin/env bash
# Quick DNAnexus SDK setup using mamba

set -e

mamba create -n dnx -y python=3.10
source "$(mamba info --base)/etc/profile.d/conda.sh"
mamba activate dnx

mamba install -y -c conda-forge -c bioconda dxpy

```


## usage

```
python -c "import dxpy; print('dxpy version:', dxpy.__version__)"
dx --version

echo "Done. Next:"
echo "  mamba activate dnx"
echo "  dx login        # enter username+password or API token"
echo "  dx whoami       # confirm username"
echo "  dx select       # choose project when available"
```
