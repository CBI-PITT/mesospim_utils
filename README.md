# mesospim_utils

## This repository offers tools to deal with mesospim data at CBI

#### Installing:

```bash
# Clone the repo
cd /dir/of/choice
git clone https://github.com/CBI-PITT/mesospim_utils.git

# Create a virtual environment
# This assumes that you have miniconda or anaconda installed
conda create -n decon python=3.12 -y

# Activate environment and install zarr_stores
conda activate decon
pip install -e /dir/of/choice/mesospim_utils
```



##### <u>Richardson-Lucy 3D Deconvolution:</u>

###### Description:

Richardson-Lucy deconvolution of mesospim .btf files. The library is designed to spin off all computations on the CBI SLURM cluster including deconvolution for each .btf file and finally conversion of the resulting files to Imaris. The scripts are intended to be used as commandline tools. 

###### Usage example:

```bash
/CBI_FastStore/cbiPythonTools/mesospim_utils/mesospim_utils/rl.py decon-dir --help
```

![decon-dir-help](https://github.com/CBI-PITT/mesospim_utils/raw/refs/heads/main/images/decon-dir-help.png)

