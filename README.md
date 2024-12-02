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

```bash
dir_loc <required>:		location of the directory with mesospim files
--out-dir: 				optional directory for output files <defaults to </dir_loc/decon>
--out-file-type:		extension of output files
--file-type:			extension of input files <currently only supports .btf>
--queue-ims:			After decon convert to ims file using SLURM nodes <default NO>
--sharpen:				unsharp filter applied to each decon <default NO>
--denoise-sigma:		Pre gaussian filter before decon to reduce noise <default None>
--num-parallel:			Decon SLURM jobs to run in parallel. 0 is unlimited. <default 0>
```



Example:

```bash
/CBI_FastStore/cbiPythonTools/mesospim_utils/mesospim_utils/rl.py decon-dir /CBI_FastStore/mesospim/081924 --out-dir /CBI_FastStore/mesospim/081924/decon_no_denoise_no_sharp --queue-ims --num-parallel 0
```

