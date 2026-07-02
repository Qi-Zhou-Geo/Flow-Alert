```sh
# Last Update: Last modified: 2026-07-02T11:40:41
# Author: Qi Zhou
```

## Prepare the Python Environment

### 1. Make sure you have Git and Mamba (or Conda)
Please install the following tools first:

- **Git**: https://git-scm.com  
- **Mamba** (recommended) or **Conda**:  
  - Mamba: https://mamba.readthedocs.io  

We do not recommend using conda, Miniconda, or Anaconda. <br>
If possible, please use Mamba instead.
---


### 2. Create the python environment via mamba
Make sure you are at the project folder:
```sh
ls
# You should see folders like:
# config
# data
# docs
# ...
```

Using mamba (recommended) that much faster:
```sh
mamba env create -f config/environment_Flow-Alert.yml
```
---

### 3. Activate the environment
```sh
mamba activate FlowAlert
```
---
