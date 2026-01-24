## Prepare the Python Environment for Flow-Alert

### 1. Make sure you have Git and Mamba (or Conda)

Please install the following tools first:

- **Git**: https://git-scm.com  
- **Mamba** (recommended) or **Conda**:  
  - Mamba: https://mamba.readthedocs.io  
  - Conda: https://docs.conda.io

---

### 2. Clone the repository and enter the folder
```sh
git clone https://github.com/Qi-Zhou-Geo/Flow-Alert.git
cd Flow-Alert
```
Or download the project ZIP file from GitHub and unzip it.
---

### 3. Create the Conda environment
Using mamba (recommended) that much faster:
```sh
mamba env create -f config/Flow-Alert-env.yml
```
Or using conda:
```sh
conda env create -f config/Flow-Alert-env.yml
```
---

### 4. Activate the environment
```sh
mamba activate flow-alert
```
---

### 5. Test the core packages (optional)
```sh
python - <<EOF
import torch, xgboost, sklearn
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF

```

### 6. Have func with Flow-Alert
Please check out the tutorial for [GLOF](demo/inference_tutorial.ipynb).