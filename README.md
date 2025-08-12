## 📢 Welcome to **Flow-Alert**
<img src="docs/image/logo.png" alt="Alt Text" width="200"/>
---

If you're interested in leveraging **machine learning** and **seismic signals** for **channelized flow early warning**, <br>
including, but not limited to, **debris flows**, **glacial lake outburst floods (GLOFs)**, and **lahars**, <br>
you've come to the right place! <br>
Check out our repository to get started.

### 🛠️ 0. Major Changes for v1.1
Retested the LSTM workflow. <br>
You may encounter bugs in the RF or XGBoost models<br>
if you do, please report them to us.

### 📁 1. Repository Structure
```bash
Flow-Alert
├── calculate_features   # Convert raw seismic data into features
├── config               # Configuration files
├── create_labels        # Create the label for time stamps
├── data_input           # Data input
├── data_output          # Data output
├── functions            # Core functions and scripts
├── trained_model        # Pre-trained models
    └── feature_imp      # Seismic geature weight
    └── v1model          # Model version 1
```

### 💪 2. Contributors <br>
**[Qi Zhou](https://github.com/Qi-Zhou-Geo)** <br>
qi.zhou@gfz.de or qi.zhou.geo@gmail.com <br>

**[Kshitij Kar](https://github.com/Kshitij301199)** <br>
kshitij.kar@gfz-potsdam.de <br>
