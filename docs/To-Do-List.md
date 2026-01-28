## To-Do-List
---
If you want to directly use our seismic features (either **Type A** or **Type B**) for your data, <br>
you'll need to prepare your seismic data as follows.

## 1. Run the Pre-trained Model
### 1.1 Download the Required Files
Please download the **0seismic_feature.zip** and **3trained_model.zip** from <br>
Supporting material for "Enhancing debris flow warning via machine learning feature reduction and model selection" <br>
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15020368.svg)](https://doi.org/10.5281/zenodo.15020368)

### 1.2 Unzip and Put the folder in the right format
Please <br>
Unzip **0seismic_feature.zip** and place the **European** folder into: <br>
**data/seismic_feature** <br>
Open [data_path.yaml](../config/data_path.yaml) and relace the path **"seismic_feature_dir"** <br>
```text
# seismic data-60s source in the GFZ-GLIC server
"glic_sac_dir": "/path/to/your/3Diversity-of-Debris-Flow-Footprints/data/seismic_feature"
```
Unzip **3trained_model.zip** and place the **LSTM, Random_Forest, XGBoost** folder into: <br>
**trained_model/v1-model** <br>

### 1.2 Run the Tutorial
Open and run the [inference tutorial](../demo/tutorial.ipynb)
to get started with the model and explore its functionality. <br>
You may see errors or bugs, please feel free to contact us.

## 2. Calaulate the Seismic Features
### 2.1 Configure the Path and Format for Your Raw Seismic Data
#### 2.1.1 Configure the seismic data path

Open [data_path.yaml](../config/data_path.yaml) and replace the value for **"glic_sac_dir"**:
```text
# seismic data source on the GFZ-GLIC server
"glic_sac_dir": "/storage/vast-gfz-hpc-01/project/seismic_data_qi/seismic"
```

Open [data_path.yaml](../config/data_path.yaml) and relace the path **"seismic_feature_dir"** <br>
```text
# seismic data-60s source on the GFZ-GLIC server
"glic_sac_dir": "/storage/vast-gfz-hpc-01/home/qizhou/3paper/0seismic_feature""
```


#### 2.1.2 Organize your raw data under **"glic_sac_dir"** in the following structure:
```text
glic_sac_dir/
└── European/ # continent
    └── Illgraben/ # catchment
        └── meta_data/ # sensor response
            └── 9S_2017_2020.xml # seismic_network_year1_year2.xml
            └── ... 
        └── 2020/ # year of the data
            └── ILL12/ # station
                └── EHZ/ # component
                    ├── 9S.ILL12.EHZ.2020.001.mseed # seismic_network.station.component.year.julday.data_format
                    ├── ... 
                    └── 9S.ILL12.EHZ.2020.243.mseed
```

#### 2.1.3 Add your dataset configuration to [data_path.yaml](config/data_path.yaml)
```text
# Europen
Illgraben-9J:
  path_mapping: "European/Illgraben"
  response_type: "xml"
  sensor_type: "do-not-need"
  normalization_factor: "do-not-need"
```

#### 2.1.4 Submit your job to calculate seismic features:
```bash
sbatch calculate_features_old/sbatch/9S/submitStep1_2020.sh # remember change the parameter1, 2, 3
sbatch calculate_features_old/sbatch/9S/submitStep2_2020.sh # remember change the parameter1, 2, 3
sbatch calculate_features_old/sbatch/9S/submitStep3_2020.sh # remember change the parameter1, 2, 3
```
---
If you want to write your own code to calculate seismic features

### 2.2 Check Out the Provided Scripts and Functions
**calculate seismic feature** in [Type_A_features.py](../calculate_features/s1_cal_TypeA_TypeB.py) <br>
or [prepare_feature4inference.py](../functions/data_process/prepare_feature4inference.py)
