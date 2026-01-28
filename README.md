## 📢 Welcome to **Flow-Alert**
<img src="docs/image/logo.png" alt="Alt Text" width="200"/>
---

If you're interested in leveraging **machine learning** and **seismic signals** for **channelized flow early warning**, <br>
including, but not limited to, **debris flows**, **glacial lake outburst floods (GLOFs)**, and **lahars**, <br>
you've come to the right place! <br>
Check out our repository to get started.

### 🛠️ 0. Major Changes for v1.3
Compared with previous versions, including:<br>
version 1.0 (https://doi.org/10.5281/zenodo.15020368), <br>
version 1.1 (https://doi.org/10.5281/zenodo.16811121), <br>
version 1.2 (https://doi.org/10.5281/zenodo.16893616), <br>
the latest version 1.3 (https://doi.org/10.5281/zenodo.18324322) includes the following major changes: <br>

**(1) Data**: The debris flow events on 2019-10-09 and 2019-10-15, recorded at the ILL12 station, are now included in the training dataset.  <br>
These events were not used in previous versions because the ILL18 station was unavailable. <br>
Earlier versions relied on a network of stations (ILL12, ILL13, and ILL18) for warning,  <br>
whereas the latest version focuses on single-station detection and classification. <br>

**(2) Labels**: Previous versions used manually labeled event timestamps, <br>
while the latest version employs STA/LTA-based event times, <br>
which are theoretically more objective. <br>
Please check [here](data/event_catalog/9S-2017-DF.txt) for details.<br>

**(3) Features**: Previous versions used all 70+ available seismic features,<br>
whereas the latest version selects 12 seismic features to train the model.<br>
Please check [feature_type_H](config/config_inference.yaml) for details.<br>

**(4) Model Structure**: This version integrates an attention mechanism layer after the LSTM, <br>
which is expected to better capture temporal dependencies in the seismic signals.<br>


### 📁 1. Repository Structure
```bash
Flow-Alert
├── calculate_features   # Convert raw seismic data into features
├── config               # Configuration files
├── data                 # Seismic data and extracted features
├── demo                 # Examples of running the case
├── docs                 # Documentation for users
├── functions            # Core functions and scripts
├── pipeline             # Train and test code
├── trained_model        # Pre-trained models
```


### 🚀 2. How to Use Our Pre-trained Models on Your Data? <br>
To get started: <br>
2.1 Check the [prepare_env.md](docs/prepare_env.md) for setting up the Python environment in your local PC <br>
2.2 Check the [tutorial](demo/tutorial.ipynb) for usage <br>
2.3 Run the [tutorial](demo/tutorial.ipynb) on your local PC or on
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Qi-Zhou-Geo/Flow-Alert/blob/main/demo/tutorial.ipynb)

### 🐛 3. Found a Bug? <br>
Feel free to open a **Pull request**, or reach out to us via email.


### ❓️4. Have Questions? <br>
4.1 Start by reading our related paper <br>
**Qi Zhou**, Hui Tang, Clément Hibert, Małgorzata Chmiel, Fabian Walter, Michael Dietze, and Jens M Turowski. <br>
"Enhancing debris flow warning via machine learning feature reduction and model selection." <br>
**_Journal of Geophysical Research: Earth Surface_**, 129, e2024JF008094. <br>
[Click here for the manuscript](https://doi.org/10.1029/2024JF008094) <br>

**Qi Zhou**, Hui Tang, Michael Dietze, Fabian Walter, Dongri Song, Yan Yan, Shuai Li, and Jens M. Turowski. <br>
"Similarity of Debris Flows in Seismic Records." <br>
[Click here for the preprint](https://doi.org/10.22541/essoar.175676964.46168374/v1) <br>


If you still have questions, feel free to contact the project contributors.

4.2 Or reach out to our research groups <br>
[Hazards and Surface Processes Research Group and Digital Earth Lab](https://www.gfz.de/en/section/earth-surface-process-modelling/topics/hazards-and-surface-processes) <br> 
Led by Dr. Hui [Tang](https://www.gfz.de/en/staff/hui.tang/sec47) <br>
[Physical Earth Surface Modelling Lab](https://www.gfz.de/en/staff/jens.turowski/sec46) <br>
Led by Dr. Jens [Turowski](https://www.gfz.de/en/staff/jens.turowski/sec46).


### 💪 5. Contributors <br>
**[Qi Zhou](https://github.com/Qi-Zhou-Geo)** <br>
qi.zhou@gfz.de or qi.zhou.geo@gmail.com <br>

**[Kshitij Kar](https://github.com/Kshitij301199)** <br>
kshitij.kar@gfz-potsdam.de <br>
