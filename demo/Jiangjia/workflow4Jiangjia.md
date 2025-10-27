# Workflow for Jiangjia Data

This document outlines the steps to process Jiangjia seismic data and run our Flow Alert model.

---

## Step 1 – Restructure the Raw Data

Restructure your raw data following the workflow. <br>
It is important to maintain the data structure below:

```text
glic_sac_dir/ # data path
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
You may write your own code to do this. <br>
```python
from obspy import read
st = read("path/to/file.mseed")
st = st[0] # if you have multiple traces, please merge it.
st.stats.network = "your_network"
st.stats.station = "your_station"
st.write("path/file.mseed", format="MSEED")
```

# Step 2 – Add Meta Information
Add one row in the catalog file:
[Flow_Bench_Catalog_work_v0dot8-10.txt](data/event_catalog/Flow_Bench_Catalog_work_v0dot8-10.txt), <br>
This will include your sensor metadata and ensure correct sensor response processing.

# Step 3 – Add Zeros and Poles
If your sensor is **SmartSolo IGU-16HR 3C**, you can directly use our pre-defined zeros and poles. <br>
If not, add your sensor information in [seismic_data_processing.py](functions/seismic/seismic_data_processing.py)
```python
# After line 149 in func. manually_remove_sensor_response
your_zeros_and_poles = {
    # fill in your sensor poles and zeros here
}

```
If you have the .xml file, you can copy this file in the **meta_data** folder in Step 1.


# Step 4 - Explore Seismic Features
You can refer to the example shell scripts:
[submitStep1_2013.sh](calculate_features/sbatch/9J/submitStep1_2013.sh) <br>
[submitStep3_2013.sh](calculate_features/sbatch/9J/submitStep3_2013.sh) <br>
These demonstrate how to prepare shell files for calculating seismic features. <br>
If you want to calculate the newtwork feature, you may check and read the  <br>
[2cal_TypeB_network.py](calculate_features/2cal_TypeB_network.py),
then you can preapre the shell file for calculate.

Tip: <br>
If running on your PC instead of a SRUM cluster, you may need to modify the shell scripts accordingly.   <br>
ChatGPT can help adapt them for local execution.  <br>

# Step 5 - Plot the Waveform and PSD
I do not have specific code for this step, <br>
you may check the func. in [visualize_seismic.py](functions/visualize/visualize_seismic.py).


# Step 6 - Run Flow Alert model
Run the pre-trained Flow Alert model using the demo script:<br>
[Jiangjia_main.py](demo/Jiangjia/Jiangjia_main.py).
