```sh
# Last Update: Last modified: 2026-07-26T12:01:37
# Author: Qi Zhou
```

## Major changes from 1.2 to 1.3
Compared with previous versions, including:
- version 1.0 (https://doi.org/10.5281/zenodo.15020368)
- version 1.1 (https://doi.org/10.5281/zenodo.16811121)
- version 1.2 (https://doi.org/10.5281/zenodo.16893616)
- the latest version 1.3 (https://doi.org/10.5281/zenodo.18324322) includes the following major changes:

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
