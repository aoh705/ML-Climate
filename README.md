# Early Detection of Droughts Using ML on Soil Moisture Data

## Aimee Oh and Audrey Leong

### Instructions on Using the Code
1. `data_and_model_trial.ipynb` should be run in full.
2. `droughtpredictor.py` should be run.
3. Run `model_{avg, wc, wd}{45, 85}.ipynb` after restarting the kernel.

### Directory Format
```bash
ML-Climate/
├── README.md
├── abstract.md
├── journal.md
├── .DS_Store
├── doc/
├── etc/
└── src/
    ├── Unused Code/
    │   ├── failed_runs.ipynb
    │   ├── groundwater_extract.py
    │   └── prism_data.py
    ├── data/
    │   ├── historical_all/
    │   │   ├── SB_historical_drought_labels.csv
    │   └── projected_airtemp_2006_2099_monthly/
    │   │   └── ...
    │   └── projected_baseflow_2006_2099_monthly/
    │   │   └── ...
    │   └── projected_ev_2006_2099_monthly/
    │   │   └── ...
    │   └── projected_rainfall_2006_2099_monthly/
    │   │   └── ...
    │   └── projected_runoff_2006_2099_monthly/
    │   │   └── ...
    │   └── projected_sm_2006_2099_monthly/
    │   │   └── ...
    │   └── projected_snowfall_2006_2099_monthly/
    │   │   └── ...
    │   └── projected_snowwater_2006_2099_monthly/
    │   │   └── ...
    │   └── drought_plots/
    │   │   └── ...
    │   ├── SB_DroughtConditions.csv
    │   ├── SB_historical_drought.csv
    │   ├── drought_plots/
    │   └── results/
    │   │   ├── RF/ (random forest)
    │   │   ├── avg45/ (average climate conditons with RCP 45)
    │   │   ├── avg85/ (average climate conditons with RCP 85)
    │   │   ├── wc45/ (wet and cool climate conditons with RCP 45)
    │   │   ├── wc85/ (wet and cool climate conditons with RCP 85)
    │   │   ├── wd45/ (warm and dry climate conditons with RCP 45)
    │   │   ├── wd85/ (warm and dry climate conditons with RCP 85)
    │   │   └──  ...
    │   └── ...
```


| Name          | Type        | Description                                                                 |
| ------------- | ----------- | --------------------------------------------------------------------------- |
| `README.md`   | Markdown    | Overview of the project — includes goals, setup instructions, and usage.    |
| `abstract.md` | Markdown    | A brief summary of the project, often used for academic or conference work. |
| `journal.md`  | Markdown    | A progress log or project journal — useful for tracking weekly updates.     |
| `.DS_Store`   | System file | A macOS-generated file that stores folder view metadata. Safe to ignore.    |
| `doc/`        | Folder      | Contains detailed documentation, figures, or references.                    |
| `etc/`        | Folder      | Typically holds configuration files, constants, or metadata used in code.   |
| `src/`        | Folder      | Main source code directory — includes models, scripts, and experiments.     |
| `src/results/`| Folder      | Main results directory - includes model performance metrics and predictions based on climate and emission type |
| `src/data/`   | Folder      | Main data directory - includes data extracted, augmented, downloaded, or created needed for training and predicting drought values |
| `src/data/lineh_vic/`   | Folder      | Directory that holds Livneh-VIC data from 1950-2013 |
| `src/data/projected_{feature}_2006_2099_monthly/`   | Folder      | Directory that holds LOCA-VIC data from 2006-2099 |
| `src/data/SB_DroughtConditions.csv`   | File        | CSV file with USDM drought labels from 2000 onwards |
| `src/data/SB_historical_drought.csv`   | File        | CSV file with dry and wet conditions from 1895-2000 |
| `src/data/drought_plots/`   | Folder    | Main plot directory |
| `src/data/Unused Code/`   | Folder        | Main folder with code and notebooks not used for final results |
| `src/data/final_historical.csv`   | File    | The main .csv file with all data for training model (years 1950-2024) |
| `src/data/future_all_df_{climate model abbreviation}_{rcp level}.csv` (i.e. `future_all_df_{wd}_{45}.csv`)  | File    | The main .csv file with all data for regressive models per emission and climate model pair |
| `src/data/SB_drought_labels.csv`   | File    | The main .csv file with all historical drought label data with data column in YYYYMMDD |
| `src/data/SB_monthly_drought_labels.csv`   | File    | The main .csv file with all historical drought label data with data column in YYYY-MM-DD |
| `src/data/treefuser_historical_predictions.csv`   | File    | The main .csv file with all historical predictions from the treefuser model |
| `src/data/data_and_model_trial.ipynb`   | File    | The main python notebook with data processing, initial model processing, and random forest autoregressor |
| `src/data/graph.ipynb`   | File    | The main python notebook with plotting the new predictions and result analysis |
| `src/data/model_{climatemodel + RCP level}.ipynb` (i.e. `model_avg45.ipynb`)   | File    | The main python notebook using the drought prediction TreeFFuser AutoRegressor to predict for future data recursively |
| `src/data/droughtpredictor.py`  | File    | The main python script with `DroughtPredictor` class that predicts future D0-D4 values and D4 classification |
