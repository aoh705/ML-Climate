# Early Detection of Droughts Using ML on Soil Moisture Data

## Aimee Oh and Audrey Leong

### Instructions on Using the Code
1. `data_and_model_trial.ipynb` should be run in full.
2. `droughtpredictor.py` should be run in terminal as `python3 droughtpredictor.py`.
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
    │   │   ├── RF/
    │   │   ├── avg45/
    │   │   ├── avg85/
    │   │   ├── wc45/
    │   │   ├── wc85/
    │   │   ├── wd45/
    │   │   ├── wd85/
    │   │   └──  ...
    │   └── ...
    ├── notebooks/
    │   ├── eda/
    │   └── experiments/
    └── utils/
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

### Instructions on Using the Code
1. `data_and_model_trial.ipynb` should be run in full.
2. `droughtpredictor.py` should be run in terminal as `python3 droughtpredictor.py`.
3. Run `model_<avg, wc, wd><45, 85>.ipynb` after restarting the kernel.
4. `graph.ipynb` should be run in full. 