import requests
import pandas as pd
from io import StringIO

# Parameters
county_code = "06071"  # San Bernardino County
state_code = "CA"
start_date = "1950-01-01"
end_date = "2024-12-31"

url = f"https://waterdata.usgs.gov/ca/nwis/dv?referred_module=gw&county_cd=06071&group_key=NONE&sitefile_output_format=html_table&column_name=agency_cd&column_name=site_no&column_name=station_nm&range_selection=date_range&begin_date={start_date}&end_date={end_date}&format=rdb&date_format=YYYY-MM-DD&rdb_compression=value&list_of_search_criteria=county_cd%2Crealtime_parameter_selection"

import requests
import pandas as pd
from io import StringIO

# Fetch and clean data
response = requests.get(url)
data = '\n'.join([line for line in response.text.splitlines() 
                  if line and not line.startswith("#")])

# Define explicit column headers (from RDB format description)
columns = [
    'agency_cd', 'site_no', 'datetime', 
    'parameter_value', 'qualification_cd'
]

# Load data with error handling
df = pd.read_csv(
    StringIO(data),
    sep='\t',
    comment='#',
    header=None,
    names=columns,
    on_bad_lines='skip',  # Skip malformed rows
    usecols=range(5),     # Force 5 columns
    dtype={'qualification_cd': 'category'}
)

# Filter valid data rows (remove metadata)
df = df[df['agency_cd'] == 'USGS']
df['datetime'] = pd.to_datetime(df['datetime'])

print(df.head())

# Save to CSV (optional)
df.to_csv("SB_groundwater.csv", index=False)