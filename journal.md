3/5/2024 Audrey 
Responding to the feedback received on 2/28 on the data details, and what models we're considering to use first. 

| Data Source | Geospatial | Time Range | Time Intervals | Units |
|-------------|------------|------------|---------------|-------|
| [USDM Drought](https://droughtmonitor.unl.edu/DmData/DataTables.aspx?county,06029) | California, per county | 1/4/2000 - 2/28/2025 | Weekly measurements | Categorical (D0-D4) or cumulative area (mi²), percent area, population, population percentage in D0-D4 categories |
| [gridMET (via ClimateEngine)](https://app.climateengine.org/climateEngine) ([Dataset Link](https://www.sciencebase.gov/catalog/item/6372cd09d34ed907bf6c6ab1)) | Entire United States, high-spatial resolution (~4-km, 1/24th degree) | 1979 - present | Daily | Precipitation (mm), mean/max/min temperature, standardized precipitation evapotranspiration index (SPEI), specific/relative humidity, evaporative demand drought index (EDDI) |
| [California Water Watch (Soil Moisture)](https://cww.water.ca.gov/maps?tab=soil) | Entire California, 3km resolution | Unclear - 12/31/24 | Every 6 hours | Top 100 cm of soil moisture percentile (Geotiff images, no spreadsheet values) |
| [California Periodic Groundwater Data](https://data.cnra.ca.gov/dataset/periodic-groundwater-level-measurements/resource/bfa9f262-24a1-45bd-8dc8-138bc8107266) | California, based on well location | 1900 - present | Weekly | Reference point to elevation in feet |

Current challenges with data: 
- I spent a good 2 hours looking through the data sources for additional details. Different government agencies have different access permission granted to the public, and the datasets are not well documented (nor located). I have narrowed our sources down to mainly the USDM Drought Monitor and gridMET, which provide labels (drought index) and features (precipitation, temperature, evapotranspiration, humidity). Groundwater data is most commonly found in online map viewers, and downloadable as geoTIFF files (which can be reformatted as pixel = label and the rest of the data = features, so it's alright). The difficulty is downloading from the interactive map viewers, selecting time periods and seeing if the data is allowed to be downloaded (or doesn't exist, or encounters errors).
- Beyond geoTIFF format, a lot of files come in NetCDF4 format, which with some python script testing, seems transformable for model use. 
- I also looked into other data to gather: crop health since it can be an indicator of drought (though in this case, more of an effect than a cause), and because California is agriculturally active, especially in Central California. I've also looked into the California Water Plan to see how irrigation is planned throughout the state and downstreamed to communities/agriculture (this was hard to find data beyond a yearly report on the plan itself, not results). Lastly, I think looking at El Niño-Southern Oscillation (ENSO) Index
data for climate drivers beyond just local climate would be very interesting. For now, I've categorized these three as beyond the scope of the project, since it's complicating the cause/effect model.

| Feature Category       | Features |
|------------------------|----------------------------------------------------------------|
| Meteorological   | Precipitation (mm), Temperature (mean/max/min), Humidity, Evaporative Demand (EDDI), Standardized Precipitation Evapotranspiration Index (SPEI) |
| Hydrological      | Soil Moisture (percentile), Groundwater Levels (ft), Reservoir Storage (%) |
| Vegetation & Land Use | NDVI (Vegetation Index), Agricultural Land Cover |
| Drought History  | Previous USDM Drought Category (D0-D4) |
| Climate Patterns   | ENSO Index, Long-term Precipitation Anomalies |

Data cleaning: 
- Because the data is on the time scale of several years, and the drought index is at a weekly measuring rate, we're considering putting all meterological data on a weekly basis, rather than interpolating to a daily one. (Aggregate via weekly averages or sums)
- if we do use groundwater and ENSO data, break down the monthly data into weekly
- Processing county level, station based, and 4km resolution data: unclear. 

Model 
Eventually, we want to build up to a causality model, but we want to start incrementally with our predictions.
1) Short term prediction:
- take 4 consecutive weeks of data, and predict the next 4 weeks (ideally multiclass according to USDM drought index) using random forest models, using mainly meterological data
2) move to LSTM to look at longer time dependencies (historical droughts, ENSO if included)


Causal Questions (I'm not sure if this type of variable isolation is the way to start) 
If precipitation increases by X mm, how much would drought severity improve?
If reservoir levels drop by 10%, what is the expected effect on drought worsening?
What would happen if soil moisture dropped 20% but temperature stayed stable?

