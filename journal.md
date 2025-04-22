4/21 Clarifying Definitions
The California Department of Water Resources’s Climate Change Technical Advisory Group reduced the larger ensemble of 32 GCMs to a more manageable set of 10 GCMs as being most suitable for California water resource climate change studies. For some study teams and users of California’s Fourth Climate Change Assessment data, even the previously identified set of 10 GCMs was too much data. Accordingly, 4 of those 10 GCMs were identified whose project future climate can be described as producing:

A “warmer/drier” simulation (HadGEM2-ES)
An “average” simulation (CanESM2)
A “cooler/wetter” simulation (CNRM-CM5)
A “dissimilar” simulation that is most unlike the other three, to produce maximal coverage of possible future climate conditions (MIROC5)
Simulations produced with all climate models show substantial future warming; the “cooler/wetter” CNRM-CM5 simulation just shows less warming than the other models. The GCMs projections hosted on Cal-Adapt were generated for the periods 2006 to 2100 (future climate) and 1950 to 2005 (modeled historical climate).


4/20 Update: 
- Data gathered for the mid and late century prediction (2006-2099) from the Cal-Adapt climate models for air temperature, rainfall, snowfall, and baseflow (still searching for dew point if possible). Ensemble of model predictions selected, specifically CanESM2 (average simulation), CNRM-CM5 (cool/wet simulation), HadGEM2-ES (warm/dry simulation), MIROC5 (dissimilar to all other simulations to cover all possible climates) at RCP 4.5 (medium emissions, peaking at 2040) and 8.5 (zero mitigation state emission rate rises throughout end of 21st century), all projections at monthly increments. This is intended for later testing the predictive model. After reading some more papers on drought types and definitions, (https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023WR034994 and https://www.sciencedirect.com/science/article/pii/S0048969724002675), I'm defining the specifics of flash vs meteorological vs hydrological droughts (usually distinguished by length of drought and characteristics temperature and radiation, though we don't include radiation data in our own model at the moment). While San Bernardino was chosen because it has multiple drought characteristics across the county (D1-D4), the data for precipitation, etc are not spatially divided, so categorizing the drought labels (flash/meteorological/hydrological) in the historical data is currently processing so we can have those predictions for the future as well. These characterizations, which can be determined once we have the initial D0-D4 predictions, will guide the interpretation of future predictions in terms of their impact, especially agriculturally. 

A version of the current predictive model that includes groundwater/baseflow data will be pushed by 4/21. 

4/14 Audrey and Aimee To Dos:
1. download more historical PRISM data (before 2019 (2000) and groundwater level data for San Bernardino, CA ([link to data](https://nwis.waterdata.usgs.gov/ca/nwis/dv?county_cd=06071&format=csv&site_status=all&referred_module=gw&begin_date=2000-01-01&end_date=2025-04-13&list_of_search_criteria=county_cd%2Crealtime_parameter_selection&range_selection=date_range))
2. download yearly predictive temperature and precipitation values yearly for San Bernardino, CA
3. Work out hypotheses for future predictions (day context, weekly context, yearly context)
4. use DoWhy for coding up DAG and finding possible causational features for drought in San Bernardino, CA

4/10 and 4/13:
Trained LSTM with PRISM data, San Bernardino County drought data, and water shortage data. Get accuracy of 0.9 at most. Next steps should be adding data for temperature, humidity, etc. to add features to predict drought. Commits have been made in `data.py` for retrieving PRISM data and `model.ipynb` for examining, pre-processing data and training data.

3/28/2024 Progress Report
Busy week with big homeworks and midterms:
1. data is hard to download/examine due to government data (may need to download in batches)
2. [Light Gradient Boosting Machine (LightGBM)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024GL111134?af=R#:~:text=Of%20which%2C%20soil%20moisture%20is,.%2C%202019%2C%202023) is also a viable model --> we probably will run multiple models and compare accuracy. We tentatively propose also discovering causation of flash drought after prediction.
3. In terms of prediction, we want to take one year of data to predict about one week forecast and a longer previous period of time to predict two weeks of forecast if possible.

3/9/25 Aimee expanding on models to use and related works examined
The first models we could think about fitting on the data we have found (other than shortn term prediction and LSTM):
Related Works and models we could think about fitting based on works:
1). Drought in Australia: https://www.nature.com/articles/s41598-024-70406-6 :
- Soft computing models + AI: DT, GLM, SVM, ANN, DL, and RF
- Decision trees, support vector machine, random forest
- Artificial neural network
2). Missouri/Columbia basins: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023WR036973?af=R :
- Build complex relationships between predictor data sets and multi-layer SM measurements from the Montana Mesonet
- Predicted with 1-2 week forecast lead times
- Measured accuracy with 2017 Montana flash drought
- Used random forest and light gradient boosting machine regression
3). Tanzania: https://www.sciencedirect.com/science/article/pii/S2214581824001423#sec0010 :
- LSTM

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

