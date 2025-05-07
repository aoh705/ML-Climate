# ML-Climate

 Updated Abstract: 
Our project focuses on predicting drought in the Southern Californian county, San Bernaradino, using the U.S. Drought Monitor's (USDM) scale of drought categories (D0-no drought to D4-extreme drought). Our process for this will include gathering both meteorological and hydrological data that could serve as drought indicators, along with existing drought labels for the time period of 1950-2024. Currently, those variables include precipitation, rainfall, snowfall, snow water equivalent, baseflow, runoff, air temperature, evapotranspiration, 

We plan to train a model 


The frequency of prolonged periods of rought in California has led to the U.S. Drought Monitor (USDM) 


Previoust Project Proposal:
Based on the U.S. Drought Monitor (USDM) scale of drought categories (D0 to D4), we would like to predict worsening or improving drought conditions in southern and central California. A few examples would be 1) taking today’s drought status in a specific county, and having a model predict whether the drought will escalate within the next 4 weeks, or 2) finding areas where no drought is and see if it will become a drought. Currently, we have data for the following categories, all within California, across various time spans: temperature and precipitation maps, the agricultural land use, soil moisture data, reservoir conditions, and groundwater conditions. These are all up to date, many of them with data as recent as December 2024. We also have data from the NASA GRACE mission, where satellites track the movement of water (how much water is gained or lost each month, especially with respect to the river basins of California). This data is from the 2000s and 2010s, so while we’re unsure how we’ll incorporate historical data, we are happy to have it on hand. For our model, while we do have the data and can build something to simply predict and forecast, we also have an interest in the causal relationships between all the factors we’ve looks into. For example, how much do reservoirs mediate lack of precipitation in drought conditions, does temperature affect droughts directly (or is it indirectly affecting droughts through soil moisture), and how immediate are these effects? Our current thoughts are that precipitation and additional factors have to be low long-term to create droughts, but that is what we’re looking to find out, and see which factors should be focused the most on in the future. 

Data: 
- United States drought monitor https://droughtmonitor.unl.edu/
- CA specific drought monitor by county https://www.drought.gov/states/california/county/San%20Bernardino
corresponding temperature and precipitation maps available with tabs on 7/30/60 day views, agriculture available (locations of cattle, sheep, hay farms)
- california water watch Soil Moisture data https://cww.water.ca.gov/maps?tab=soil (most recent 12/31/24)
data on precipitation, temperature, reservoir conditions, streamflow, groundwater, snowpack, soil moisture/vegetation conditions
- NASA GRACE mission launched 2002: satellites to see movement of water, how much water gained or lost each month, california specific with river basins outlined across central valley (bc agriculture) 
- CA farmland map https://maps-cadoc.opendata.arcgis.com/datasets/cadoc::california-important-farmland-most-recent/explore?location=34.010574%2C-118.365300%2C14.80
