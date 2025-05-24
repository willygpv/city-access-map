import os
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from traffic_light_accessibility import calculate_traffic_light_accessibility, compare_accessibility

# Load your existing POI and urban center data
# This follows the pattern from your 1_calculateAccess.py file
# download POIs at https://github.com/MorbZ/OsmPoisPbf/ using uac_filter.txt
# java -jar osmpois.jar --filterFile uac_filter.txt --printHeader planet.osm.pbf
df = pd.read_csv("data/raw/poi/poi.csv", sep='|')
df = df[df.category.str.isnumeric()==True]
df['category'] = df['category'].astype(float)
df = df.rename(columns={"category": "poi_type_id"})

poi_types = pd.read_excel("data/raw/poi/poi_code_name_mapper.xlsx")
poi_types = poi_types.replace(" ", np.NaN).dropna()

df = df.merge(poi_types, on="poi_type_id")

# Load urban center data
uc = pd.read_csv("data/raw/GHS_STAT_UCDB2015MT_GLOBE_R2019A/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.csv", 
                encoding="ISO-8859-1", engine='python')
uc = uc[['ID_HDC_G0', "CTR_MN_NM", "UC_NM_MN", "P15", "AREA"]].dropna()
uc["UC_NM_CTR"] = uc["UC_NM_MN"] + ", " + uc["CTR_MN_NM"]

# Create GeoDataFrame of POIs
gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.lon, df.lat)).set_crs(4326)

# Load geometries of urban centers
geo_uc = gpd.read_file("data/GHS_STAT_UCDB2015MT_GLOBE_R2019A/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.gpkg")
geo_uc = geo_uc[['ID_HDC_G0', "CTR_MN_NM", "UC_NM_MN", "P15", "AREA", "geometry"]].dropna()
geo_uc["UC_NM_CTR"] = geo_uc["UC_NM_MN"] + ", " + geo_uc["CTR_MN_NM"]

# Join POIs with urban centers
gdf = gdf.sjoin(geo_uc, how="inner")
df_keep = gdf

# Either process all cities or pick a specific one to test
# For testing, start with just one city
city_id = df_keep["ID_HDC_G0"].iloc[0]  # Use first city for testing

# Run the traffic light accessibility calculation for this city
access_with_signals = calculate_traffic_light_accessibility(city_id, df_keep)

# Compare with existing accessibility data (if you have it)
comparison, summary = compare_accessibility(city_id)

print(f"Summary of traffic light impact for {city_id}:")
print(summary)

# Optional: save the comparison results
comparison.to_csv(f"data/processed/comparisons/{city_id}_comparison.csv")
summary.to_csv(f"data/processed/comparisons/{city_id}_summary.csv")