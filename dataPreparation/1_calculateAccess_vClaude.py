# %matplotlib inline
import glob
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import sys
import dask.dataframe as dd
import matplotlib.pyplot as plt
from pathlib import Path

from pandana.loaders import osm
from pandana.loaders.pandash5 import network_to_pandas_hdf5
import pandana as pdna
import geopandas as gpd
from shapely.geometry import Point
from shapely import wkt
import osmnx as ox
from random import sample
from tqdm import tqdm

if len(sys.argv) > 1:
    working_dir = Path(sys.argv[1]).resolve()
else:
    # get the directory containing this script (or cwd), then go one level up
    base_dir = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
    working_dir = base_dir.parent

print(f"Working directory: {working_dir}")
os.chdir(working_dir)


# Create necessary directories if they don't exist
os.makedirs("data/processed/access", exist_ok=True)

def check_file_exists(filepath, description=""):
    """Check if a file exists and provide helpful error message if not"""
    if not Path(filepath).exists():
        print(f"ERROR: {description} file not found: {filepath}")
        print("Please ensure the file exists before running the script.")
        return False
    return True

print("Loading data...")

# Check if required files exist
required_files = [
    (r"downloads/planet-250512.csv", "POI data"),
    (r"dataPreparation/poi_code_name_mapper.xlsx", "POI type mapper"),
    (r"downloads/GHS_STAT_UCDB2015MT_GLOBE_R2019A/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.csv", "Urban centers data"),
    (r"downloads/GHS_STAT_UCDB2015MT_GLOBE_R2019A/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.gpkg", "Urban centers geometry")
]

for filepath, description in required_files:
    if not check_file_exists(filepath, description):
        sys.exit(1)

# Load POI data
try:
    df = pd.read_csv("downloads/planet-250512.csv", sep='|')
    df = df[df.category.str.isnumeric() == True]
    df['category'] = df['category'].astype(float)
    df = df.rename(columns={"category": "poi_type_id"})
    print(f"Loaded {len(df)} POIs")
except Exception as e:
    print(f"Error loading POI data: {e}")
    sys.exit(1)

# Load POI type mapper
try:
    poi_types = pd.read_excel("dataPreparation/poi_code_name_mapper.xlsx")
    poi_types = poi_types.replace(" ", np.nan).dropna()
    df = df.merge(poi_types, on="poi_type_id")
    print(f"Merged POI types, {len(df)} POIs remaining")
except Exception as e:
    print(f"Error loading POI type mapper: {e}")
    sys.exit(1)

# Load urban centers data
try:
    uc = pd.read_csv("downloads/GHS_STAT_UCDB2015MT_GLOBE_R2019A/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.csv", 
                     encoding="ISO-8859-1", engine='python')
    print(f"Loaded {len(uc)} urban centers")
except Exception as e:
    print(f"Error loading urban centers data: {e}")
    sys.exit(1)

print("Preparing data...")

# Process urban centers data
uc = uc[['ID_HDC_G0', "CTR_MN_NM", "UC_NM_MN", "P15", "AREA"]].dropna()
uc["UC_NM_CTR"] = uc["UC_NM_MN"] + ", " + uc["CTR_MN_NM"]

# Create GeoDataFrame from POI data
gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.lon, df.lat)).set_crs('EPSG:4326')

# Load urban centers geometry
try:
    geo_uc = gpd.read_file("downloads/GHS_STAT_UCDB2015MT_GLOBE_R2019A/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.gpkg")
    geo_uc = geo_uc[['ID_HDC_G0', "CTR_MN_NM", "UC_NM_MN", "P15", "AREA", "geometry"]].dropna()
    geo_uc["UC_NM_CTR"] = geo_uc["UC_NM_MN"] + ", " + geo_uc["CTR_MN_NM"]
    print(f"Loaded geometry for {len(geo_uc)} urban centers")
except Exception as e:
    print(f"Error loading urban centers geometry: {e}")
    sys.exit(1)

# Spatial join POIs with urban centers
gdf = gdf.sjoin(geo_uc, how="inner")
df = gdf

# Calculate POI statistics per urban center
df["count"] = 1
df_poi_per_pop = df.groupby(["ID_HDC_G0", "UC_NM_CTR"]).agg({
    "P15": "mean", 
    "count": "sum", 
    "AREA": "mean"
}).reset_index()

df_poi_per_pop["poi_per_pop"] = df_poi_per_pop["count"] / df_poi_per_pop["P15"]
df_poi_per_pop["poi_per_km2"] = df_poi_per_pop["count"] / df_poi_per_pop["AREA"]

######## USER PARAMETER ########
# For now test only Barcelona; later replace this list with any city IDs you want
TEST_CITY_IDS = [2051]
################################
# Keep only our test cities
df_keep = df[df['ID_HDC_G0'].isin(TEST_CITY_IDS)]
print(f"Filtered to {len(TEST_CITY_IDS)} test city(ies), {len(df_keep)} POIs total")


def create_access_gdf(pois=None, network=None, maxdist=1000):
    """
    Computes walking distances from each street intersection to each of the POI categories
    """
    if pois is None or network is None:
        raise ValueError("Both pois and network must be provided")
    
    try:
        print("Initialize network POIs")
        
        # Get unique categories - but for groupby, we need numeric or properly aggregatable data
        # The original code works because after merging, 'category' becomes the string category names
        # Let's get unique categories directly
        cat_list_str = list(pois['category'].unique())
        print(f"Processing {len(cat_list_str)} categories: {cat_list_str}")
        
        print("Creating dummy df")
        # Create dummy dataframe (initialize accessibility DataFrame properly)
        # This follows the original code pattern of doing a dummy run first
        accessibility = None
        # Run the dummy “1 m” pass _for every category_, so the last one seeds the structure
        for cat in cat_list_str:
            pois_subset = pois[pois['category'] == cat]
            if len(pois_subset) == 0:
                continue
            network.set_pois(category=cat, maxdist=1, maxitems=len(pois_subset),
                             x_col=pois_subset['lon'], y_col=pois_subset['lat'])
            accessibility = network.nearest_pois(distance=1, category=cat)
        
        if accessibility is None:
            raise ValueError("No valid POI categories found to initialize accessibility calculation")
        
        print("Calculating category distances")
        # Now calculate actual distances for all categories
        for cat in cat_list_str:
            print(f"  Processing category: {cat}")
            pois_subset = pois[pois['category'] == cat]
            if len(pois_subset) == 0:
                print(f"    No POIs found for category {cat}")
                continue
                
            print(f"    Found {len(pois_subset)} POIs for {cat}")
            
            try:
                network.set_pois(
                    category=cat, 
                    maxdist=maxdist, 
                    maxitems=len(pois_subset),
                    x_col=pois_subset['lon'], 
                    y_col=pois_subset['lat']
                )
                accessibility[str(cat)] = network.nearest_pois(distance=maxdist, category=cat)
                print(f"    Successfully calculated distances for {cat}")
            except Exception as e:
                print(f"    Warning: Could not process category {cat}: {e}")
                continue

        print("Cleaning up the output, adding metadata")
        # Merge accessibility values with network nodes geodataframe
        accessibility_df = accessibility.reset_index().drop(1, axis=1)
        
        access = pd.merge(  
            accessibility_df,
            network.nodes_df.reset_index(),
            on='id'
        )
        
        # Add metadata (using .unique()[0] like in original code)
        if len(pois) > 0:
            access["ID_HDC_G0"] = pois["ID_HDC_G0"].unique()[0]
            access["CTR_MN_NM"] = pois["CTR_MN_NM"].unique()[0] 
            access["UC_NM_MN"] = pois["UC_NM_MN"].unique()[0]
            access["P15"] = pois["P15"].unique()[0]
            access["AREA"] = pois["AREA"].unique()[0]
            access["UC_NM_CTR"] = pois["UC_NM_CTR"].unique()[0]
        
        # Convert to geodataframe
        access = gpd.GeoDataFrame(
            access, 
            geometry=gpd.points_from_xy(access.x, access.y)
        )
        # Set CRS using original format (for compatibility)
        access.crs = {'init': 'epsg:4326'}
        
        # Drop NaNs
        access = access.dropna()
        
        return access
        
    except Exception as e:
        print(f"Error in create_access_gdf: {e}")
        import traceback
        traceback.print_exc()
        return None

print("Beginning main processing loop...")

# Prepare processed‐cities list (so we can skip if already done)
processed_dir = Path("data/processed/access/")
if processed_dir.exists():
    p_cities = {p.stem for p in processed_dir.glob("*.csv")}
else:
    p_cities = set()

failed = []
for city in TEST_CITY_IDS:
    city_str = str(city)
    if city_str in p_cities:
        print(f"City {city} already processed; skipping.")
        continue

    city_name = df_keep[df_keep["ID_HDC_G0"] == city]["UC_NM_CTR"].iloc[0]
    print(f"\n=== Processing {city_name} (ID {city}) ===")
    try:
        # Subset POIs
        pois = df_keep[df_keep["ID_HDC_G0"] == city].copy()
        pois = gpd.GeoDataFrame(
            pois,
            geometry=gpd.points_from_xy(pois.lon, pois.lat)
        )

        # Compute bounding box
        lng_min, lat_min, lng_max, lat_max = pois.total_bounds

        # Download network
        print("Downloading OpenStreetMap network…")
        network = osm.pdna_network_from_bbox(
            lat_min, lng_min, lat_max, lng_max,
            network_type='walk'
        )

        # Calculate accessibility
        print("Calculating accessibility metrics…")
        access = create_access_gdf(pois=pois, network=network, maxdist=5000)

        # Save
        output_path = f"data/processed/access/{city}.csv"
        access.to_csv(output_path, index=False)
        print(f"Saved {output_path}")

    except Exception as e:
        print(f"ERROR processing {city_name}: {e}")
        failed.append(city)

if failed:
    print("Some cities failed:", failed)
else:
    print("All test cities processed successfully.")