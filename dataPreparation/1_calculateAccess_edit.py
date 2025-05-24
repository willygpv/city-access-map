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

# Filter to Barcelona, Spain specifically
barcelona_candidates = df_poi_per_pop[
    df_poi_per_pop['UC_NM_CTR'].str.contains('Barcelona', case=False, na=False) &
    df_poi_per_pop['UC_NM_CTR'].str.contains('Spain', case=False, na=False)
]

if len(barcelona_candidates) == 0:
    print("ERROR: Barcelona, Spain not found in the dataset!")
    print("Available cities containing 'Barcelona':")
    barcelona_any = df_poi_per_pop[df_poi_per_pop['UC_NM_CTR'].str.contains('Barcelona', case=False, na=False)]
    for city in barcelona_any['UC_NM_CTR'].unique():
        print(f"  - {city}")
    sys.exit(1)
elif len(barcelona_candidates) > 1:
    print("Multiple Barcelona candidates found:")
    for idx, row in barcelona_candidates.iterrows():
        print(f"  - {row['UC_NM_CTR']} (ID: {row['ID_HDC_G0']}, POIs: {row['count']}, Area: {row['AREA']:.1f})")
    # Take the one with most POIs
    barcelona_id = barcelona_candidates.loc[barcelona_candidates['count'].idxmax(), 'ID_HDC_G0']
    barcelona_name = barcelona_candidates.loc[barcelona_candidates['count'].idxmax(), 'UC_NM_CTR']
    print(f"Selected: {barcelona_name}")
else:
    barcelona_id = barcelona_candidates.iloc[0]['ID_HDC_G0']
    barcelona_name = barcelona_candidates.iloc[0]['UC_NM_CTR']
    print(f"Found Barcelona: {barcelona_name}")

# Filter to only Barcelona
df_keep = df[df['ID_HDC_G0'] == barcelona_id]
print(f"Filtered to Barcelona with {len(df_keep)} POIs")
print(f"Barcelona stats: Population: {barcelona_candidates.iloc[0]['P15']:.0f}, Area: {barcelona_candidates.iloc[0]['AREA']:.1f} km²")

def create_access_gdf(pois=None, network=None, maxdist=1000):
    """
    Computes walking distances from each street intersection to each of the POI categories
    """
    if pois is None or network is None:
        raise ValueError("Both pois and network must be provided")
    
    try:
        print("Initialize network POIs")
        
        cat_list_str = list(pois.groupby(['category']).mean().reset_index()['category'])
        print(f"Processing {len(cat_list_str)} categories")
        
        # Initialize accessibility DataFrame with network nodes
        accessibility = pd.DataFrame(index=network.node_ids)
        
        print("Calculating category distances")
        # Calculate distances for each category
        for cat in cat_list_str:
            pois_subset = pois[pois['category'] == cat]
            if len(pois_subset) == 0:
                continue
                
            try:
                network.set_pois(
                    category=cat, 
                    maxdist=maxdist, 
                    maxitems=len(pois_subset),
                    x_col=pois_subset['lon'], 
                    y_col=pois_subset['lat']
                )
                accessibility[str(cat)] = network.nearest_pois(distance=maxdist, category=cat)
            except Exception as e:
                print(f"Warning: Could not process category {cat}: {e}")
                continue

        print("Cleaning up the output, adding metadata")
        # Merge accessibility values with network nodes geodataframe
        access = pd.merge(
            accessibility.reset_index(),
            network.nodes_df.reset_index(),
            on='id'
        )
        
        # Add metadata (using .iloc[0] to be safer than .unique()[0])
        if len(pois) > 0:
            access["ID_HDC_G0"] = pois["ID_HDC_G0"].iloc[0]
            access["CTR_MN_NM"] = pois["CTR_MN_NM"].iloc[0] 
            access["UC_NM_MN"] = pois["UC_NM_MN"].iloc[0]
            access["P15"] = pois["P15"].iloc[0]
            access["AREA"] = pois["AREA"].iloc[0]
            access["UC_NM_CTR"] = pois["UC_NM_CTR"].iloc[0]
        
        # Convert to geodataframe
        access = gpd.GeoDataFrame(
            access, 
            geometry=gpd.points_from_xy(access.x, access.y)
        )
        # Set CRS using modern format
        access.crs = 'EPSG:4326'
        
        # Drop NaNs
        access = access.dropna()
        
        return access
        
    except Exception as e:
        print(f"Error in create_access_gdf: {e}")
        return None

print("Beginning main processing loop...")

# Get list of already processed cities
processed_dir = Path("data/processed/access/")
if processed_dir.exists():
    p_cities = [city.stem for city in processed_dir.glob("*.csv")]
else:
    p_cities = []

# Check if Barcelona is already processed
barcelona_output = f"data/processed/access/{barcelona_id}.csv"
if Path(barcelona_output).exists():
    print(f"Barcelona already processed! Results available at: {barcelona_output}")
    sys.exit(0)

print("Processing Barcelona...")

# Process Barcelona
failed_cities = []
city = barcelona_id
try:
    city_name = df_keep[df_keep["ID_HDC_G0"] == city]["UC_NM_CTR"].iloc[0]
    print(f"Calculating accessibility for {city_name}")
    
    # Subset POIs for Barcelona
    pois = df_keep[df_keep["ID_HDC_G0"] == city].copy()
    pois = gpd.GeoDataFrame(
        pois, geometry=gpd.points_from_xy(pois.lon, pois.lat)
    )
    
    print(f"Found {len(pois)} POIs in Barcelona")
    print("POI categories:")
    for cat, count in pois['category'].value_counts().items():
        print(f"  - {cat}: {count}")
    
    # Get boundary coordinates of Barcelona
    bounds = pois.total_bounds
    lng_min, lat_min, lng_max, lat_max = bounds
    
    print(f"Barcelona bounds: {lat_min:.4f}°N to {lat_max:.4f}°N, {lng_min:.4f}°E to {lng_max:.4f}°E")
    
    # Add small buffer to ensure we capture the full network
    buffer = 0.01  # approximately 1km
    lng_min -= buffer
    lat_min -= buffer  
    lng_max += buffer
    lat_max += buffer
    
    # Get pedestrian network with error handling
    try:
        print("Downloading OpenStreetMap network for Barcelona...")
        network = osm.pdna_network_from_bbox(
            lat_min, lng_min, lat_max, lng_max, 
            network_type='walk'
        )
        
        if network is None or len(network.node_ids) == 0:
            print(f"Warning: No network found for {city_name}")
            print("Barcelona analysis failed: No pedestrian network available")
        else:
            print(f"Network loaded: {len(network.node_ids)} nodes, {len(network.edges_df)} edges")
            
    except Exception as e:
        print(f"Error getting network for {city_name}: {e}")
        print("Barcelona analysis failed due to network download error")
        sys.exit(1)
    
    # Calculate accessibility
    print("Calculating accessibility metrics...")
    access = create_access_gdf(pois=pois, network=network, maxdist=5000)
    
    if access is not None and len(access) > 0:
        # Save results
        output_path = f"data/processed/access/{city}.csv"
        access.to_csv(output_path, index=False)
        print(f"SUCCESS! Saved accessibility data for {city_name}")
        print(f"Results: {len(access)} network nodes analyzed")
        print(f"Output file: {output_path}")
        
        # Print summary statistics
        numeric_cols = access.select_dtypes(include=[np.number]).columns
        accessibility_cols = [col for col in numeric_cols if col not in ['id', 'x', 'y', 'ID_HDC_G0', 'P15', 'AREA']]
        
        if accessibility_cols:
            print("\nAccessibility summary (average distances in meters):")
            for col in accessibility_cols:
                avg_dist = access[col].mean()
                print(f"  - {col}: {avg_dist:.0f}m")
        
    else:
        print(f"ERROR: No accessibility data generated for {city_name}")
        sys.exit(1)
        
except Exception as e:
    print(f"Error processing Barcelona: {e}")
    sys.exit(1)

print("\nBarcelona accessibility analysis complete!")