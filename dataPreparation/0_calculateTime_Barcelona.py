# %matplotlib inline
import glob
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import dask.dataframe as dd
import matplotlib.pyplot as plt
import time # Import the time module for performance measurement

# Import specific pandana loaders for direct control over network creation
# from pandana.loaders import osm # We will replace this with direct OSMnx graph conversion
from pandana.loaders.pandash5 import network_to_pandas_hdf5
import pandana as pdna
import geopandas as gpd
from shapely.geometry import Point
from shapely import wkt
import osmnx as ox # Explicitly import osmnx
from random import sample
from tqdm import tqdm # For progress bars

# --- Configuration Constants ---
WALKING_SPEED_MPS = 1.4 # meters per second, a typical average walking speed
TRAFFIC_LIGHT_DELAY_SECONDS = 30 # seconds, estimated average delay at a traffic light
MAX_DISTANCE = 1000 # Maximum distance in meters for accessibility queries

# Directory to save/load precomputed pandana networks
PANDANA_NETWORK_CACHE_DIR = "data/processed/pandana_networks"
os.makedirs(PANDANA_NETWORK_CACHE_DIR, exist_ok=True) # Ensure the directory exists

# Set working directory (adjust as needed)
# Ensure this directory exists and contains your data
directory = os.chdir(r'C:\Users\Guillermo\Documents\CSH\15_minute_city\city-access-map')

print("loading data")
# download POIs at https://github.com/MorbZ/OsmPoisPbf/ using uac_filter.txt
# java -jar osmpois.jar --filterFile uac_filter.txt --printHeader planet.osm.pbf
df = pd.read_csv("downloads/planet-250512.csv", sep='|')
df = df[df.category.str.isnumeric()==True]
df['category'] = df['category'].astype(float)
df = df.rename(columns={"category": "poi_type_id"})

poi_types = pd.read_excel("dataPreparation/poi_code_name_mapper.xlsx")
poi_types = poi_types.replace(" ", np.NaN).dropna()

df = df.merge(poi_types, on="poi_type_id")

# download this data at http://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_STAT_UCDB2015MT_GLOBE_R2019A/V1-2/
uc = pd.read_csv("downloads/GHS_STAT_UCDB2015MT_GLOBE_R2019A/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.csv", encoding = "ISO-8859-1", engine='python')

print("preparing data")
uc = uc[['ID_HDC_G0', "CTR_MN_NM", "UC_NM_MN", "P15", "AREA"]].dropna()
# city + country
uc["UC_NM_CTR"] = uc["UC_NM_MN"] + ", " + uc["CTR_MN_NM"]

# merge df with uc data
gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.lon, df.lat)).set_crs(4326)

geo_uc = gpd.read_file("downloads/GHS_STAT_UCDB2015MT_GLOBE_R2019A/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.gpkg")
geo_uc = geo_uc[['ID_HDC_G0', "CTR_MN_NM", "UC_NM_MN", "P15", "AREA", "geometry"]].dropna()
geo_uc["UC_NM_CTR"] = geo_uc["UC_NM_MN"] + ", " + geo_uc["CTR_MN_NM"]

gdf = gdf.sjoin(geo_uc, how="inner")

df = gdf

# make df of ratio poi/pop to filter data
df["count"] = 1
df_poi_per_pop = df.groupby(["ID_HDC_G0", "UC_NM_CTR"]).agg({"P15":"mean", "count":"sum", "AREA":"mean"}).reset_index()
df_poi_per_pop["poi_per_pop"] = df_poi_per_pop["count"]/df_poi_per_pop["P15"]
df_poi_per_pop["poi_per_km2"] = df_poi_per_pop["count"]/df_poi_per_pop["AREA"]

# list of urban centers to keep (at least 1 POI per km2)
uc_keep = df_poi_per_pop[(df_poi_per_pop.poi_per_km2>=1)&(df_poi_per_pop["count"]>=20)].ID_HDC_G0.to_list()
df_keep = df[df['ID_HDC_G0'].isin(uc_keep)]


# --- Refactored function to get accessibility for a single category and impedance type ---
def get_category_accessibility(pois_subset, network, maxdist, category_name, impedance_name):
    """
    Computes walking times from each street intersection to a specific category of urban amenities
    using a given network impedance.

    Args:
        pois_subset (gpd.GeoDataFrame): POIs for the current category (expected to be in network's CRS).
        network (pandana.Network): The pandana network object.
        maxdist (float): Maximum distance (in impedance units, e.g., seconds) to look for POIs.
        category_name (str): The name of the POI category.
        impedance_name (str): The name of the impedance column to use for calculation.

    Returns:
        pd.Series: A Series indexed by network node IDs, containing the travel times
                   to the nearest POI of the specified category.
    """
    # Set POIs for the current category on the network using projected coordinates
    # Ensure pois_subset has 'geometry' column with x and y attributes
    network.set_pois(category=category_name, maxdist=maxdist, maxitems=len(pois_subset),
                     x_col=pois_subset.geometry.x, y_col=pois_subset.geometry.y)

    # Calculate nearest POIs using the specified impedance.
    # network.nearest_pois returns a DataFrame with node IDs as index and columns '1', '2', etc.
    # for the distances to the 1st, 2nd, etc. nearest POI.
    # We are interested in the 1st nearest POI, which is column '1'.
    accessibility_df = network.nearest_pois(distance=maxdist, category=category_name, imp_name=impedance_name)
    
    # Return the Series for the 1st nearest POI
    return accessibility_df[1]


print("beginning loop")
### global loop
# processed cities
p_cities = [city.split(".")[0] for city in os.listdir("data/processed/access/")]
# # remaning cities (uncomment and use this line to process remaining cities)
# r_cities = list(df_keep[~df_keep["ID_HDC_G0"].isin(p_cities)].groupby("ID_HDC_G0").mean().sort_values("AREA", ascending=False).reset_index().ID_HDC_G0.unique())
# only process Barcelona (ID 2051) for demonstration, 945 NY, 154 CDMX, 11862 jakarta, 2565 abuja
r_cities = [154, 11862]#[2051, 945, 154, 11862, 2565]

for city in tqdm(r_cities, desc="Processing cities"):
    start_time = time.time() # Start timer for the current city

    city_name = df_keep[df_keep["ID_HDC_G0"]==city]["UC_NM_CTR"].unique()[0]
    print(f"\nCalculating accessibility for {city_name}") # Added newline for better readability with tqdm
    
    # subset pois for specific urban center
    pois = df_keep[df_keep["ID_HDC_G0"]==city]
    # Ensure POIs are in the correct CRS before using their geometry for snapping
    pois = gpd.GeoDataFrame(
        pois, geometry=gpd.points_from_xy(pois.lon, pois.lat)).set_crs(4326)
    
    # Get the specific urban center polygon for the current city
    urban_center_polygon = geo_uc[geo_uc["ID_HDC_G0"] == city].geometry.iloc[0]

    # Define the path for the cached pandana network file
    pandana_network_file = os.path.join(PANDANA_NETWORK_CACHE_DIR, f"{city}_pandana_network.h5")

    # Define the maximum time (in seconds) for accessibility queries
    max_time_seconds = MAX_DISTANCE / WALKING_SPEED_MPS 

    network = None # Initialize network variable
    nodes_osm = None # Initialize nodes_osm to store graph nodes
    graph_crs = None # Initialize graph_crs to store the projected CRS

    if os.path.exists(pandana_network_file):
        print(f"Loading precomputed network for {city_name} from cache...")
        network = pdna.Network.from_hdf5(pandana_network_file)
        # When loading from HDF5, pandana network object has nodes_df and edges_df
        # We need to reconstruct nodes_osm and graph_crs for consistency with the 'else' block
        nodes_osm = network.nodes_df.set_index('id').rename_axis('osmid')
        graph_crs = network.nodes_df.crs # Get CRS from the loaded nodes_df
        if graph_crs is None: # Fallback if CRS is not directly stored in nodes_df
            # If CRS is not directly in nodes_df, it might be in the HDF5 metadata or assumed
            # For robustness, we might need to re-project nodes_osm if it's not explicitly set.
            # However, pandana.Network.from_hdf5 should preserve the CRS if saved correctly.
            print("Warning: CRS not found directly in loaded network.nodes_df. Assuming original projected CRS.")
            # If you know the original projected CRS, you could set it here, e.g., graph_crs = 'EPSG:XXXX'
            # For now, we'll let the access_crs logic handle it later.
            pass

    else:
        print(f"Downloading street network for {city_name} using polygon...")
        G = ox.graph.graph_from_polygon(
            urban_center_polygon,
            network_type='walk',
            retain_all=True # Ensure all nodes and edges are kept
        )
        
        # Project the graph to a local UTM zone
        G = ox.project_graph(G)
        graph_crs = G.graph['crs'] # Store the projected CRS

        # Convert the OSMnx graph to nodes and edges GeoDataFrames
        nodes_osm, edges_osm = ox.utils_graph.graph_to_gdfs(G)
        
        # Calculate base travel time for each edge (without traffic lights)
        edges_osm['travel_time_seconds_no_tl'] = edges_osm['length'] / WALKING_SPEED_MPS

        # Download traffic signals from OSMnx using the polygon
        print(f"Downloading traffic signals for {city_name} using polygon...")
        traffic_signals = ox.features.features_from_polygon(
            urban_center_polygon,
            tags={'highway': 'traffic_signals'}
        )

        # Identify traffic signal nodes in the network
        traffic_signal_node_ids = set()
        if not traffic_signals.empty:
            # Ensure traffic_signals are points; if not, convert their centroids
            traffic_signals_points = traffic_signals.geometry.centroid if not traffic_signals.geometry.type.iloc[0] == 'Point' else traffic_signals.geometry
            
            # Get the nearest network node for each traffic signal point
            # Ensure traffic_signals_points are in the same CRS as G before snapping
            traffic_signals_points_proj = traffic_signals_points.to_crs(graph_crs)
            traffic_signal_nodes = ox.nearest_nodes(
                G,
                X=traffic_signals_points_proj.x,
                Y=traffic_signals_points_proj.y
            )
            traffic_signal_node_ids.update(traffic_signal_nodes)
        
        print(f"Found {len(traffic_signal_node_ids)} traffic signal nodes.")

        # Calculate travel time including traffic light delays
        edges_osm['travel_time_seconds'] = edges_osm['travel_time_seconds_no_tl'].copy() # Start with base time
        
        # Apply traffic light delay to edges leading into traffic signal nodes
        # Access 'v' (to node) from the MultiIndex (level 1)
        edges_osm['travel_time_seconds'] = edges_osm.apply(
            lambda row: row['travel_time_seconds'] + TRAFFIC_LIGHT_DELAY_SECONDS
            if row.name[1] in traffic_signal_node_ids else row['travel_time_seconds'],
            axis=1
        )

        # Prepare edges DataFrame for pandana.Network by resetting index
        # This converts 'u', 'v', 'key' from index levels to regular columns
        edges_for_pandana = edges_osm.reset_index()

        # Create a single pandana network with both impedances
        print("Creating pandana network with multiple impedances...")
        network = pdna.Network(
            nodes_osm['x'], nodes_osm['y'],
            edges_for_pandana['u'], # 'u' node IDs from the new column
            edges_for_pandana['v'], # 'v' node IDs from the new column
            edges_for_pandana[['travel_time_seconds', 'travel_time_seconds_no_tl']] # Pass both impedance columns
        )
        
        # Precompute the single network for the maximum travel time
        print("Precomputing network...")
        network.precompute(max_time_seconds + 1)
        
        # Save the precomputed network to disk
        print(f"Saving precomputed network for {city_name} to cache...")
        network.save_hdf5(pandana_network_file)

    # --- Project POIs to the network's CRS ---
    # This must happen AFTER the network's CRS (graph_crs) is determined
    pois_projected = pois.to_crs(graph_crs)

    # --- Calculate Accessibility for Both Scenarios ---
    all_access_with_tl_series = []
    all_access_no_tl_series = []

    cat_list_str = list(pois_projected['category'].unique()) # Use projected POIs for categories

    print("Calculating category travel times (with and without traffic lights)...")
    for cat in tqdm(cat_list_str, desc=f"Processing categories for {city_name}"):
        pois_subset = pois_projected[pois_projected['category'] == cat]

        # Calculate accessibility with traffic lights using the single network
        access_series_with_tl = get_category_accessibility(
            pois_subset=pois_subset, # Pass the projected POI subset
            network=network, # Use the single network object
            maxdist=max_time_seconds,
            category_name=cat,
            impedance_name='travel_time_seconds' # Specify the impedance name
        )
        # Rename the series for clear identification in the final DataFrame
        all_access_with_tl_series.append(access_series_with_tl.rename(f'{cat}_with_tl'))

        # Calculate accessibility without traffic lights using the single network
        access_series_no_tl = get_category_accessibility(
            pois_subset=pois_subset, # Pass the projected POI subset
            network=network, # Use the single network object
            maxdist=max_time_seconds,
            category_name=cat,
            impedance_name='travel_time_seconds_no_tl' # Specify the impedance name
        )
        # Rename the series for clear identification in the final DataFrame
        all_access_no_tl_series.append(access_series_no_tl.rename(f'{cat}_no_tl'))

    # Combine all accessibility series for each scenario into DataFrames
    # The index of these DataFrames will be the pandana node IDs (osmid)
    access_df_with_tl = pd.concat(all_access_with_tl_series, axis=1)
    access_df_no_tl = pd.concat(all_access_no_tl_series, axis=1)

    # Merge the two accessibility DataFrames on their common index (node IDs)
    combined_access_df = access_df_with_tl.merge(
        access_df_no_tl, left_index=True, right_index=True, how='inner'
    )

    print("Cleaning up the output and adding metadata...")
    # Merge combined accessibility values with network nodes to get geometry and original IDs
    # nodes_osm has 'osmid' as its index. Reset index and rename to 'id' for merging.
    nodes_for_merge = nodes_osm.reset_index().rename(columns={'osmid': 'id'})
    
    # Select only necessary columns from nodes_for_merge for the final output
    access = nodes_for_merge[['id', 'x', 'y', 'geometry']].merge(
        combined_access_df.reset_index().rename(columns={'osmid': 'id'}), # FIX: Rename 'osmid' to 'id'
        on='id',
        how='inner'
    )

    # Add metadata (from the 'pois' dataframe, which is consistent for the city)
    access["ID_HDC_G0"] = pois["ID_HDC_G0"].unique()[0]
    access["CTR_MN_NM"] = pois["CTR_MN_NM"].unique()[0]
    access["UC_NM_MN"] = pois["UC_NM_MN"].unique()[0]
    access["P15"] = pois["P15"].unique()[0]
    access["AREA"] = pois["AREA"].unique()[0]
    access["UC_NM_CTR"] = pois["UC_NM_CTR"].unique()[0]
    
    # Convert to GeoDataFrame and set right crs
    # The CRS is now stored in graph_crs
    access = gpd.GeoDataFrame(access, geometry=access['geometry'], crs=graph_crs)
    
    # Drop NaNs (e.g., if a node couldn't reach any POI within maxdist in either scenario)
    access = access.dropna()

    # Save the combined results to a new CSV file
    access.to_csv(f"data/processed/access/{city}_time_comparison.csv")
    print(f"Saved accessibility comparison for {city_name} to data/processed/access/{city}_time_comparison.csv")

    end_time = time.time() # End timer for the current city
    print(f"Total computational time for {city_name}: {end_time - start_time:.2f} seconds")
