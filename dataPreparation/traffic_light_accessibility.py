import osmnx as ox
import pandas as pd
import geopandas as gpd
import numpy as np
from pandana.loaders import osm
import pandana as pdna
from shapely.geometry import Point
from tqdm import tqdm

def get_network_with_traffic_lights(lat_min, lng_min, lat_max, lng_max):
    """
    Get pedestrian network and identify traffic signals
    
    Returns:
        network: Pandana network with traffic light impedances
        traffic_nodes: DataFrame with traffic signal locations
    """
    # First, get the pedestrian network as usual
    network = osm.pdna_network_from_bbox(lat_min, lng_min, lat_max, lng_max, network_type='walk')
    
    # Use OSMnx to download additional data including traffic signals
    G = ox.graph_from_bbox(lat_min, lng_min, lat_max, lng_max, network_type='walk', simplify=False)
    
    # Extract traffic light nodes (intersections with traffic signals)
    traffic_nodes = []
    
    for node, data in G.nodes(data=True):
        if 'highway' in data and data['highway'] == 'traffic_signals':
            traffic_nodes.append({
                'node_id': node,
                'x': data['x'],
                'y': data['y']
            })
    
    traffic_nodes_df = pd.DataFrame(traffic_nodes)
    
    if len(traffic_nodes_df) > 0:
        # Convert traffic nodes to geodataframe for spatial join
        traffic_gdf = gpd.GeoDataFrame(
            traffic_nodes_df, 
            geometry=gpd.points_from_xy(traffic_nodes_df.x, traffic_nodes_df.y),
            crs="EPSG:4326"
        )
        
        # Convert Pandana nodes to geodataframe
        nodes_gdf = gpd.GeoDataFrame(
            network.nodes_df.reset_index(), 
            geometry=gpd.points_from_xy(network.nodes_df.x, network.nodes_df.y),
            crs="EPSG:4326"
        )
        
        # Join traffic lights to network nodes (finding closest network node to each traffic light)
        joined = gpd.sjoin_nearest(traffic_gdf, nodes_gdf, how="left")
        
        # Extract IDs of network nodes that are traffic lights
        traffic_light_node_ids = joined['id'].tolist()
        
        # Create a new edge attribute for time - initially set to distance (assuming walking speed of 1 m/s)
        network.edges_df['time'] = network.edges_df['distance']
        
        # Add waiting time at traffic lights
        # Identify edges that connect to traffic light nodes
        for node_id in traffic_light_node_ids:
            # Find edges where node_id is the to_node (meaning pedestrian arrives at traffic light)
            incoming_edges = network.edges_df[network.edges_df['to'] == node_id]
            
            # Add waiting time (assume average wait of 30 seconds at traffic lights)
            for idx in incoming_edges.index:
                network.edges_df.loc[idx, 'time'] += 30  # adding 30 seconds
        
        return network, traffic_gdf
    else:
        print("No traffic signals found in this area")
        return network, None

def create_time_based_access_gdf(pois, network, maxtime=600):
    """
    Computes walking times from each street intersection to POIs,
    accounting for delays at traffic lights
    
    Args:
        pois: GeoDataFrame with POIs
        network: Pandana network with time impedances
        maxtime: Maximum travel time in seconds
    
    Returns:
        access: GeoDataFrame with accessibility metrics
    """
    print("Initialize network POIs")
    
    # Set the impedance to be time instead of distance
    network.precompute(maxtime + 1, impede_factor='time')
    
    cat_list_str = list(pois.groupby(['category']).mean().reset_index()['category'])
    
    # Create initial dataframe with dummy calculations at 1s
    print("Creating dummy df")
    for cat in cat_list_str:
        pois_subset = pois[pois['category'] == cat]
        network.set_pois(
            category=cat, 
            maxdist=1,  # 1 second time 
            maxitems=len(pois_subset), 
            x_col=pois_subset['lon'], 
            y_col=pois_subset['lat']
        )
        accessibility = network.nearest_pois(distance=1, category=cat)
    
    print("Calculating category travel times")
    # Now calculate actual travel times
    for cat in cat_list_str:
        pois_subset = pois[pois['category'] == cat]
        network.set_pois(
            category=cat, 
            maxdist=maxtime,  # Max time in seconds
            maxitems=len(pois_subset), 
            x_col=pois_subset['lon'], 
            y_col=pois_subset['lat']
        )
        
        # Calculate using time as impedance
        accessibility[str(cat)] = network.nearest_pois(
            distance=maxtime, 
            category=cat,
            imp_name='time'  # Use time impedance
        )
    
    print("Cleaning up the output, adding metadata")
    # Merge accessibility values with walk nodes
    access = pd.merge(
        accessibility.reset_index().drop(1, axis=1),
        network.nodes_df.reset_index(),
        on='id'
    )
    
    # Add metadata
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
    access.crs = {"init": "epsg:4326"}
    
    # Drop NaNs
    access = access.dropna()
    
    return access

# Example of how to modify your main analysis loop:
def calculate_traffic_light_accessibility(city_id, df_keep):
    """
    Calculate accessibility for a city with traffic light considerations
    """
    city_name = df_keep[df_keep["ID_HDC_G0"] == city_id]["UC_NM_CTR"].unique()[0]
    print(f"Calculating accessibility with traffic lights for {city_name}")
    
    # Subset POIs for specific urban center
    pois = df_keep[df_keep["ID_HDC_G0"] == city_id]
    pois = gpd.GeoDataFrame(
        pois, geometry=gpd.points_from_xy(pois.lon, pois.lat)
    )
    
    # Get boundary coords of urban center
    lng_min = pois.total_bounds[0]
    lat_min = pois.total_bounds[1]
    lng_max = pois.total_bounds[2]
    lat_max = pois.total_bounds[3]
    
    # Get pedestrian network with traffic light impedance
    network, traffic_lights = get_network_with_traffic_lights(lat_min, lng_min, lat_max, lng_max)
    
    # Calculate time-based accessibility
    access_with_signals = create_time_based_access_gdf(pois=pois, network=network, maxtime=900)
    
    # Save outputs
    access_with_signals.to_csv(f"data/processed/access_with_signals/{city_id}.csv")
    
    if traffic_lights is not None:
        traffic_lights.to_csv(f"data/processed/traffic_lights/{city_id}.csv")
    
    # For comparison, also calculate without traffic signals
    # Use original network and distance-based calculation from your current code
    # [...]
    
    return access_with_signals

# To compare results
def compare_accessibility(city_id):
    """
    Compare regular accessibility vs accessibility with traffic light delays
    """
    # Load regular accessibility results
    access_regular = pd.read_csv(f"data/processed/access/{city_id}.csv")
    
    # Load traffic light-adjusted accessibility
    access_signals = pd.read_csv(f"data/processed/access_with_signals/{city_id}.csv")
    
    # Calculate differences and statistics
    comparison = pd.DataFrame()
    
    for cat in ['active_living', 'community_space', 'education', 
                'food_choices', 'health_wellbeing', 'nightlife', 'mobility']:
        # Regular values are in meters, signal values are in seconds
        # Convert regular to time assuming 1.4 m/s walking speed
        comparison[f"{cat}_distance"] = access_regular[cat]
        comparison[f"{cat}_time_no_signals"] = access_regular[cat] / 1.4  # Convert to seconds
        comparison[f"{cat}_time_with_signals"] = access_signals[cat]
        comparison[f"{cat}_delay_pct"] = ((comparison[f"{cat}_time_with_signals"] / 
                                          comparison[f"{cat}_time_no_signals"]) - 1) * 100
    
    # Calculate overall statistics
    summary = comparison.describe()
    
    return comparison, summary
