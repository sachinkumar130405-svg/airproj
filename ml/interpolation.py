import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def idw_interpolation(stations_df, grid_resolution=0.01, power=2):
    """
    Interpolates PM2.5 values across a grid using Inverse Distance Weighting (IDW).

    Parameters
    ----------
    stations_df : DataFrame with columns ['lat', 'lon', 'pm25']
    grid_resolution : float, grid spacing in degrees (~0.01 ≈ 1km)
    power : float, IDW power parameter

    Returns
    -------
    grid_df : DataFrame with columns ['lat', 'lon', 'pm25_est']
    """
    lats = np.arange(stations_df['lat'].min(), stations_df['lat'].max(), grid_resolution)
    lons = np.arange(stations_df['lon'].min(), stations_df['lon'].max(), grid_resolution)
    grid_lat, grid_lon = np.meshgrid(lats, lons)
    grid_points = np.vstack([grid_lat.ravel(), grid_lon.ravel()]).T

    tree = cKDTree(stations_df[['lat', 'lon']].values)
    dist, idx = tree.query(grid_points, k=4)

    weights = 1 / (dist ** power + 1e-9)
    weights /= weights.sum(axis=1, keepdims=True)
    interpolated = np.sum(weights * stations_df.iloc[idx]['pm25'].values, axis=1)

    grid_df = pd.DataFrame({
        'lat': grid_points[:, 0],
        'lon': grid_points[:, 1],
        'pm25_est': interpolated
    })
    return grid_df
