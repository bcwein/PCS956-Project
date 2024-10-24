# src/generate_data.py

import numpy as np
import pandas as pd

def generate_synthetic_vineyard_data(field_width=20, field_length=200, resolution=0.5):
    """
    Generates synthetic vineyard data with 2D grids for DTM, CHM, NDVI, LAI, and Botrytis Risk.
    Returns a DataFrame containing the data.
    """
    # Calculate the grid dimensions based on resolution
    grid_width = int(field_width / resolution)
    grid_length = int(field_length / resolution)

    # Create a smooth gradient for the DTM along the 200-meter length (from 76m to 67m)
    y_slope = np.linspace(76, 67, grid_length)[:, None]  # Slope along the length
    local_variations = np.random.uniform(low=-0.5, high=0.5, size=(grid_length, grid_width))  # Local variations
    DTM = y_slope + local_variations

    # Simulate the canopy height (CHM) along rows with local variations (1-2m)
    CHM = np.zeros((grid_length, grid_width))
    for i in range(grid_width):
        if i % 5 == 0:  # Create rows every 5 cells along the width
            CHM[:, i] = np.random.uniform(low=1.0, high=2.0, size=(grid_length,))
    CHM += np.random.normal(0, 0.2, size=(grid_length, grid_width))  # Add local variation
    CHM[CHM < 0.5] = 0  # Apply threshold to isolate CHM

    # Update NDVI based on the CHM
    NDVI = 0.3 + (CHM - CHM.min()) / (CHM.max() - CHM.min()) * 0.6  # Scale NDVI from 0.3 to 0.9

    # LAI correlated with CHM but with noise
    LAI = 0.1 * CHM + np.random.normal(0, 0.1, size=(grid_length, grid_width))

    # Botrytis Risk - create risk clusters based on updated CHM and NDVI
    Botrytis_Risk = np.zeros((grid_length, grid_width))
    cluster_centers = [(int(grid_length * 0.4), int(grid_width * 0.3)), (int(grid_length * 0.7), int(grid_width * 0.7))]
    cluster_radius = 15  # Cluster radius in grid cells
    for center in cluster_centers:
        x, y = center
        for i in range(grid_length):
            for j in range(grid_width):
                dist = np.sqrt((i - x) ** 2 + (j - y) ** 2)
                if dist <= cluster_radius and CHM[i, j] > 0.5:
                    terrain_influence = (DTM[i, j] - 67) / (76 - 67)  # normalized terrain influence
                    veg_influence = (1 - NDVI[i, j]) * 0.5
                    Botrytis_Risk[i, j] = (1 / (1 + dist)) + terrain_influence + veg_influence

    # Normalize risk between 0 and 1
    Botrytis_Risk = Botrytis_Risk / Botrytis_Risk.max()

    # Return the data as a DataFrame
    return pd.DataFrame({
        'DTM': DTM.flatten(),
        'CHM': CHM.flatten(),
        'NDVI': NDVI.flatten(),
        'LAI': LAI.flatten(),
        'Botrytis_Risk': Botrytis_Risk.flatten()
    })

def save_to_parquet(output_file='data/synthetic_vineyard_data.parquet'):
    """
    Generates synthetic vineyard data and saves it to a Parquet file.
    """
    data = generate_synthetic_vineyard_data()
    data.to_parquet(output_file, index=False)
    print(f"Data saved to {output_file}")

# If run as a script
if __name__ == "__main__":
    save_to_parquet()
