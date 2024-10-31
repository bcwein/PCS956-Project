import os
import numpy as np
import pandas as pd

def generate_synthetic_vineyard_data(field_width=20, field_length=200, resolution=0.5):
    """
    Generates synthetic vineyard data with 2D grids for DTM, CHM, NDVI, LAI, and Botrytis Risk.
    Applies a threshold to CHM and NDVI, setting values below 0.5 to NaN.
    Returns a DataFrame containing the data.
    """
    # Calculate the grid dimensions based on resolution
    grid_width = int(field_width / resolution)
    grid_length = int(field_length / resolution)

    # 1. Digital Terrain Model (DTM) - Simulate a sloping field with small local variations
    y_slope = np.linspace(76, 67, grid_length)[:, None]  # Slope along the length
    local_variations = np.random.uniform(low=-0.5, high=0.5, size=(grid_length, grid_width))  # Local variations
    DTM = y_slope + local_variations

    # 2. Canopy Height Model (CHM) - Rows of canopy height with threshold applied at 0.5m
    CHM = np.zeros((grid_length, grid_width))
    for i in range(grid_width):
        if i % 5 == 0:  # Create rows every 5 cells along the width
            CHM[:, i] = np.random.uniform(low=1.0, high=2.0, size=(grid_length,))
    CHM += np.random.normal(0, 0.2, size=(grid_length, grid_width))  # Add local variation
    CHM[CHM < 0.5] = np.nan  # Apply threshold to isolate CHM (canopy only where CHM > 0.5)

    # 3. Normalized Difference Vegetation Index (NDVI) - Masked by CHM threshold
    NDVI = np.full_like(CHM, np.nan)  # Initialize NDVI with NaN values
    ndvi_values = 0.3 + (CHM - np.nanmin(CHM)) / (np.nanmax(CHM) - np.nanmin(CHM)) * 0.6  # Scale NDVI from 0.3 to 0.9
    NDVI[~np.isnan(CHM)] = ndvi_values[~np.isnan(CHM)] + np.random.normal(0, 0.05, size=CHM[~np.isnan(CHM)].shape)  # Add noise
    NDVI[NDVI > 1] = 1  # Cap NDVI at 1

    # 4. Leaf Area Index (LAI) - Correlated with CHM and NDVI
    LAI = np.full((grid_length, grid_width), np.nan)
    LAI[~np.isnan(CHM)] = 0.1 * CHM[~np.isnan(CHM)] + 0.05 * NDVI[~np.isnan(CHM)] + np.random.normal(0, 0.1, size=CHM[~np.isnan(CHM)].shape)

    # 5. Botrytis Risk - Higher probability in lower elevation areas with canopy
    Botrytis_Risk = np.zeros((grid_length, grid_width))
    
    # Apply risk pattern with stronger trend in lower elevation areas with canopy
    for i in range(grid_length):
        for j in range(grid_width):
            # Main rule: risk in lower elevations with canopy
            if i > 2 * grid_length / 3 and not np.isnan(CHM[i, j]):  # Only in lower end and where canopy exists
                if np.random.rand() < 0.8:  # 80% chance
                    terrain_influence = 1 - (DTM[i, j] - 67) / (76 - 67)
                    veg_influence = (1 - NDVI[i, j]) * 0.5
                    lai_influence = LAI[i, j] * 0.2
                    random_factor = np.random.uniform(0.7, 1.0)
                    Botrytis_Risk[i, j] = max(0, terrain_influence + veg_influence - lai_influence) * random_factor

            # Small chance of Botrytis in higher elevation areas
            if i <= 2 * grid_length / 3 and not np.isnan(CHM[i, j]):
                if np.random.rand() < 0.1:  # 10% chance
                    veg_influence = (1 - NDVI[i, j]) * 0.5
                    lai_influence = LAI[i, j] * 0.2
                    random_factor = np.random.uniform(0.5, 0.9)
                    Botrytis_Risk[i, j] = max(0, veg_influence - lai_influence) * random_factor

    # Normalize Botrytis Risk between 0 and 1
    Botrytis_Risk = Botrytis_Risk / np.nanmax(Botrytis_Risk)

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
    Checks if the file already exists to avoid overwriting.
    """
    # Check if the file already exists
    if not os.path.exists(output_file):
        print(f"File {output_file} not found. Generating new data...")
        data = generate_synthetic_vineyard_data()
        data.to_parquet(output_file, index=False)
        print(f"Data saved to {output_file}")
    else:
        print(f"File {output_file} already exists. Skipping data generation.")

# If run as a script
if __name__ == "__main__":
    # Ensure the data folder exists
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Save data to Parquet format
    save_to_parquet()
