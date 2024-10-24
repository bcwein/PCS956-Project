import os
import numpy as np
import pandas as pd

# Function to generate synthetic vineyard data with stronger but more random Botrytis Risk patterns
def generate_synthetic_vineyard_data(field_width=20, field_length=200, resolution=0.5):
    """
    Generates synthetic vineyard data with 2D grids for DTM, CHM, NDVI, LAI, and Botrytis Risk.
    Returns a DataFrame containing the data, emphasizing higher Botrytis Risk at lower elevations and under canopies, with more randomness.
    """
    # Calculate the grid dimensions based on resolution
    grid_width = int(field_width / resolution)
    grid_length = int(field_length / resolution)

    # 1. Digital Terrain Model (DTM) - Simulate a sloping field with small local variations
    y_slope = np.linspace(76, 67, grid_length)[:, None]  # Slope along the length
    local_variations = np.random.uniform(low=-0.5, high=0.5, size=(grid_length, grid_width))  # Local variations
    DTM = y_slope + local_variations

    # 2. Canopy Height Model (CHM) - Rows of canopy height influenced by DTM
    CHM = np.zeros((grid_length, grid_width))
    for i in range(grid_width):
        if i % 5 == 0:  # Create rows every 5 cells along the width
            CHM[:, i] = np.random.uniform(low=1.0, high=2.0, size=(grid_length,))
    CHM += np.random.normal(0, 0.2, size=(grid_length, grid_width))  # Add local variation
    CHM[CHM < 0.5] = 0  # Apply threshold to isolate CHM (canopy is only where CHM > 0.5)

    # 3. Normalized Difference Vegetation Index (NDVI) - Influenced by CHM and DTM
    NDVI = 0.3 + (CHM - CHM.min()) / (CHM.max() - CHM.min()) * 0.6  # Scale NDVI from 0.3 to 0.9
    NDVI += np.random.normal(0, 0.05, size=(grid_length, grid_width))  # Add small noise
    NDVI[NDVI > 1] = 1  # Cap NDVI at 1

    # 4. Leaf Area Index (LAI) - Correlated with CHM, some influence from NDVI
    LAI = 0.1 * CHM + 0.05 * NDVI + np.random.normal(0, 0.1, size=(grid_length, grid_width))

    # 5. Botrytis Risk - Higher probability in lower elevation areas with canopy, with more randomness
    Botrytis_Risk = np.zeros((grid_length, grid_width))
    
    # Stronger trend: Lower elevation areas, especially with canopy, have Botrytis Risk near 1
    # Some randomness introduced to break the pattern

    for i in range(grid_length):
        for j in range(grid_width):
            # Main rule: risk in lower elevations with canopy
            if i > 2 * grid_length / 3:  # Only in the lower elevation end of the field
                if CHM[i, j] > 0.5:  # Only where canopy exists
                    # Randomly determine if Botrytis occurs here (80% chance, and base risk is stronger)
                    if np.random.rand() < 0.8:
                        terrain_influence = 1 - (DTM[i, j] - 67) / (76 - 67)  # Invert terrain influence, lower = higher risk
                        veg_influence = (1 - NDVI[i, j]) * 0.5  # Lower NDVI contributes to higher risk
                        lai_influence = LAI[i, j] * 0.2  # Higher LAI might lower risk (better canopy health)
                        # Add randomness but make base risk stronger
                        random_factor = np.random.uniform(0.7, 1.0)
                        Botrytis_Risk[i, j] = max(0, terrain_influence + veg_influence - lai_influence) * random_factor

            # Special case: Small chance of Botrytis risk in higher elevation areas (10% chance, stronger trend)
            if i <= 2 * grid_length / 3 and CHM[i, j] > 0.5:
                if np.random.rand() < 0.1:  # 10% chance of risk in higher areas
                    veg_influence = (1 - NDVI[i, j]) * 0.5  # Lower NDVI contributes to higher risk
                    lai_influence = LAI[i, j] * 0.2  # Higher LAI might lower risk
                    # Random factor added here too
                    random_factor = np.random.uniform(0.5, 0.9)
                    Botrytis_Risk[i, j] = max(0, veg_influence - lai_influence) * random_factor

    # Normalize Botrytis Risk between 0 and 1
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
