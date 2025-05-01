from spectral import open_image
import numpy as np
from skimage import filters, measure, morphology
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt

path_out = '/uoa/home/c03jd24/sharedscratch/HSI-processing/csv_out'

# Load CSV file and extract filenames
try:
    # Read the CSV file
    #Alter to ensure path is correct
    df = pd.read_csv('/uoa/scratch/shared/2025_hackathon/RGB_and_VIS-NIR_HSI_data_for_90_rice_seed_varieties/RGB_and_VIS-NIR_HSI_data_for_90_rice_seed_varieties/index.csv')
    print(df)
    # Check if 'File Name' column exists
    if 'File Name' not in df.columns:
        raise ValueError("Column 'File Name' not found in CSV")
    
    # Extract filenames and append '.hdr'
    #partial-dataset
    graintypes = df['File Name'].unique()
    filenames = df.apply(lambda row: f"/uoa/scratch/shared/2025_hackathon/RGB_and_VIS-NIR_HSI_data_for_90_rice_seed_varieties/RGB_and_VIS-NIR_HSI_data_for_90_rice_seed_varieties/{row['Folder']}/{row['File Name']}.hdr", axis=1).to_numpy()
    #grainfiles = df.apply(lambda row: f"/uoa/home/c03jd24/sharedscratch/HSI-processing/out/{row['Folder']}/{row['File Name']}.hdr_grain_data.csv", axis=1).to_numpy()
   # grainfiles = df.apply(lambda row: f"/uoa/home/c03jd24/sharedscratch/HSI-processing/out/{row['Folder']}/{row['File Name']}.csv", axis=1).to_numpy()
    # Print the resulting array
    #print(filenames)
    print(grainfiles)
except FileNotFoundError:
    print("Error: 'index.csv' not found")
except Exception as e:
    print(f"Error: {e}")

for filename in filenames:
    try:
        # Load the HSI image
        hsi_image = open_image(filename)
        hsi_data = hsi_image.load()  # Shape: (height, width, bands)
        
        # Preprocess: Compute mean reflectance
        hsi_mean = np.mean(hsi_data, axis=2)
        
        # Segment grains
        thresh = filters.threshold_otsu(hsi_mean)
        binary = hsi_mean > thresh
        binary = morphology.remove_small_objects(binary, min_size=50)
        binary = morphology.binary_closing(binary, morphology.disk(3))
        labeled_image = measure.label(binary, connectivity=2)
        regions = measure.regionprops(labeled_image)
        
        # Extract spectral data for each grain
        grain_data = []
        for region in regions:
            coords = region.coords
            spectra = hsi_data[coords[:, 0], coords[:, 1], :]
            mean_spectrum = np.mean(spectra, axis=0)
            grain_info = {
                'grain_id': region.label,
                'grain_type': filename,
                'area': region.area,
                'eccentricity': region.eccentricity,
                'perimeter': region.perimeter,
                **{f'band_{i+1}': mean_spectrum[i] for i in range(mean_spectrum.shape[0])}
            }
            grain_data.append(grain_info)

        # Save to CSV
        df_grain = pd.DataFrame(grain_data)
        name = os.path.basename(os.path.normpath(filename))
        #df_grain.to_csv(os.path.join(path_out,f'{filename}_grain_data.csv'))
        df_grain.to_csv(os.path.join(path_out, name + '.csv'))
        #df_grain.to_csv(path_out, f'{filename}_grain_data.csv', index=False)
        

    except Exception as e:
        print(f"Error processing {filename}: {e}")


#Combine all grain data into a single CSV
final_grain_data = []
for grainfile in glob.glob('/uoa/home/c03jd24/sharedscratch/HSI-processing/csv_out/*.csv'):
    print(grainfile)
    try:
        df_grain = pd.read_csv(grainfile)  
        final_grain_data.append(df_grain)
    except FileNotFoundError:
        print(f"Error: {grainfile} not found")
    except Exception as e:
        print(f"Error reading {grainfile}: {e}")

# Check if final_grain_data is not empty, then concatenate and save
if final_grain_data:
    combined_df = pd.concat(final_grain_data, ignore_index=True)
    combined_df.to_csv('full_grain_data.csv', index=False)
    print("Combined grain data saved to 'full_grain_data.csv'")
else:
    print("No grain data to combine")

