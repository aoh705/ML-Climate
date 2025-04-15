# extract data from https://data.prism.oregonstate.edu/daily/ppt/

import os
import requests
from datetime import datetime, timedelta
from tqdm import tqdm
import zipfile
import time

# Configuration
base_url = "https://data.prism.oregonstate.edu/daily/ppt/"
start_year = 2019
end_year = 2023
download_dir = "prism_downloads"
unzip_dir = "prism_unzipped"

# Create directories if they don't exist
os.makedirs(download_dir, exist_ok=True)
os.makedirs(unzip_dir, exist_ok=True)

def download_prism_data():
    current_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    
    # Progress bar setup
    total_days = (end_date - current_date).days + 1
    pbar = tqdm(total=total_days, desc="Downloading PRISM data")

    while current_date <= end_date:
        year = current_date.year
        date_str = current_date.strftime("%Y%m%d")
        filename = f"PRISM_ppt_stable_4kmD2_{date_str}_bil.zip"
        url = f"{base_url}{year}/{filename}"
        
        local_path = os.path.join(download_dir, filename)
        unzip_path = os.path.join(unzip_dir, date_str)

        # Skip if already unzipped
        if not os.path.exists(unzip_path):
            try:
                # Check if file exists
                response = requests.head(url)
                if response.status_code == 200:
                    # Download the file
                    response = requests.get(url, stream=True, timeout=30)
                    with open(local_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # Unzip the file
                    with zipfile.ZipFile(local_path, 'r') as zip_ref:
                        zip_ref.extractall(unzip_path)
                    
                    # Remove zip file
                    os.remove(local_path)
                
            except Exception as e:
                tqdm.write(f"Error processing {date_str}: {str(e)}")
            finally:
                # Add delay to be polite to server
                time.sleep(0.5)

        current_date += timedelta(days=1)
        pbar.update(1)
    
    pbar.close()

if __name__ == "__main__":
    download_prism_data()
    print("Download and extraction complete!")
