# extract data from https://data.prism.oregonstate.edu/monthly

import os
import requests
from datetime import datetime
from tqdm import tqdm
import zipfile
import time

# Configuration
base_urls = {
    "ppt": "https://data.prism.oregonstate.edu/monthly/ppt/",
    "tdmean": "https://data.prism.oregonstate.edu/monthly/tdmean/",
    "tmax": "https://data.prism.oregonstate.edu/monthly/tmax/",
    "tmean": "https://data.prism.oregonstate.edu/monthly/tmean/",
    "tmin": "https://data.prism.oregonstate.edu/monthly/tmin/",
    "vpdmax": "https://data.prism.oregonstate.edu/monthly/vpdmax/",
    "vpdmin": "https://data.prism.oregonstate.edu/monthly/vpdmin/"
}

start_year = 1950
end_year = 2024

path = os.getcwd()
download_dir_base = os.path.join(path, "data/prism_downloads")
unzip_dir_base = os.path.join(path, "data/prism_unzipped")


def download_prism_data(data_type, base_url):
    print(f"\nProcessing PRISM data for type: {data_type}")
    
    download_dir_type = os.path.join(download_dir_base, data_type)
    unzip_dir_type = os.path.join(unzip_dir_base, data_type)
    os.makedirs(download_dir_type, exist_ok=True)
    os.makedirs(unzip_dir_type, exist_ok=True)

    with tqdm(total=(end_year - start_year + 1), desc=f"{data_type.upper()} Yearly Progress") as pbar:
        for year in range(start_year, end_year + 1):
            m_num = "2" if (data_type == "ppt" and year < 1981) else "3"
            file_type = "stable" if year < 2024 else "provisional"
            all_no = "_all" if year < 2024 else ""

            date_str = f"{year}"
            filename = f"PRISM_{data_type}_{file_type}_4kmM{m_num}_{date_str}{all_no}_bil.zip"
            url = f"{base_url}{year}/{filename}"

            local_path = os.path.join(download_dir_type, filename)
            unzip_path = os.path.join(unzip_dir_type, date_str)

            if os.path.exists(unzip_path):
                pbar.write(f"{date_str}: Already extracted, skipping.")
                pbar.update(1)
                continue

            try:
                response = requests.head(url, timeout=10)
                if response.status_code == 200:
                    response = requests.get(url, stream=True, timeout=30)
                    with open(local_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    with zipfile.ZipFile(local_path, 'r') as zip_ref:
                        zip_ref.extractall(unzip_path)

                    os.remove(local_path)
                    pbar.write(f"{date_str}: Downloaded and extracted.")
                else:
                    pbar.write(f"{date_str}: File not found (HTTP {response.status_code}).")
            except Exception as e:
                pbar.write(f"{date_str}: Error - {e}")
            finally:
                time.sleep(0.5)
                pbar.update(1)


if __name__ == "__main__":
    for data_type, base_url in base_urls.items():
        download_prism_data(data_type, base_url)

    # just do for 1983 - 2024 for ppt (M3 instead of M2)

    print("\nAll downloads and extractions completed successfully!")