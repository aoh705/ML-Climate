# # extract data from https://data.prism.oregonstate.edu/monthly

# import os
# import requests
# from datetime import datetime, timedelta
# from tqdm import tqdm
# import zipfile
# import time

# # Configuration
# ppt_url = "https://data.prism.oregonstate.edu/monthly/ppt/"
# tdmean_url = "https://data.prism.oregonstate.edu/monthly/tdmean/"
# tmax_url = "https://data.prism.oregonstate.edu/monthly/tmax/"
# tmean_url = "https://data.prism.oregonstate.edu/monthly/tmean/"
# tmin_url = "https://data.prism.oregonstate.edu/monthly/tmin/"
# vpdmax_url = "https://data.prism.oregonstate.edu/monthly/vpdmax/"
# vpdmin_url = "https://data.prism.oregonstate.edu/monthly/vpdmin/"

# types_url = [ppt_url, tdmean_url, tmax_url, tmean_url, tmin_url, vpdmax_url, vpdmin_url]
# types = ["ppt", "tdmean", "tmax", "tmean", "tmin", "vpdmax", "vpdmin"]

# start_year = 1950
# end_year = 2024

# path = os.getcwd()
# download_dir_base = path + "/data/prism_downloads"
# unzip_dir_base = path + "/data/prism_unzipped"

# def download_prism_data(url, type):
#     # Create directories if they don't exist
#     download_dir_type = download_dir_base + f"/{type}"
#     unzip_dir_type = unzip_dir_base + f"/{type}"
#     os.makedirs(download_dir_type, exist_ok=True)
#     os.makedirs(unzip_dir_type, exist_ok=True)

#     current_date = datetime(start_year)
#     end_date = datetime(end_year)
    
#     # Progress bar setup
#     total_years = (end_year - current_date) + 1
#     pbar = tqdm(total=total_years, desc="Downloading PRISM data")

#     while current_date <= end_date:
#         year = current_date.year
#         date_str = current_date.strftime("%Y")

#         #M2 for ppt and M3 for everything else
#         m_num = None
#         if type == "ppt":
#             m_num = 2
#         else:
#             m_num = 3
#         filename = f"PRISM_ppt_stable_4kmM{m_num}_{date_str}_all_bil.zip"
#         url = f"{url}{year}/{filename}"
        
#         local_path = os.path.join(download_dir_type, filename)
#         unzip_path = os.path.join(unzip_dir_type, date_str)

#         # Skip if already unzipped
#         if not os.path.exists(unzip_path):
#             try:
#                 # Check if file exists
#                 response = requests.head(url)
#                 if response.status_code == 200:
#                     # Download the file
#                     response = requests.get(url, stream=True, timeout=30)
#                     with open(local_path, 'wb') as f:
#                         for chunk in response.iter_content(chunk_size=8192):
#                             f.write(chunk)
                    
#                     # Unzip the file
#                     with zipfile.ZipFile(local_path, 'r') as zip_ref:
#                         zip_ref.extractall(unzip_path)
                    
#                     # Remove zip file
#                     os.remove(local_path)
                
#             except Exception as e:
#                 tqdm.write(f"Error processing {date_str}: {str(e)}")
#             finally:
#                 # Add delay to be polite to server
#                 time.sleep(0.5)

#         current_date += timedelta(years=1)
#         pbar.update(1)
    
#     pbar.close()

# if __name__ == "__main__":
#     for i in range(0, len(types)):
#         download_prism_data(types_url[i], types[i])
    
#     print("Download and extraction complete!")


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