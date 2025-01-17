import requests
import hashlib
import zipfile
import os

datasetUrl = "https://github.com/microsoft/lamar-benchmark/releases/download/v1.1/NavVisTestData.zip"
expectedSha1 = "0a28991004e58d9dc51167ea23f9f2ba66b26ab5"
zipFileName = "NavVisTestData.zip"

print("Downloading NavVis test data...")

if(os.path.exists(zipFileName)):
    os.remove(zipFileName)

if(os.path.exists("test_data")):
    os.rmdir("test_data")

response = requests.get(datasetUrl)
if response.status_code == 200:
    sha1 = hashlib.sha1(response.content).hexdigest()
    if sha1 == expectedSha1:
        with open(zipFileName, "wb") as file:
            file.write(response.content)
        with zipfile.ZipFile(zipFileName, "r") as zip_ref:
            zip_ref.extractall(".")
        os.remove(zipFileName)
        print("NavVis test data downloaded and extracted successfully.")
    else:
        print("SHA1 hash mismatch. The downloaded file may be corrupted.")
else:
    print("Failed to download the zip file from", datasetUrl)

