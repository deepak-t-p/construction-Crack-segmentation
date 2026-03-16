"""Download the forked cracks dataset from the user's Roboflow workspace."""
from roboflow import Roboflow

API_KEY = "oCjYQE9b0H20mzgczvov"
rf = Roboflow(api_key=API_KEY)

print("Downloading forked cracks dataset (xyz-gooly/cracks-3ii36-oqlk7 v1)...")
project = rf.workspace("xyz-gooly").project("cracks-3ii36-oqlk7")
version = project.version(1)
version.download("voc", location="data/cracks", overwrite=True)
print("✅ Cracks dataset downloaded to data/cracks/")
