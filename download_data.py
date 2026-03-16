"""Download datasets from Roboflow for the Drywall QA project."""
from roboflow import Roboflow

API_KEY = "oCjYQE9b0H20mzgczvov"

rf = Roboflow(api_key=API_KEY)

# ── 1. Cracks dataset ──
print("\n[1/2] Downloading Cracks dataset...")
try:
    project1 = rf.workspace("fyp-ny1jt").project("cracks-3ii36")
    # List versions
    versions1 = project1.versions()
    print(f"  Found {len(versions1)} version(s)")
    if versions1:
        # Try getting the version object directly
        v = versions1[0]
        print(f"  Version info: {v}")
        v.download("voc", location="data/cracks", overwrite=True)
        print(f"  -> Saved to: data/cracks/")
    else:
        # Try common version numbers
        for vnum in [1, 2, 3]:
            try:
                v = project1.version(vnum)
                v.download("voc", location="data/cracks", overwrite=True)
                print(f"  -> Downloaded version {vnum} to: data/cracks/")
                break
            except Exception as e:
                print(f"  Version {vnum}: {e}")
except Exception as e:
    print(f"  [ERROR] {e}")

# ── 2. Drywall Joint Detect dataset ──
print("\n[2/2] Downloading Drywall Join Detect dataset...")
try:
    project2 = rf.workspace("objectdetect-pu6rn").project("drywall-join-detect")
    versions2 = project2.versions()
    print(f"  Found {len(versions2)} version(s)")
    if versions2:
        v = versions2[0]
        print(f"  Version info: {v}")
        v.download("voc", location="data/drywall-joints", overwrite=True)
        print(f"  -> Saved to: data/drywall-joints/")
    else:
        for vnum in [1, 2, 3]:
            try:
                v = project2.version(vnum)
                v.download("voc", location="data/drywall-joints", overwrite=True)
                print(f"  -> Downloaded version {vnum} to: data/drywall-joints/")
                break
            except Exception as e:
                print(f"  Version {vnum}: {e}")
except Exception as e:
    print(f"  [ERROR] {e}")

print("\n✅ Done!")
