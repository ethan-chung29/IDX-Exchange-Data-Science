"""School-district layer: spatial join of each home to the CA School District Areas 2024-25.

Source (game plan Week 6): https://data.ca.gov/dataset/california-school-district-areas-2024-25
Download the shapefile ZIP to data/external/ca_school_district_areas_2024_25.zip (git-ignored):
    curl -L -o data/external/ca_school_district_areas_2024_25.zip \\
      "https://gis.data.ca.gov/api/download/v1/items/b0e3b936426a47ce9d9a2e77e2bb86cc/shapefile?layers=0"

Every California home lies in either one Unified (K-12) district, or one Elementary plus one
High district. Each home gets:
- school_district_k8: the Elementary district if there is one, otherwise the Unified district
- school_district_hs: the High district if there is one, otherwise the Unified district
- district_type: "Unified" or "Elementary + High"
- district_k8_enrollment, district_k8_assistance: size and state accountability status
  ("General" / "Differentiated, Year 1/2") of the K-8 district

Deliberately NOT used: the layer's race/ethnicity, English-learner, foster/homeless/migrant and
socioeconomically-disadvantaged shares. A valuation model that learns from neighborhood
demographics can discount homes by who lives nearby (the fair-lending risk AVM rules target),
so those columns are dropped here and never reach the model.

These are fixed public facts about a location, not learned from sale prices, so the join is
safe to run on any split. Note: boundaries and enrollment are the 2024-25 snapshot, applied to
sales from 2024-2026.

Usage (from the repo root, after cleaning.py):
    python src/geo.py        # writes data/processed/school_districts.parquet
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DISTRICTS_ZIP = os.path.join(REPO_ROOT, "data", "external", "ca_school_district_areas_2024_25.zip")
OUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "school_districts.parquet")

KEEP = ["CDSCode", "DistrictNa", "DistrictTy", "EnrollTota", "AssistStat", "geometry"]
COLUMNS = ["school_district_k8", "school_district_hs", "district_type",
           "district_k8_enrollment", "district_k8_assistance"]


def load_districts(path=DISTRICTS_ZIP):
    g = gpd.read_file(f"zip://{path}")[KEEP]
    g["geometry"] = g.geometry.make_valid()  # a few boundaries self-intersect
    # Name plus CDS code: unique even when two counties have same-named districts
    g["label"] = g["DistrictNa"] + " [" + g["CDSCode"] + "]"
    return g


def assign_districts(lat, lon, districts=None, index=None):
    """District features for each (lat, lon). Rows with no coordinates or no district get NaN."""
    districts = load_districts() if districts is None else districts
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    index = pd.RangeIndex(len(lat)) if index is None else pd.Index(index)
    out = pd.DataFrame(np.nan, index=index, columns=COLUMNS).astype(
        {c: "object" for c in ["school_district_k8", "school_district_hs", "district_type", "district_k8_assistance"]})

    ok = ~(np.isnan(lat) | np.isnan(lon))
    pts = gpd.GeoDataFrame({"row": np.flatnonzero(ok)},
                           geometry=gpd.points_from_xy(lon[ok], lat[ok]), crs="EPSG:4326").to_crs(districts.crs)
    hits = gpd.sjoin(pts, districts, how="inner", predicate="within")

    def pick(kind_first):
        # Per home: the first matching district type in order of preference
        h = hits[hits["DistrictTy"].isin(kind_first)].copy()
        h["rank"] = h["DistrictTy"].map({k: i for i, k in enumerate(kind_first)})
        return h.sort_values(["row", "rank"]).drop_duplicates("row").set_index("row")

    k8 = pick(["Elementary", "Unified"])
    hs = pick(["High", "Unified"])
    out.iloc[k8.index, out.columns.get_loc("school_district_k8")] = k8["label"].to_numpy()
    out.iloc[k8.index, out.columns.get_loc("district_k8_enrollment")] = k8["EnrollTota"].to_numpy()
    out.iloc[k8.index, out.columns.get_loc("district_k8_assistance")] = k8["AssistStat"].to_numpy()
    out.iloc[hs.index, out.columns.get_loc("school_district_hs")] = hs["label"].to_numpy()
    both = out["school_district_k8"].notna() & out["school_district_hs"].notna()
    out.loc[both, "district_type"] = np.where(
        out.loc[both, "school_district_k8"] == out.loc[both, "school_district_hs"], "Unified", "Elementary + High")
    return out


def lookup(lat, lon, districts=None):
    """District features for a single home (used by the prediction app)."""
    return assign_districts([lat], [lon], districts).iloc[0].to_dict()


def main():
    clean = pd.read_parquet(os.path.join(REPO_ROOT, "data", "processed", "sfr_clean.parquet"),
                            columns=["ListingKey", "Latitude", "Longitude"])
    out = assign_districts(clean["Latitude"], clean["Longitude"], index=clean["ListingKey"])
    out.index.name = "ListingKey"
    out.reset_index().to_parquet(OUT_PATH, index=False)
    print(f"{out['school_district_k8'].notna().mean():.2%} of {len(out):,} homes matched to a district "
          f"({clean['Latitude'].isna().sum():,} have no coordinates)")
    print(out["district_type"].value_counts(dropna=False).to_string())
    print(f"{out['school_district_k8'].nunique()} K-8 districts, {out['school_district_hs'].nunique()} high-school districts")


if __name__ == "__main__":
    main()
