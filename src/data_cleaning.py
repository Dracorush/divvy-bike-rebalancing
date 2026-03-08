"""
Cleaning Specifications

Input:
A raw Divvy monthly trip dataframe with already-validated column structure.

Output:
A cleaned dataframe ready to be merged and written to parquet.

Cleaning Rules

1. Convert started_at and ended_at to pandas datetime objects.
2. Ensure latitude and longitude columns are numeric.
3. Ensure identifier and categorical columns are stored as strings:
   ride_id
   rideable_type
   start_station_id
   end_station_id
   member_casual
4. Remove rows containing missing critical values:
   start_station_id
   end_station_id
   started_at
   ended_at
   start_lat
   start_lng
   end_lat
   end_lng
"""

import pandas as pd

def clean_divvy_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
    df["ended_at"] = pd.to_datetime(df["ended_at"], errors="coerce")

    df["start_lat"] = pd.to_numeric(df["start_lat"], errors="coerce")
    df["start_lng"] = pd.to_numeric(df["start_lng"], errors="coerce")
    df["end_lat"] = pd.to_numeric(df["end_lat"], errors="coerce")
    df["end_lng"] = pd.to_numeric(df["end_lng"], errors="coerce")

    cat_cols = [
        "ride_id",
        "rideable_type",
        "start_station_id",
        "end_station_id",
        "member_casual"
    ]

    df[cat_cols] = df[cat_cols].astype("string")

    df = df.dropna(subset=[
        "start_station_id",
        "end_station_id",
        "started_at",
        "ended_at",
        "start_lat",
        "start_lng",
        "end_lat",
        "end_lng"
    ]).copy()

    return df