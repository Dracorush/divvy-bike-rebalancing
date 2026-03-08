"""
Feature Engineering Specifications

Input:
A cleaned Divvy trip dataframe.

Output:
A station-day feature dataframe built from trip-level ride records.

Feature Engineering Rules

1. Derive trip-level date columns from timestamps.
2. Aggregate rides into station-day departures and arrivals.
3. Build a dense station-day calendar so quiet days are included.
4. Construct station-day flow features for later forecasting.

Assumptions

1. The cleaned input dataframe already has valid timestamps and station identifiers.
2. A trip departure belongs to the start station on the trip start date.
3. A trip arrival belongs to the end station on the trip end date.
4. A station is considered active beginning on the first date it appears either as a start station or an end station.
5. Once a station becomes active, every date through the dataset end should appear in the station-day calendar, even if no trips occur on that date.
6. Missing departures or arrivals after merging with the dense station-day calendar represent zero observed trips, not unknown values.

Available Columns

ride_id
rideable_type
started_at
ended_at
start_station_name
start_station_id
end_station_name
end_station_id
start_lat
start_lng
end_lat
end_lng
member_casual
"""

import pandas as pd


# derive trip_date from started_at for daily aggregation
def add_trip_date_column(df):

    df = df.copy()

    df["trip_date"] = df["started_at"].dt.normalize()

    return df


# aggregate trips into daily departures and arrivals by station
def build_daily_flows(df):

    departures = (
        df.groupby(["start_station_id", "trip_date"])
          .size()
          .reset_index(name="trips_departed")
          .rename(columns={"start_station_id": "station_id"})
    )

    arrivals = (
        df.assign(arrival_date=df["ended_at"].dt.normalize())
          .groupby(["end_station_id", "arrival_date"])
          .size()
          .reset_index(name="trips_arrived")
          .rename(columns={
              "end_station_id": "station_id",
              "arrival_date": "trip_date"
          })
    )

    daily_flows = departures.merge(
        arrivals,
        on=["station_id", "trip_date"],
        how="outer"
    )

    daily_flows["trips_departed"] = daily_flows["trips_departed"].fillna(0).astype(int)
    daily_flows["trips_arrived"] = daily_flows["trips_arrived"].fillna(0).astype(int)

    return daily_flows


# build a dense station-day calendar so quiet days are included
def build_station_day_calendar(df):

    start_dates = (
        df.groupby("start_station_id")["trip_date"]
          .min()
          .reset_index()
          .rename(columns={"start_station_id": "station_id"})
    )

    end_dates = (
        df.assign(arrival_date=df["ended_at"].dt.normalize())
          .groupby("end_station_id")["arrival_date"]
          .min()
          .reset_index()
          .rename(columns={
              "end_station_id": "station_id",
              "arrival_date": "first_arrival_date"
          })
    )

    station_bounds = start_dates.merge(end_dates, on="station_id", how="outer")

    station_bounds["trip_date"] = pd.to_datetime(station_bounds["trip_date"])
    station_bounds["first_arrival_date"] = pd.to_datetime(station_bounds["first_arrival_date"])

    station_bounds["first_active_date"] = station_bounds[
        ["trip_date", "first_arrival_date"]
    ].min(axis=1)

    dataset_end = max(df["trip_date"].max(), df["ended_at"].dt.normalize().max())

    calendar_parts = []

    for _, row in station_bounds.iterrows():
        station_id = row["station_id"]
        first_active_date = row["first_active_date"]

        station_dates = pd.date_range(
            start=first_active_date,
            end=dataset_end,
            freq="D"
        )

        station_calendar = pd.DataFrame({
            "station_id": station_id,
            "trip_date": station_dates
        })

        calendar_parts.append(station_calendar)

    station_day_calendar = pd.concat(calendar_parts, ignore_index=True)

    return station_day_calendar


# merge dense station-day calendar with observed flows
def build_station_day_flow_table(df):

    df = add_trip_date_column(df)

    daily_flows = build_daily_flows(df)
    station_day_calendar = build_station_day_calendar(df)

    station_day_flows = station_day_calendar.merge(
        daily_flows,
        on=["station_id", "trip_date"],
        how="left"
    )

    station_day_flows["trips_departed"] = station_day_flows["trips_departed"].fillna(0).astype(int)
    station_day_flows["trips_arrived"] = station_day_flows["trips_arrived"].fillna(0).astype(int)

    station_day_flows["trips_departed"] = pd.to_numeric(
        station_day_flows["trips_departed"],
        downcast="unsigned"
    )

    station_day_flows["trips_arrived"] = pd.to_numeric(
        station_day_flows["trips_arrived"],
        downcast="unsigned"
    )

    station_day_flows["station_id"] = station_day_flows["station_id"].astype("string")

    return station_day_flows

# add previous-day flow features for each station
def add_lag_features(df):

    df = df.copy()

    df = df.sort_values(["station_id", "trip_date"])

    df["trips_departed_prev"] = (
        df.groupby("station_id")["trips_departed"]
          .shift(1)
    )

    df["trips_arrived_prev"] = (
        df.groupby("station_id")["trips_arrived"]
          .shift(1)
    )

    df["trips_departed_prev"] = pd.to_numeric(
        df["trips_departed_prev"],
        downcast="unsigned"
    )

    df["trips_arrived_prev"] = pd.to_numeric(
        df["trips_arrived_prev"],
        downcast="unsigned"
    )

    return df

# remove rows that lack lag context
def drop_missing_lag_rows(df):

    df = df.dropna(subset=[
        "trips_departed_prev",
        "trips_arrived_prev"
    ])

    return df