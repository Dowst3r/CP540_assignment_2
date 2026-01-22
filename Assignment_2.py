# data for rain and solar is in aberdeenshire/00163_aberdeen-mannofield-resr

import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

# spring months = 1st March -> 31st May (https://weather.metoffice.gov.uk/learn-about/weather/seasons/spring)
seasons = {"Spring":[3,4,5], "Summer":[6,7,8], "Autumn":[9,10,11], "Winter":[12,1,2]}

def required_storage_from_net(net_per_day):
    net = net_per_day.dropna().sort_index()
    cum = net.cumsum()
    drawdown = cum.cummax() - cum
    return float(drawdown.max()), drawdown, cum

def parse_ob_end_time(series):
    s = series.astype(str).str.strip()

    d = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    d = d.fillna(pd.to_datetime(s, format="%Y-%m-%d %H:%M", errors="coerce"))
    d = d.fillna(pd.to_datetime(s, format="%d/%m/%Y %H:%M:%S", errors="coerce"))
    d = d.fillna(pd.to_datetime(s, format="%d/%m/%Y %H:%M", errors="coerce"))
    d = d.fillna(pd.to_datetime(s, errors="coerce", dayfirst=True))
    return d

# Source - https://stackoverflow.com/a
# Posted by Gaurav Singh, modified by community. See post 'Timeline' for change history
# Retrieved 2026-01-16, License - CC BY-SA 4.0

# -------------------------------------
# RAIN DATA EXTRACTION AND ORGANISATION
# -------------------------------------

rain_path = 'rain_data' # use your path - all the rain.csv files are in this folder
all_rain_files = glob.glob(os.path.join(rain_path , "*.csv")) # making a list of all csv files within the folder

def read_midas(rain_path):
    with open(rain_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.lower().startswith("ob_end_time"): # going down until the actual start of the data as first few rows are not important for analysis
                return pd.read_csv(rain_path, skiprows=i)
    raise ValueError(f"Header not found in {rain_path}")

rain_list = []
for filename in all_rain_files:
    df = read_midas(filename)
    rain_list.append(df)

rain_frame = pd.concat(rain_list, ignore_index=True) # merge all the rows into one big data table

rain_frame["ob_end_time"] = parse_ob_end_time(rain_frame["ob_end_time"])

rain_frame["prcp_amt"] = pd.to_numeric(rain_frame["prcp_amt"], errors="coerce")
rain_frame["prcp_dur"] = pd.to_numeric(rain_frame["prcp_dur"], errors="coerce")
rain_frame["ob_hour_count"] = pd.to_numeric(rain_frame["ob_hour_count"], errors="coerce")
rain_frame = rain_frame[
    (rain_frame["ob_hour_count"] == 1) &
    (rain_frame["prcp_dur"].between(0, 60)) &
    (rain_frame["ob_end_time"].dt.minute == 0)
].reset_index(drop=True)

rain_frame = rain_frame.dropna(subset=["ob_end_time", "prcp_amt"]).sort_values("ob_end_time")

rain_daily_mm = (
    rain_frame
    .set_index("ob_end_time")["prcp_amt"]
    .resample("D")
    .sum(min_count=1)          # prevents all-missing days turning into 0
    .sort_index()
)

annual_water_m3 = 200.0
demand_m3_day = annual_water_m3 / 365.0

rain_m3_m2_day = (rain_daily_mm / 1000.0) # mm/day -> m/day; m³ per m² per day

rain_annual_mm = rain_daily_mm.resample("YE").sum(min_count=1).dropna() # Catchment sizing (m²) to meet 200 m³/yr using worst year
rain_annual_m3_per_m2 = rain_annual_mm / 1000.0  # m³ per m² per year

catchment_required_by_year = annual_water_m3 / rain_annual_m3_per_m2
catchment_typical = float(catchment_required_by_year.median())
catchment_worst = float(catchment_required_by_year.max())

inflow_m3_day = rain_m3_m2_day * catchment_worst
net_m3_day = inflow_m3_day - demand_m3_day

tank_m3, tank_drawdown, tank_cum = required_storage_from_net(net_m3_day)
print("Tank capacity needed based on worst rain (m³):", tank_m3)

# 1mm of rain = 1L of water/m^2 of perfectly flat area.
# therefore need an area that the rain is being collected in to reach the desired volume.

# prcp_amt = precipitation 
# prcp_dur = precipitation duration

# --------------------------------------
# SOLAR DATA EXTRACTION AND ORGANISATION
# --------------------------------------

solar_path = 'solar_data' # use your path - all the solar.csv files are in this folder
all_solar_files = glob.glob(os.path.join(solar_path , "*.csv")) # making a list of all csv files within the folder

solar_list = []
for filename in all_solar_files:
    solar_df = read_midas(filename)
    solar_list.append(solar_df)
solar_frame = pd.concat(solar_list, axis=0, ignore_index=True) # stacking each individual csv file table into one big table

solar_frame["ob_end_time"] = parse_ob_end_time(solar_frame["ob_end_time"])
solar_frame = solar_frame.dropna(subset=["ob_end_time"]).copy()

solar_frame["ob_hour_count"] = pd.to_numeric(solar_frame["ob_hour_count"], errors="coerce") # make numeric + keep only true hourly rows
solar_frame["glbl_irad_amt"] = pd.to_numeric(solar_frame["glbl_irad_amt"], errors="coerce")

solar_hourly = solar_frame[solar_frame["ob_hour_count"] == 1].dropna(
    subset=["ob_end_time", "glbl_irad_amt"]
).copy() # hourly rows (true hourly timestamps)

solar_hourly["hour"] = solar_hourly["ob_end_time"].dt.floor("h")

solar_hourly_hourly = (
    solar_hourly.groupby("hour")["glbl_irad_amt"].mean().sort_index()
)

solar_daily_from_hourly = solar_hourly_hourly.resample("D").sum(min_count=1) # daily series made by summing hourly values

solar_daily_direct = solar_frame[solar_frame["ob_hour_count"] == 24].dropna(
    subset=["ob_end_time", "glbl_irad_amt"]
).copy() # daily rows that exist in some years (often timestamped 23:59)

solar_daily_direct["date_solar"] = solar_daily_direct["ob_end_time"].dt.floor("D")

solar_daily_direct = (
    solar_daily_direct.groupby("date_solar")["glbl_irad_amt"].mean().sort_index()
)

# --- scale daily-direct to match hourly-summed daily where they overlap ---
daily_compare = pd.concat(
    [solar_daily_from_hourly.rename("hourly_sum"),
     solar_daily_direct.rename("daily_direct")],
    axis=1
).dropna()

if len(daily_compare) >= 10:
    ratio = (daily_compare["hourly_sum"] / daily_compare["daily_direct"])
    ratio = ratio.replace([float("inf"), float("-inf")], pd.NA).dropna()
    scale = ratio.median()
    # only apply if it looks sensible
    if pd.notna(scale) and scale > 0:
        solar_daily_direct = solar_daily_direct * scale

# now combine: prefer hourly-summed daily values, otherwise scaled daily-direct
solar_daily_all = solar_daily_from_hourly.combine_first(solar_daily_direct).sort_index()

# -----------------------
# SOLAR MAIN CALCULATIONS
# -----------------------

pv_eff = 0.15 # efficiency
annual_demand_kwh = 5000
solar_daily_kj_m2 = solar_daily_all

solar_kwh_m2 = solar_daily_kj_m2 / 3600.0 # kWh/m²/day incident
solar_elec_kwh_m2 = solar_kwh_m2 * pv_eff # kWh/m²/day electricity produced per m² of solar pannels

solar_annual_per_m2 = solar_elec_kwh_m2.resample("YE").sum(min_count=1) # kWh/m²/year

area_required_by_year = annual_demand_kwh / solar_annual_per_m2 # Area needed per year

area_typical = area_required_by_year.median() # typical area required based on solar radiation in
area_worst = area_required_by_year.max() # worst solar year needs biggest area
area_p90 = area_required_by_year.quantile(0.9)

daily_demand_kwh = annual_demand_kwh / 365.0

for label, panel_area_m2 in [("typical", area_typical), ("worst", area_worst)]:
    gen_kwh_day = solar_elec_kwh_m2 * panel_area_m2
    net_kwh_day = gen_kwh_day - daily_demand_kwh
    battery_kwh, _, _ = required_storage_from_net(net_kwh_day)
    print(f"Battery needed (kWh) using {label} panel area:", battery_kwh)

print("typical area required for solar pannels (m²)", area_typical)
print("worst area required for solar pannels (m²)", area_worst)

# ------------------------------------
# SEASONAL ANALYSIS PLOTS (OVER YEARS)
# ------------------------------------

def seasonal_series(daily_series, months, agg="sum"):
    """
    daily_series: pandas Series with DatetimeIndex (daily)
    months: list of month numbers
    agg: "sum" or "mean"
    Returns: Series indexed by "season_year" (int), one value per year for that season.
    Winter is assigned to the year of Jan/Feb (e.g. Dec 1999 belongs to Winter 2000).
    """
    s = daily_series.dropna().copy()
    s = s[s.index.month.isin(months)]

    # season-year assignment (handle winter crossing year boundary)
    if set(months) == {12, 1, 2}:
        season_year = s.index.year + (s.index.month == 12).astype(int)
    else:
        season_year = s.index.year

    if agg == "mean":
        return s.groupby(season_year).mean().sort_index()
    else:
        return s.groupby(season_year).sum().sort_index()
    
def seasonal_totals_with_coverage(daily_series, months, min_days=60):
    s = daily_series.dropna().copy()
    s = s[s.index.month.isin(months)]

    if set(months) == {12, 1, 2}:
        season_year = s.index.year + (s.index.month == 12).astype(int)
    else:
        season_year = s.index.year

    totals = s.groupby(season_year).sum()
    counts = s.groupby(season_year).size()

    return totals[counts >= min_days]  # drop incomplete seasons

def plot_seasonal_lines_over_years(daily_series, title_fmt, ylabel):
    for season_name, months in seasons.items():
        seasonal = seasonal_series(daily_series, months, agg="sum").dropna()

        plt.figure()
        plt.plot(seasonal.index, seasonal.values, marker="o")
        plt.title(title_fmt.format(season=season_name))
        plt.xlabel("Year")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()

plot_seasonal_lines_over_years(
    rain_daily_mm,
    title_fmt="Rainfall total per {season} (mm) over time",
    ylabel="Seasonal rainfall total (mm)"
)

plot_seasonal_lines_over_years(
    solar_daily_all,
    title_fmt="Solar irradiation total per {season} (kJ/m²) over time",
    ylabel="Seasonal solar irradiation total (kJ/m²)"
)

# --------
# BOXPLOTS
# --------

def year_by_year_daily_boxplot(daily_series, title, ylabel, min_days=50, max_years=30):
    s = daily_series.dropna().sort_index()
    years = sorted(pd.Index(s.index.year).unique())

    data, tick = [], []
    for y in years:
        vals = s[s.index.year == y].values
        if len(vals) >= min_days:
            data.append(vals)
            tick.append(str(y))

    if len(data) == 0:
        print(f"[WARN] No years had >= {min_days} daily points for: {title}")
        return

    # keep most recent N years if too many
    if len(data) > max_years:
        data = data[-max_years:]
        tick = tick[-max_years:]

    fig, ax = plt.subplots(figsize=(max(10, len(tick) * 0.35), 5))
    ax.boxplot(data, showfliers=True)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(1, len(tick) + 1))
    ax.set_xticklabels(tick, rotation=90)
    fig.tight_layout()
    plt.show()

def seasonal_totals_boxplot(daily_series, seasons_dict, title, ylabel):
    seasonal_data, tick = [], []

    for season_name, months in seasons_dict.items():
        season_totals = seasonal_series(daily_series, months, agg="sum").dropna()
        seasonal_data.append(season_totals.values)
        tick.append(season_name)

    # guard
    if len(seasonal_data) != len(tick):
        raise ValueError("Seasonal boxplot: labels/data mismatch (should never happen).")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(seasonal_data, showfliers=True)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(1, len(tick) + 1))
    ax.set_xticklabels(tick)
    fig.tight_layout()
    plt.show()

# ---- RAIN: year-by-year boxplots of DAILY rainfall (mm/day)
year_by_year_daily_boxplot(
    rain_daily_mm,
    title="Rainfall: daily distribution by year (mm/day)",
    ylabel="Daily rainfall (mm/day)"
)

# ---- RAIN: 4 seasonal boxes of SEASON TOTAL rainfall (mm) across years
seasonal_totals_boxplot(
    rain_daily_mm,
    seasons_dict=seasons,
    title="Rainfall: seasonal totals variability across years (mm/season)",
    ylabel="Seasonal rainfall total (mm)"
)

# ---- SOLAR: year-by-year boxplots of DAILY solar irradiation (kJ/m²/day)
year_by_year_daily_boxplot(
    solar_daily_all,
    title="Solar irradiation: daily distribution by year (kJ/m²/day)",
    ylabel="Daily solar irradiation (kJ/m²/day)"
)

# ---- SOLAR: 4 seasonal boxes of SEASON TOTAL solar irradiation (kJ/m²) across years
seasonal_totals_boxplot(
    solar_daily_all,
    seasons_dict=seasons,
    title="Solar irradiation: seasonal totals variability across years (kJ/m²/season)",
    ylabel="Seasonal solar irradiation total (kJ/m²)"
)

# --- MINIMA (year + season) ---
# Year totals (one value per year)
rain_annual_totals = rain_daily_mm.resample("YE").sum(min_count=1).dropna()
solar_annual_totals = solar_daily_all.resample("YE").sum(min_count=1).dropna()

print("Min annual rain (mm):", rain_annual_totals.min(), "in", rain_annual_totals.idxmin().year)
print("Min annual solar (kJ/m²):", solar_annual_totals.min(), "in", solar_annual_totals.idxmin().year)

# Season totals per year (one value per year per season)
season_mins = {}
for season_name, months in seasons.items():
    rain_season = seasonal_series(rain_daily_mm, months, agg="sum").dropna()
    solar_season = seasonal_series(solar_daily_all, months, agg="sum").dropna()

    season_mins[season_name] = {
        "rain_min_mm": float(rain_season.min()),
        "rain_min_year": int(rain_season.idxmin()),
        "solar_min_kJm2": float(solar_season.min()),
        "solar_min_year": int(solar_season.idxmin()),
    }

print("\nSeason minimums (totals across years):")
for k, v in season_mins.items():
    print(k, v)

panel_area_m2 = area_typical
net_kwh_day = gen_kwh_day - daily_demand_kwh

battery_kwh, battery_drawdown, battery_cum = required_storage_from_net(net_kwh_day)
print("Battery capacity needed (kWh):", battery_kwh)

charging_days = (net_kwh_day > 0).sum()
draining_days  = (net_kwh_day < 0).sum()
print("Charging days:", charging_days, "Draining days:", draining_days)

def season_net_totals(net_series, seasons_dict):
    out = {}
    for season_name, months in seasons_dict.items():
        season_net = seasonal_series(net_series, months, agg="sum").dropna()
        out[season_name] = season_net
        print(season_name, "median net:", float(season_net.median()), "min net:", float(season_net.min()))
    return out

season_net = season_net_totals(net_kwh_day, seasons)

winter_months = seasons["Winter"]
winter_net_by_year = seasonal_series(net_kwh_day, winter_months, agg="sum").dropna()
worst_winter_year = int(winter_net_by_year.idxmin())
print("Worst winter net (kWh) in year:", worst_winter_year, "value:", float(winter_net_by_year.min()))

def count_boxplot_outliers_by_year(daily_series, whisker=1.5, min_days=50):
    """"
    Returns a DataFrame with per-year:
      n_days, q1, q3, iqr, lower_fence, upper_fence, n_outliers, pct_outliers
    Outliers are defined like matplotlib boxplots: outside [Q1 - w*IQR, Q3 + w*IQR].
    """
    s = daily_series.dropna().sort_index()
    years = sorted(pd.Index(s.index.year).unique())

    rows = []
    for y in years:
        vals = s[s.index.year == y].values
        if len(vals) < min_days:
            continue

        q1 = np.percentile(vals, 25)
        q3 = np.percentile(vals, 75)
        iqr = q3 - q1

        lower = q1 - whisker * iqr
        upper = q3 + whisker * iqr

        n_out = int(((vals < lower) | (vals > upper)).sum())
        rows.append({
            "year": y,
            "number of full days that year": int(len(vals)),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "lower_fence": float(lower),
            "upper_fence": float(upper),
            "number of outliers that year": n_out,
            "percentage of outliers that year": 100.0 * n_out / len(vals),
        })

    return pd.DataFrame(rows).set_index("year")
out_rain_rainydays = count_boxplot_outliers_by_year(
    rain_daily_mm[rain_daily_mm > 0],
    whisker=1.5,
    min_days=50
)

print(out_rain_rainydays[["number of full days that year", "number of outliers that year", "percentage of outliers that year"]])