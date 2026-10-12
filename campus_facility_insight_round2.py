# Campus Facility Insight — SAT 2 Round 2
# Synthetic data project. Install: pip install pandas numpy matplotlib seaborn scikit-learn
# 1. Import libraries and set reproducibility
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold

SEED = 42
np.random.seed(SEED)
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)

# If running in Google Colab, upload the ZIP containing datasets OR upload the five CSV files.
# This cell supports either a local 'datasets' folder or Colab file upload.
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

DATA_DIR = Path("datasets")
if not DATA_DIR.exists():
    DATA_DIR = Path("/content/datasets") if Path("/content/datasets").exists() else Path(".")

required_files = ["facilities.csv", "bookings.csv", "occupancy.csv", "calendar.csv", "maintenance.csv"]
missing_files = [f for f in required_files if not (DATA_DIR / f).exists()]
if missing_files and IN_COLAB:
    print("Upload the five CSV files together.")
    uploaded = files.upload()
    DATA_DIR = Path(".")
    missing_files = [f for f in required_files if not (DATA_DIR / f).exists()]
if missing_files:
    raise FileNotFoundError(f"Missing files: {missing_files}. Place the CSVs in a 'datasets' folder or upload them.")

print("Data directory:", DATA_DIR.resolve())


# 2. Load all five datasets
facilities = pd.read_csv(DATA_DIR / "facilities.csv")
bookings = pd.read_csv(DATA_DIR / "bookings.csv")
occupancy = pd.read_csv(DATA_DIR / "occupancy.csv")
calendar = pd.read_csv(DATA_DIR / "calendar.csv")
maintenance = pd.read_csv(DATA_DIR / "maintenance.csv")

datasets = {
    "facilities": facilities,
    "bookings": bookings,
    "occupancy": occupancy,
    "calendar": calendar,
    "maintenance": maintenance
}

for name, df in datasets.items():
    print(f"\\n{name.upper()}: {df.shape[0]} rows x {df.shape[1]} columns")
    display(df.head())


# 3. Initial understanding: data types, nulls, duplicates, descriptive statistics
for name, df in datasets.items():
    print(f"\\n--- {name.upper()} ---")
    print("Data types:")
    display(df.dtypes.to_frame("dtype"))
    print("Missing values:")
    display(df.isna().sum().to_frame("missing_count"))
    print("Exact duplicate rows:", int(df.duplicated().sum()))
    display(df.describe(include="all").T)


# 4. Data quality checks before cleaning
print("Bookings duplicate rows:", bookings.duplicated().sum())
print("Occupancy missing count:", occupancy["count"].isna().sum())
print("Maintenance missing values:")
display(maintenance.isna().sum().to_frame("missing_count"))

# Parse dates/times for validation
bookings["date"] = pd.to_datetime(bookings["date"], errors="coerce")
occupancy["timestamp"] = pd.to_datetime(occupancy["timestamp"], errors="coerce")
calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")
maintenance["date"] = pd.to_datetime(maintenance["date"], errors="coerce")

# Capacity violations using normalized names
name_lookup = {str(n).strip().casefold(): n for n in facilities["facility"]}
occupancy["facility_normalized"] = occupancy["facility"].astype("string").str.strip().str.casefold().map(name_lookup)
capacity_lookup = facilities.set_index("facility")["capacity"].to_dict()
occupancy["capacity_check"] = occupancy["facility_normalized"].map(capacity_lookup)
violations = occupancy[occupancy["count"] > occupancy["capacity_check"]]
print("Occupancy readings above capacity:", len(violations))
display(violations.head())


# 5. Clean and standardize datasets
# Keep original copies for auditability
bookings_raw = bookings.copy()
occupancy_raw = occupancy.copy()
maintenance_raw = maintenance.copy()

# Standardize facility names using a case-insensitive lookup
name_lookup = {str(n).strip().casefold(): n for n in facilities["facility"]}

bookings["facility"] = bookings["facility"].astype("string").str.strip().str.casefold().map(name_lookup)
occupancy["facility"] = occupancy["facility"].astype("string").str.strip().str.casefold().map(name_lookup)
maintenance["facility"] = maintenance["facility"].astype("string").str.strip().str.casefold().map(name_lookup)

# Remove exact duplicate booking rows
bookings = bookings.drop_duplicates().copy()

# Parse dates and times
bookings["date"] = pd.to_datetime(bookings["date"], errors="coerce")
calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")
maintenance["date"] = pd.to_datetime(maintenance["date"], errors="coerce")
occupancy["timestamp"] = pd.to_datetime(occupancy["timestamp"], errors="coerce")

# Fill missing booking purpose with a transparent placeholder
bookings["purpose"] = bookings["purpose"].fillna("Unknown")

# Drop rows with invalid essential identifiers/dates
bookings = bookings.dropna(subset=["facility", "date", "start", "end"])
occupancy = occupancy.dropna(subset=["facility", "timestamp"])
maintenance = maintenance.dropna(subset=["facility", "date"])

# Convert counts and downtime to numeric
occupancy["count"] = pd.to_numeric(occupancy["count"], errors="coerce")
maintenance["downtime"] = pd.to_numeric(maintenance["downtime"], errors="coerce")

# Impute missing occupancy with median occupancy for the same facility.
# Keep a flag to identify imputed observations.
occupancy["count_was_missing"] = occupancy["count"].isna()
facility_medians = occupancy.groupby("facility")["count"].transform("median")
occupancy["count"] = occupancy["count"].fillna(facility_medians)
occupancy["count"] = occupancy["count"].fillna(0)

# Incomplete maintenance fields: fill issue/status explicitly; downtime median by facility then global median
maintenance["issue"] = maintenance["issue"].fillna("Unknown issue")
maintenance["status"] = maintenance["status"].fillna("Unknown")
maintenance["downtime"] = maintenance["downtime"].fillna(
    maintenance.groupby("facility")["downtime"].transform("median")
)
maintenance["downtime"] = maintenance["downtime"].fillna(0)

# Merge capacity into occupancy and flag above-capacity readings; don't silently alter measured counts.
occupancy = occupancy.merge(
    facilities[["facility", "facility_id", "type", "capacity", "block"]],
    on="facility", how="left", validate="many_to_one"
)
occupancy["above_capacity"] = occupancy["count"] > occupancy["capacity"]

# Add calendar context to occupancy readings
occupancy["date"] = occupancy["timestamp"].dt.normalize()
occupancy = occupancy.merge(
    calendar[["date", "working_day", "event", "exam_period"]],
    on="date", how="left", validate="many_to_one"
)

# Merge booking metadata with facilities and calendar
bookings = bookings.merge(
    facilities[["facility", "facility_id", "type", "capacity", "block"]],
    on="facility", how="left", validate="many_to_one"
)
bookings["start_hour"] = pd.to_datetime(bookings["start"], format="%H:%M", errors="coerce").dt.hour
bookings["end_hour"] = pd.to_datetime(bookings["end"], format="%H:%M", errors="coerce").dt.hour
bookings["duration_hours"] = (bookings["end_hour"] - bookings["start_hour"]).clip(lower=0)
bookings = bookings.merge(
    calendar[["date", "working_day", "event", "exam_period"]],
    on="date", how="left", validate="many_to_one"
)

# Maintenance metadata
maintenance = maintenance.merge(
    facilities[["facility", "facility_id", "type", "capacity", "block"]],
    on="facility", how="left", validate="many_to_one"
)

print("Cleaning complete.")
print("Bookings:", bookings.shape)
print("Occupancy:", occupancy.shape)
print("Maintenance:", maintenance.shape)


# 6. Validate cleaned data
print("Missing values after cleaning")
for name, df in {"bookings": bookings, "occupancy": occupancy, "maintenance": maintenance}.items():
    print(f"\\n{name}:")
    display(df.isna().sum().to_frame("missing_count"))

print("Remaining exact duplicate booking rows:", bookings.duplicated().sum())
print("Unknown facility names in bookings:", bookings["facility"].isna().sum())
print("Unknown facility names in occupancy:", occupancy["facility"].isna().sum())
print("Above-capacity occupancy rows (flagged, not deleted):", occupancy["above_capacity"].sum())


# 7. Create analysis-level summaries for facility usage
booking_summary = bookings.groupby("facility", as_index=False).agg(
    booking_count=("booking_id", "count"),
    total_booked_hours=("duration_hours", "sum"),
    average_booking_hours=("duration_hours", "mean")
)
occupancy_summary = occupancy.groupby("facility", as_index=False).agg(
    average_occupancy=("count", "mean"),
    maximum_occupancy=("count", "max"),
    occupancy_readings=("count", "size"),
    above_capacity_readings=("above_capacity", "sum")
)
maintenance_summary = maintenance.groupby("facility", as_index=False).agg(
    maintenance_incidents=("maintenance_id", "count"),
    total_downtime_hours=("downtime", "sum"),
    average_downtime_hours=("downtime", "mean")
)

facility_analysis = facilities.merge(booking_summary, on="facility", how="left")
facility_analysis = facility_analysis.merge(occupancy_summary, on="facility", how="left")
facility_analysis = facility_analysis.merge(maintenance_summary, on="facility", how="left")
numeric_cols = facility_analysis.select_dtypes(include="number").columns
facility_analysis[numeric_cols] = facility_analysis[numeric_cols].fillna(0)

facility_analysis["average_utilization_pct"] = (
    facility_analysis["average_occupancy"] / facility_analysis["capacity"] * 100
).round(2)
facility_analysis["underused_flag"] = facility_analysis["average_utilization_pct"] < 25
display(facility_analysis.sort_values("average_utilization_pct", ascending=False))


# 8. EDA visualization: most-booked facilities
top_booked = facility_analysis.sort_values("booking_count", ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(data=top_booked, x="booking_count", y="facility")
plt.title("Top 10 Facilities by Booking Count")
plt.xlabel("Number of bookings")
plt.ylabel("Facility")
plt.tight_layout()
plt.show()


# 9. EDA visualization: average utilization by facility
util = facility_analysis.sort_values("average_utilization_pct", ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(data=util, x="average_utilization_pct", y="facility")
plt.axvline(25, linestyle="--", label="25% underuse threshold")
plt.title("Average Occupancy as a Percentage of Facility Capacity")
plt.xlabel("Average utilization (%)")
plt.ylabel("Facility")
plt.legend()
plt.tight_layout()
plt.show()


# 10. EDA visualization: peak usage by hour
occupancy["hour"] = occupancy["timestamp"].dt.hour
hourly = occupancy.groupby("hour", as_index=False).agg(
    average_occupancy=("count", "mean"),
    total_readings=("count", "size")
)
plt.figure(figsize=(10, 5))
sns.lineplot(data=hourly, x="hour", y="average_occupancy", marker="o")
plt.title("Average Campus Occupancy by Hour")
plt.xlabel("Hour of day")
plt.ylabel("Average occupancy count")
plt.xticks(sorted(occupancy["hour"].dropna().unique()))
plt.tight_layout()
plt.show()
display(hourly)


# 11. EDA visualization: exam period vs non-exam period
exam_comparison = occupancy.groupby("exam_period", dropna=False).agg(
    average_occupancy=("count", "mean"),
    median_occupancy=("count", "median"),
    readings=("count", "size")
).reset_index()
exam_comparison["period"] = exam_comparison["exam_period"].map({True: "Exam period", False: "Non-exam period"})
plt.figure(figsize=(7, 5))
sns.barplot(data=exam_comparison, x="period", y="average_occupancy")
plt.title("Average Occupancy: Exam vs Non-Exam Period")
plt.xlabel("")
plt.ylabel("Average occupancy count")
plt.tight_layout()
plt.show()
display(exam_comparison)


# 12. EDA visualization: maintenance downtime by facility
downtime_plot = facility_analysis.sort_values("total_downtime_hours", ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(data=downtime_plot, x="total_downtime_hours", y="facility")
plt.title("Top 10 Facilities by Total Maintenance Downtime")
plt.xlabel("Total downtime (hours)")
plt.ylabel("Facility")
plt.tight_layout()
plt.show()


# 13. EDA visualization: capacity vs maximum observed occupancy
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=facility_analysis,
    x="capacity",
    y="maximum_occupancy",
    hue="type",
    s=100
)
max_cap = max(facility_analysis["capacity"].max(), facility_analysis["maximum_occupancy"].max())
plt.plot([0, max_cap], [0, max_cap], linestyle="--", label="Capacity boundary")
plt.title("Facility Capacity vs Maximum Observed Occupancy")
plt.xlabel("Facility capacity")
plt.ylabel("Maximum observed occupancy")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()


# 14. Relationships and patterns
corr_cols = ["capacity", "booking_count", "total_booked_hours", "average_occupancy",
             "maximum_occupancy", "maintenance_incidents", "total_downtime_hours",
             "average_utilization_pct"]
plt.figure(figsize=(10, 7))
sns.heatmap(facility_analysis[corr_cols].corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Facility Usage and Maintenance Metrics")
plt.tight_layout()
plt.show()

print("Top-used facilities by booking count:")
display(facility_analysis.nlargest(5, "booking_count")[["facility", "booking_count", "total_booked_hours"]])
print("Potentially underused facilities (average utilization < 25%):")
display(facility_analysis[facility_analysis["underused_flag"]][["facility", "capacity", "average_utilization_pct", "booking_count"]])
print("Facilities with the most downtime:")
display(facility_analysis.nlargest(5, "total_downtime_hours")[["facility", "maintenance_incidents", "total_downtime_hours"]])


# 15. Feature engineering
# Occupancy-level features
occupancy["hour"] = occupancy["timestamp"].dt.hour
occupancy["day_of_week"] = occupancy["timestamp"].dt.day_name()
occupancy["is_weekend"] = occupancy["timestamp"].dt.dayofweek >= 5
occupancy["utilization_pct"] = (occupancy["count"] / occupancy["capacity"] * 100).round(2)
occupancy["occupancy_band"] = pd.cut(
    occupancy["utilization_pct"],
    bins=[-np.inf, 25, 60, 85, np.inf],
    labels=["Low", "Moderate", "High", "Over-capacity"]
)
occupancy["time_period"] = pd.cut(
    occupancy["hour"],
    bins=[7, 11, 14, 17, 20],
    labels=["Morning", "Midday", "Afternoon", "Evening"],
    right=False
)

# Booking-level features
bookings["booking_day_of_week"] = bookings["date"].dt.day_name()
bookings["is_weekend"] = bookings["date"].dt.dayofweek >= 5
bookings["booking_time_period"] = pd.cut(
    bookings["start_hour"],
    bins=[7, 11, 14, 17, 20],
    labels=["Morning", "Midday", "Afternoon", "Evening"],
    right=False
)
bookings["booking_utilization_proxy_pct"] = np.nan  # No attendee count exists in bookings.csv; avoid inventing it.

display(occupancy[["timestamp", "facility", "count", "capacity", "utilization_pct", "occupancy_band", "hour", "day_of_week", "exam_period"]].head())
display(bookings[["booking_id", "facility", "date", "start", "end", "duration_hours", "purpose", "booking_time_period"]].head())


# 16. Encode categorical variables where appropriate
# One-hot encoding for modeling-ready features; keep original tables unchanged.
feature_source = occupancy[
    ["count", "capacity", "hour", "is_weekend", "exam_period", "working_day",
     "type", "block", "event", "day_of_week", "time_period"]
].copy()

# Normalize missing categories before encoding
for col in ["type", "block", "event", "day_of_week", "time_period"]:
    feature_source[col] = feature_source[col].astype("string").fillna("Unknown")

encoded_features = pd.get_dummies(
    feature_source,
    columns=["type", "block", "event", "day_of_week", "time_period"],
    drop_first=False,
    dtype=int
)
print("Encoded feature matrix shape:", encoded_features.shape)
display(encoded_features.head())


# 17. Feature selection (variance filter)
# Remove constant columns only; this is a simple, defensible selection step.
numeric_feature_matrix = encoded_features.select_dtypes(include=["number"]).copy()
selector = VarianceThreshold(threshold=0.0)
selected_array = selector.fit_transform(numeric_feature_matrix)
selected_feature_names = numeric_feature_matrix.columns[selector.get_support()].tolist()
selected_features = pd.DataFrame(selected_array, columns=selected_feature_names, index=numeric_feature_matrix.index)

print("Features before selection:", numeric_feature_matrix.shape[1])
print("Features after removing zero-variance columns:", selected_features.shape[1])
print("Selected features:")
print(selected_feature_names)
display(selected_features.head())


# 18. Export cleaned datasets and analysis outputs
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

facilities.to_csv(OUTPUT_DIR / "facilities_reference.csv", index=False)
bookings.to_csv(OUTPUT_DIR / "cleaned_bookings.csv", index=False)
occupancy.to_csv(OUTPUT_DIR / "cleaned_occupancy.csv", index=False)
calendar.to_csv(OUTPUT_DIR / "calendar.csv", index=False)
maintenance.to_csv(OUTPUT_DIR / "cleaned_maintenance.csv", index=False)
facility_analysis.to_csv(OUTPUT_DIR / "facility_analysis_summary.csv", index=False)
hourly.to_csv(OUTPUT_DIR / "hourly_occupancy_summary.csv", index=False)
exam_comparison.to_csv(OUTPUT_DIR / "exam_period_comparison.csv", index=False)
selected_features.to_csv(OUTPUT_DIR / "selected_features.csv", index=False)

print("Export complete. Files saved in:", OUTPUT_DIR.resolve())
print("Output files:", sorted(p.name for p in OUTPUT_DIR.iterdir()))


# 19. Download outputs as a ZIP when running in Google Colab
if IN_COLAB:
    import shutil
    shutil.make_archive("Campus_Facility_Insight_Round2_Outputs", "zip", OUTPUT_DIR)
    from google.colab import files
    files.download("Campus_Facility_Insight_Round2_Outputs.zip")
else:
    print("Outputs are in the local 'outputs' folder.")


