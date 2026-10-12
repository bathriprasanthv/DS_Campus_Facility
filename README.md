# Campus Facility Insight — Data Science SAT 2 (Round 2)

## Included
- `Campus_Facility_Insight_SAT2.ipynb`: complete step-by-step Python notebook.
- `campus_facility_insight_round2.py`: Python script version.
- `datasets/`: five synthetic CSV datasets matching the use-case file names.
- `outputs/`: folder for generated analysis results (initially empty; run the notebook to populate).
- `requirements.txt`: dependencies.

## Run in Google Colab
1. Open https://colab.research.google.com/
2. Upload `Campus_Facility_Insight_SAT2.ipynb`.
3. Upload the five CSV files from `datasets/` into the notebook session, or upload the ZIP and extract it.
4. Run cells from top to bottom.
5. Download the output ZIP at the final cell.

## Data integrity note
All supplied CSVs are synthetic/simulated data generated for implementation practice, not actual campus records. The data intentionally includes missing values, exact duplicate booking rows, inconsistent facility name capitalization, occupancy above capacity, and incomplete maintenance records so the cleaning tasks can be demonstrated. Confirm whether synthetic data is permitted by your instructor and disclose it if required.

## Main steps covered
1. Load and understand data
2. Check data types, missing values, duplicates, unusual values
3. Clean and combine datasets
4. Perform descriptive analysis and visualizations
5. Study patterns and relationships
6. Encode categorical variables
7. Create new features
8. Apply simple feature selection (remove zero-variance features)
9. Export cleaned data and summaries
