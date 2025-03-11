from pathlib import Path
import pandas as pd

# Define the base directory to search for files.
experiment_name = "test_prnn"

base_dir = Path(f"outputs/{experiment_name}")
df_list = []

# Use rglob to find all files named "results.csv" recursively.
files = [f for f in base_dir.rglob("results.csv")]
print(f"Found {len(files)} files in {base_dir}.")
for i, file in enumerate(files):
    # Check if any part of the file's path contains "" (empty string).
    if any("" in part for part in file.parts):
        print(f"processing file {i}:", file)
        try:
            df = pd.read_csv(file)
            df["source"] = str(file)  # Optionally record the source file.
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

if df_list:
    combined_df = pd.concat(df_list, ignore_index=True)
    filename = f"combined_results_{experiment_name}.csv"
    file_path = Path(base_dir, filename)
    combined_df.to_csv(file_path, index=False)
    print(f"Combined results of {len(df_list)} files saved to {file_path}.")
else:
    print("No CSV files were read.")
