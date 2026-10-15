
import os
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Search for MATLAB files in the project
mat_files = list(BASE_DIR.rglob("*.mat"))

report_path = RESULTS_DIR / "raw_dataset_inspection.txt"

with open(report_path, "w", encoding="utf-8") as report:

    report.write("RAW DATASET INSPECTION REPORT\n")
    report.write("=" * 50 + "\n\n")

    if not mat_files:
        report.write("No .mat files found in the project directory.\n")
        print("No MATLAB files found. Check your dataset location.")

    else:
        report.write(f"Total MATLAB files found: {len(mat_files)}\n\n")

        for file in mat_files[:30]:
            report.write(f"File: {file.name}\n")
            report.write(f"Location: {file}\n")
            report.write("-" * 40 + "\n")

            try:
                from scipy.io import whosmat

                variables = whosmat(file)

                for name, shape, dtype in variables:
                    report.write(
                        f"Variable: {name}, "
                        f"Shape: {shape}, "
                        f"Type: {dtype}\n"
                    )

            except Exception as e:
                report.write(f"Could not inspect file: {e}\n")

            report.write("\n")

print("Dataset inspection completed!")
print(f"Report saved at: {report_path}")