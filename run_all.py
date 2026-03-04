import os
import sys

# List of scripts to run in order
# Note: 00_install.py and 01_download_data.py are setup steps —
#       run them manually once before running this pipeline.
scripts = [
    "scripts/02_preprocess.py",
    "scripts/03_eda.py",
    "scripts/04_feature_extraction.py",
    "scripts/05_model_screening.py",
    "scripts/06_stability_testing.py",
    "scripts/07_final_model.py",
    "scripts/08_holdout_eval.py",
    # Note: Streamlit usually blocks execution, so run it manually after this script finishes
]

for script in scripts:
    print(f"--------------------------------------------------")
    print(f"Running {script}...")
    print(f"--------------------------------------------------")
    
    # Run the script and capture the exit code
    exit_code = os.system(f"python {script}")
    
    # Stop if a script fails (non-zero exit code)
    if exit_code != 0:
        print(f"Error: {script} failed! Stopping execution.")
        sys.exit(1)

print("\nAll scripts completed successfully!")
print("Now run: streamlit run app/streamlit_app.py")