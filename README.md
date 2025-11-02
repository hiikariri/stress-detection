# Stress Detection — PPG Feature Extraction

This repository contains tools to process photoplethysmography (PPG) signals and extract features useful for stress detection and HRV analysis. The main interactive application is a Streamlit app (`ppg_feature_extraction.py`) that guides you through preprocessing, DWT-based respiratory/vasometric extraction, time-domain HRV analysis, frequency-domain HRV analysis, and non-linear (Poincaré) features.

## Features

- Preprocessing: resampling, bandpass filtering, Z-score normalization.
- Peak detection and pulse-interval (PI / RR) extraction.
- Time-domain HRV features and tachogram plotting.
- Histogram of RR/PI intervals with time-based bin width (seconds or milliseconds).
- Frequency-domain HRV analysis (Welch PSD, LF/HF, band powers).
- DWT decomposition for respiratory and vasometric feature extraction.
- Non-linear analysis: Poincaré plot and SD1/SD2 metrics.
- Export extracted features to CSV.

## Repository structure

- `ppg_feature_extraction.py` — main Streamlit application.
- `lib/` — helper modules used by the app:
  - `bmepy.py`, `dwt.py`, `frequency_analysis.py`, `poincare.py`, `time_domain.py`, etc.
- `data/` — example data files (PPG CSVs and ground-truth RR/respiration files).
- `LICENSE` — project license.

Files you may see in this workspace (non-exhaustive):
- `data/data_davis_50hz_rr.csv`, `data/data_davis_50hz.csv`, `data/davis_3_downsampled.csv`, etc.

## Prerequisites

- Windows, macOS, or Linux
- Python 3.8+ (3.9/3.10 recommended)

Python packages (install the ones below). There is no `requirements.txt` in the repo; install these manually:

- streamlit
- numpy
- pandas
- scipy
- plotly

Optional (used by helper modules):
- matplotlib (for some plots)

## Quick setup

1. Create and activate a virtual environment (PowerShell example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install required packages:

```powershell
pip install --upgrade pip
pip install streamlit numpy pandas scipy plotly matplotlib
```

3. Run the Streamlit app:

```powershell
streamlit run ppg_feature_extraction.py
```

## How to use the app

1. Launch the app with Streamlit (see above).
2. Upload a PPG CSV file. The app expects a CSV where the PPG signal column is named `ir` (case-insensitive). If not present, the first column will be used.
3. Set the original sampling rate and desired target sampling rate.
4. Adjust peak detection parameters on the main page (peak height, distance, prominence) until peaks are detected reliably.
5. Navigate pages using the sidebar radio menu:
   - Respiratory & Vasometric Extraction: DWT decomposition and respiratory/vasometric features.
   - Time Domain Analysis: Tachogram, PI intervals, histogram of RR intervals, and time-domain HRV features.
   - Frequency Domain Analysis: Welch PSD and band-power metrics.
   - Non-Linear Analysis: Poincaré plot and SD1/SD2 metrics.
   - All Features & Export: view and download all extracted features as CSV.

### Histogram bin width

The Time Domain Analysis page provides a histogram of RR/PI intervals. You can choose units (`seconds` or `milliseconds`) and enter a time-based bin width (for example, `8` for 8 ms when using milliseconds, or `0.008` when using seconds). The histogram uses Plotly's `xbins` (start/end/size) so the bins correspond to the time interval you specify.

## Example data

Some example PPG and RR files are under the `data/` folder. Use them to test the app. If your data uses different column names or formatting, open the CSV to ensure the PPG column is available or adjust the loader.

## Notes & troubleshooting

- "Not enough peaks detected": tune `Peak Height`, `Peak Distance`, and `Peak Prominence` in the main page until the filtered PPG signal shows clear local maxima.
- If the uploaded ground-truth RR file fails to load, ensure it is a simple two-column CSV (time,force) or a comma-separated file where the second column contains the respiration/force values.
- If the app fails with plotting errors, ensure the required packages (`plotly`, `matplotlib`) are installed in the active environment.

## Development notes

- The code uses helper modules in `lib/`. If you change those modules, re-run the Streamlit app. Streamlit does hot-reload on save but sometimes caches require a restart.
- To add unit tests, consider using `pytest` and adding a `tests/` folder for small fast-running tests for the helper functions.

## Contributing

Contributions are welcome. If you plan to change processing or add features:

1. Fork the repository and create a branch for your feature.
2. Add tests when changing logic.
3. Open a pull request describing the changes and include screenshots if the UI changes.

## License

See the `LICENSE` file in the repository root for license details.

## Contact / Authors

This project contains code by the repository maintainers and contributors. See the project files for author credits.