# smps-plot

Code-only Kigali SMPS particle size distribution plotting and analysis workflow.

This repository keeps the Python and PowerShell scripts used to process, plot, and analyze SMPS particle size distribution data. Raw instrument files, selected AIM exports, merged data tables, generated figures, and supporting PM/met data are intentionally not tracked in GitHub.

The scripts expect data to be available locally in the Kigali working folder or supplied through command-line paths.

## Repository Layout

| Path | What it contains |
| --- | --- |
| `tools/` | Main curated Kigali SMPS processing and plotting scripts. |
| Root legacy `*.py` scripts | Original scripts that were already present in the GitHub repository before this cleanup. |
| `external_scripts/smps_python_code/` | Legacy and comparison scripts copied from `D:\Documents\PhD-Research\SMPS python code`. |
| `external_scripts/smps_comparison/` | Related scripts copied from `D:\Documents\PhD-Research\SMPS Comparison`. |
| `requirements.txt` | Python packages used by the plotting and analysis scripts. |

## Main Scripts

| Script | Purpose |
| --- | --- |
| `tools/merge_aim_txt_exports.py` | Merges AIM SMPS tab-delimited exports into long, wide, and scan-summary CSV files. |
| `tools/plot_daily_smps_from_master.py` | Builds time-ordered master SMPS files and daily 24-hour dN/dlogDp contour plots. |
| `tools/plot_smps_contour.py` | Plots dN/dlogDp contours and average PSD curves for one or more AIM text exports. |
| `tools/analyze_smps_psd_basic.py` | Computes PSD-only scan metrics, daily summaries, size-range metrics, and methods notes from a merged wide file. |
| `tools/plot_kigali_smps_baseline_psd.py` | Creates campaign-average, seasonal, and diurnal baseline PSD visualizations. |
| `tools/batch_smps_acsm_pm25_daily.py` | Runs the daily Kigali SMPS, ACSM/BC, PM2.5, condensation sink, and candidate-event comparison workflow. |
| `tools/plot_contour_cs_example.py` | Makes a single-day SMPS contour with sulfuric-acid condensation-sink overlay. |
| `tools/plot_cs_distribution.py` | Summarizes and plots campaign condensation-sink distributions from daily contour outputs. |
| `tools/rename_aim_txt_by_time.py` | Renames AIM exports using the first and last scan timestamps inside each file. |
| `tools/make_aim_inventory.ps1` | PowerShell helper for inventorying AIM instrument/export files. |

## External Script Archive

The `external_scripts/` folder keeps earlier scripts used for PSD plotting, contour plotting, Rwanda/Kigali time series, ACSM-SMPS comparison, and other SMPS sites. These scripts are preserved for traceability and reuse, but the `tools/` folder is the curated Kigali workflow.

## Quick Start

Create an environment and install the plotting stack:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Example workflow using local, untracked data folders:

```powershell
python tools\merge_aim_txt_exports.py "final files selected" --out "merged outputs\master_all_processed"
python tools\plot_daily_smps_from_master.py --wide "merged outputs\master_all_processed\smps_merged_wide.csv" --summary "merged outputs\master_all_processed\smps_scan_summary.csv" --out "merged outputs\master_all_processed"
python tools\analyze_smps_psd_basic.py --input "merged outputs\master_all_processed\smps_master_wide_time_ordered.csv" --out "outputs\psd_basic_analysis"
python tools\plot_kigali_smps_baseline_psd.py --raw-wide "merged outputs\master_all_processed\smps_master_wide_time_ordered.csv" --summary-dir "outputs\psd_basic_analysis" --output-dir "outputs\psd_basic_analysis"
```

Run the daily SMPS + ACSM/BC + PM2.5 workflow:

```powershell
python tools\batch_smps_acsm_pm25_daily.py --smps-wide "merged outputs\master_all_processed\smps_master_wide_time_ordered.csv" --smps-metrics "outputs\psd_basic_analysis\smps_psd_scan_metrics.csv" --output "outputs\smps_acsm_pm25_batch"
```

## Data Policy

This repository is code-only. The following local folders and file types are ignored:

- Raw SMPS/AIM files: `*.S80`, `*.p80`, `*.p72`
- Selected text exports: `final files selected/`
- Processed products: `merged outputs/`, `outputs/`
- Generated figures: `figures/`, `examples/`
- Supporting local data: `met data/`, `other data/`

Keep those data products locally, in an institutional storage location, or in a separate data release if needed.
