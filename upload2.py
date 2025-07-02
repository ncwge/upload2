import streamlit as st
import pandas as pd
import re
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.title("upload2")

st.header("Step 1: Enter Competitor SKUs")
uploaded_file = st.file_uploader("Upload Excel file with SKUs", type=["xlsx", "xls"])
pasted_data = st.text_area("Paste competitor SKU data here:")

def extract_skus_from_excel(df):
    all_text = df.astype(str).values.flatten()
    sku_pattern = re.compile(r"\b[A-Z]{2,}[0-9]{2,}[A-Z0-9]*\b")
    skus = []
    seen = set()
    for text in all_text:
        matches = sku_pattern.findall(text)
        for match in matches:
            if len(match) >= 6 and match not in seen:
                skus.append(match)
                seen.add(match)
    return skus

def extract_skus_from_text(text):
    sku_pattern = re.compile(r"\b[A-Z]{2,}[0-9]{2,}[A-Z0-9]*\b")
    skus = []
    seen = set()
    for line in text.upper().splitlines():
        matches = sku_pattern.findall(line)
        for sku in matches:
            if len(sku) >= 6 and sku not in seen:
                skus.append(sku)
                seen.add(sku)
    return skus

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

skus = []
if uploaded_file:
    df_upload = pd.read_excel(uploaded_file, header=None)
    skus = extract_skus_from_excel(df_upload)
elif pasted_data.strip():
    skus = extract_skus_from_text(pasted_data)

if skus:
    st.success(f"✅ Found {len(skus)} unique SKUs:")
    st.dataframe(pd.DataFrame({'SKU': skus}))
    excel_data = to_excel(pd.DataFrame({'SKU': skus}))
    st.download_button("Download SKUs to Excel", data=excel_data, file_name="sku_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.header("Step 2: Upload Appliance Catalog (Tall Format, Multiple Sheets OK)")
appliance_file = st.file_uploader(
    "Upload catalog Excel (tall, features as columns, multiple sheets allowed)", 
    type=["xlsx", "xls"], 
    key="appliance_upload"
)
if not appliance_file:
    st.stop()

all_sheets = pd.read_excel(appliance_file, sheet_name=None)
sheet_lookup = {}
required = ['SKU', 'Brand', 'Model Status', 'Configuration']

# PATCHED extract_width
# Handles fractions, decimals, various inch formats

def extract_width(description):
    if pd.isnull(description):
        return np.nan
    desc = str(description).lower()
    # 29 3/4, 30 1/2, 29.75, etc.
    frac_match = re.search(r'(\d{2,3})\s+(\d{1,2})/(\d{1,2})', desc)
    if frac_match:
        whole = float(frac_match.group(1))
        num = float(frac_match.group(2))
        denom = float(frac_match.group(3))
        value = whole + num / denom
        return value
    # 30", 30 in, 30-in, 30inch, etc. (also fallback for decimals)
    match = re.search(
        r'(\d{2,3}(?:\.\d+)?)(?:\s*[-]?)\s*(?:"|”|in\.?|inch(?:es)?|\-in\b)',
        desc
    )
    if match:
        return float(match.group(1))
    return np.nan


def extract_capacity(description):
    if pd.isnull(description):
        return np.nan
    # Handles: 1.9 cu ft, 1.9cu.ft, 1.9cuft, 1.9 "cu ft", etc
    match = re.search(r'(\d+(?:\.\d+)?)\s*(cu\.?\s*ft|cuft|cubic\s*feet|cubic\s*ft)', str(description).lower())
    if match:
        return float(match.group(1))
    return np.nan

  
def extract_wattage(description):
    if pd.isnull(description):
        return np.nan
    match = re.search(r'(\d{3,4})\s*(w|watt)', str(description).lower())
    if match:
        return float(match.group(1))
    return np.nan

def parse_power_range(val):
    if pd.isnull(val):
        return None, None
    # Try for range: 1000-1200
    match = re.match(r"(\d{3,4})\s*-\s*(\d{3,4})", str(val))
    if match:
        return float(match.group(1)), float(match.group(2))
    # Try for '1000-1200 Range' etc
    match = re.match(r"(\d{3,4})\s*-\s*(\d{3,4})\s*range", str(val).lower())
    if match:
        return float(match.group(1)), float(match.group(2))
    # Single number: 1000, '1100W'
    match = re.match(r"(\d{3,4})", str(val))
    if match:
        n = float(match.group(1))
        return n, n
    return None, None

discard_cols = ['SKU', 'Brand', 'Model Status', 'combined_specs']
for name, df_sheet in all_sheets.items():
    df_sheet.columns = [str(c).strip() for c in df_sheet.columns]
    if 'Description' in df_sheet.columns:
        df_sheet['ExtractedWidth'] = df_sheet['Description'].apply(extract_width)
        df_sheet['ExtractedCapacity'] = df_sheet['Description'].apply(extract_capacity)
        df_sheet['ExtractedWattage'] = df_sheet['Description'].apply(extract_wattage)
        # Fill missing or blank Width with ExtractedWidth
        if 'Width' in df_sheet.columns:
            df_sheet['Width'] = df_sheet['Width'].replace('', np.nan)
            df_sheet['Width'] = df_sheet['Width'].fillna(df_sheet['ExtractedWidth'])
        else:
            df_sheet['Width'] = df_sheet['ExtractedWidth']
        # Use wattage from description if found, else fallback to Power Level column
        if 'Power Level' in df_sheet.columns:
            df_sheet['Power Level'] = df_sheet.apply(
                lambda row: row['ExtractedWattage'] if not pd.isnull(row['ExtractedWattage']) else row['Power Level'],
                axis=1
            )
        else:
            df_sheet['Power Level'] = df_sheet['ExtractedWattage']

    if all(col in df_sheet.columns for col in required):
        for sku in df_sheet['SKU'].astype(str):
            sheet_lookup[sku] = name
    all_sheets[name] = df_sheet


def get_structured_similarity(target, candidate, features):
    sim = 0
    weight_total = 0
    for col in features:
        t_val = target.get(col, np.nan)
        c_val = candidate.get(col, np.nan)
        if pd.isnull(t_val) or pd.isnull(c_val):
            continue
        try:
            if col.lower() == "power level":
                t_min, t_max = parse_power_range(t_val)
                c_min, c_max = parse_power_range(c_val)
                if None not in (t_min, t_max, c_min, c_max):
                    # Ranges overlap
                    overlap = not (c_max < t_min or c_min > t_max)
                    if overlap:
                        sim_val = 1
                    else:
                        diff = min(abs(t_min - c_max), abs(t_max - c_min))
                        rng = max(t_max, c_max, 1)
                        sim_val = 1 - min(diff / rng, 1)
                    weight = 2 if col in features else 1
                else:
                    # Fallback to number comparison if possible
                    try:
                        t_float = float(t_val)
                        c_float = float(c_val)
                        diff = abs(t_float - c_float)
                        if diff <= 25:
                            sim_val = 1
                        elif diff <= 100:
                            sim_val = 0.95
                        else:
                            rng = max(abs(t_float), abs(c_float), 1)
                            sim_val = 1 - min(diff / rng, 1)
                        weight = 2 if col in features else 1
                    except Exception:
                        sim_val = int(str(t_val).strip().lower() == str(c_val).strip().lower())
                        weight = 2 if col in features else 1
            elif col.lower() == "width":
                t_float = float(t_val)
                c_float = float(c_val)
                diff = abs(t_float - c_float)
                rng = max(abs(t_float), abs(c_float), 1)
                sim_val = 1 - min(diff / rng, 1)
                weight = 2
            else:
                t_float = float(t_val)
                c_float = float(c_val)
                diff = abs(t_float - c_float)
                rng = max(abs(t_float), abs(c_float), 1)
                sim_val = 1 - min(diff / rng, 1)
                weight = 2 if col in features else 1
        except Exception:
            if col == "Configuration":
                sim_val = 1 if str(t_val).lower() == str(c_val).lower() else 0
                weight = 2 if col in features else 1
            else:
                sim_val = int(str(t_val).strip().lower() == str(c_val).strip().lower())
                weight = 2 if col in features else 1
        sim += sim_val * weight
        weight_total += weight
    return sim / weight_total if weight_total > 0 else 0


results = []
vectorizers = {}
ge_tfidfs = {}
ge_dfs = {}

for sku in skus:
    sheet_name = sheet_lookup.get(sku)
    if not sheet_name:
        results.append({
            'Entered SKU': sku,
            'Closest GE SKU': 'Not found (SKU not in catalog)',
            'Matched GE Model Status': '',
            'Similarity Score': 0
        })
        continue
    sheet_df = all_sheets[sheet_name]
    sheet_df.columns = [str(c).strip() for c in sheet_df.columns]
    all_features = [col for col in sheet_df.columns if col not in discard_cols]
    if 'combined_specs' not in sheet_df.columns:
        sheet_df['combined_specs'] = sheet_df[all_features].astype(str).agg(' '.join, axis=1)
    # Only build vectorizer per sheet
    if sheet_name not in vectorizers:
        vec = TfidfVectorizer()
        tfidf_mat = vec.fit_transform(sheet_df['combined_specs'])
        vectorizers[sheet_name] = vec
        # Only "GE" brand AND "active model"
        ge_mask = (sheet_df['Brand'].str.lower() == 'ge') & (sheet_df['Model Status'].str.lower() == 'active model')
        ge_df = sheet_df[ge_mask].reset_index(drop=True)
        nge_tfidf = vec.transform(ge_df['combined_specs'])
        ge_dfs[sheet_name] = ge_df
        ge_tfidfs[sheet_name] = nge_tfidf
    else:
        vec = vectorizers[sheet_name]
        ge_df = ge_dfs[sheet_name]
        nge_tfidf = ge_tfidfs[sheet_name]
    comp_row = sheet_df[sheet_df['SKU'] == sku]
    if comp_row.empty:
        results.append({
            'Entered SKU': sku,
            'Closest GE SKU': 'Not found (SKU not in sheet)',
            'Matched GE Model Status': '',
            'Similarity Score': 0
        })
        continue
    comp_config = comp_row.iloc[0]['Configuration']
    config_mask = ge_df['Configuration'].str.lower() == str(comp_config).lower()
    filtered_ge = ge_df[config_mask]
    filtered_ge_tfidf = nge_tfidf[config_mask.values]
    if filtered_ge.empty:
        results.append({
            'Entered SKU': sku,
            'Closest GE SKU': 'Not found (no GE match for config)',
            'Matched GE Model Status': '',
            'Similarity Score': 0
        })
        continue
    # --- Structured similarity scoring ---
# Use key features per category for better matching
    category_key_features = {
        'Microwaves': [
            'Capacity (cu ft)', 'Width', 'Configuration', 'Power Level'
        ],
        'Dishwashers': [
            'Width', 'Configuration', 'Sound Level'
        ],
        'Refrigerators': [
            'Capacity (cu ft)', 'Width', 'Configuration'
        ],
        'Laundry Centers and Combos': [
            'Capacity (cu ft)', 'Width', 'Configuration'
        ],
        'Ranges': [
            'Width', 'Cooking Type', 'Number of Burners'
        ],
        'Wall Ovens': [
            'Width', 'Configuration', 'Oven Type'
        ],
        'Cooktops': [
            'Width', 'Configuration', 'Number of Burners'
        ],
        'Freezers': [
            'Capacity (cu ft)', 'Width', 'Configuration'
        ],
        'Wine & Beverage Coolers': [
            'Width', 'Bottle Capacity'
        ],
        'Trash Compactors': [
            'Width', 'Compaction Ratio'
        ],
        'Garbage Disposers': [
            'Horsepower', 'Configuration'
        ],
        'Air Conditioners': [
            'Capacity (BTU)', 'Width', 'Type'
        ],
        'Dehumidifiers': [
            'Capacity (pints/day)', 'Width'
        ],
        'Dryers': [
            'Capacity (cu ft)', 'Width', 'Configuration'
        ],
        'Washers': [
            'Capacity (cu ft)', 'Width', 'Configuration'
        ],
        'Freestanding Icemakers': [
            'Capacity (lbs/day)', 'Width', 'Configuration'
        ],
        # Add more categories and features as needed
    }
    product_category = sheet_name  # If your sheet names match categories
    features = category_key_features.get(product_category, [col for col in sheet_df.columns if col not in discard_cols])

    # Penalize matches missing key structured features
    missing_key_penalty = 0.2

    target = comp_row.iloc[0]
    best_score = 0
    best_idx = -1
    best_sku = None
    best_structured_sim = 0
    best_text_sim = 0
    for row_idx, (idx, candidate) in enumerate(filtered_ge.iterrows()):
        structured_sim = get_structured_similarity(target, candidate, features)
        comp_tfidf = vec.transform([target['combined_specs']])
        tfidf_sim = cosine_similarity(comp_tfidf, filtered_ge_tfidf[row_idx]).item()
        # Penalize matches with missing values in any key feature
        key_missing = any(pd.isnull(candidate.get(k, None)) or pd.isnull(target.get(k, None)) for k in features)
        penalty = missing_key_penalty if key_missing else 0
        combined_score = (0.92 * structured_sim + 0.08 * tfidf_sim) - penalty
            })

        if combined_score > best_score:
            best_score = combined_score
            best_idx = idx
            best_sku = candidate['SKU']
            best_structured_sim = structured_sim
            best_text_sim = tfidf_sim
    if best_idx == -1 or not best_sku:
        results.append({
            'Entered SKU': sku,
            'Closest GE SKU': 'Not found (no similar GE model)',
            'Matched GE Model Status': '',
            'Similarity Score': 0
        })
    else:
        best_status = filtered_ge.loc[best_idx, 'Model Status']
        results.append({
            'Entered SKU': sku,
            'Closest GE SKU': best_sku,
            'Matched GE Model Status': best_status,
            'Similarity Score': round(best_score, 3),
            #'Structured Score':            # Uncomment for extra debug
            # 'Structured Score': round(best_structured_sim, 3),
            # 'TFIDF Score': round(best_text_sim, 3)
        })

results_df = pd.DataFrame(results)
# Exclude 'Matched GE Model Status' from output and download, but keep for logic/filtering
display_cols = [col for col in results_df.columns if col != 'Matched GE Model Status']

st.subheader("Matching Results")
st.dataframe(results_df[display_cols])

if not results_df.empty:
    custom_filename = st.text_input(
        "Enter a name for the Excel file to download (no spaces or .xlsx needed):",
        value="matching_results"
    )
    if not custom_filename.strip():
        custom_filename = "matching_results"
    download_filename = custom_filename.strip().replace(" ", "_") + ".xlsx"
    results_excel = to_excel(results_df[display_cols])
    st.download_button(
        "Download Matching Results to Excel",
        data=results_excel,
        file_name=download_filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
