import streamlit as st
import ee
import geemap.foliumap as geemap
import folium
import requests
import pandas as pd
from fpdf import FPDF                       # fpdf2  -  same import, different output API
import datetime
import os
import urllib.request
import time
import math
import json
from collections import Counter
from google.oauth2 import service_account

# ============================================================
# 1. PAGE CONFIG & STYLING
# ============================================================
st.set_page_config(
    page_title="PMRDA Construction Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
    html, body, [class*="css"] { font-family: 'Share Tech Mono', monospace !important; }
    .main { background-color: #050505; }
    [data-testid="stMetric"] {
        background-color: #0a0a0a;
        border: 1px solid #1f77b4;
        border-left: 4px solid #00ffcc;
        padding: 15px;
        box-shadow: 0 0 15px rgba(0, 255, 204, 0.1);
        margin-bottom: 20px;
    }
    [data-testid="stMetricLabel"] {
        color: #888888 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    [data-testid="stMetricValue"] { color: #00ffcc !important; font-size: 1.8rem !important; }
    .terminal-console {
        background-color: #000000;
        border: 1px solid #333333;
        padding: 15px;
        color: #00ff00;
        font-family: 'Share Tech Mono', monospace;
        height: 240px;
        overflow-y: auto;
        margin-bottom: 20px;
    }
    .v3-badge {
        background-color: #001a00;
        border: 1px solid #00ff00;
        color: #00ff00 !important;
        padding: 3px 10px;
        font-size: 0.75rem;
        display: inline-block;
        margin-left: 10px;
        vertical-align: middle;
    }
    h1 { color: #ffffff; text-transform: uppercase; letter-spacing: 2px; }
    h2, h3 { color: #1f77b4; text-transform: uppercase; }
    hr  { border-color: #333333; }
    </style>
""", unsafe_allow_html=True)

st.title("PMRDA Geospatial Intelligence Portal")
st.markdown(
    "<span style='color:#888;'>SUBSYSTEM: OPTICAL-PRIMARY MULTI-TEMPORAL FUSION ENGINE</span>"
    "<span class='v3-badge'>v3 OPTICAL-PRIMARY</span>",
    unsafe_allow_html=True
)
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Primary Sensor",   "Sentinel-2 MSI (10m)")
col2.metric("Confirmation",     "SAR C-Band (VV+VH)")
col3.metric("Detection Logic",  "Optical + SAR/Context")
col4.metric("Cloud Threshold",  "<= 10%")
st.markdown("---")

# ============================================================
# 2. EARTH ENGINE AUTHENTICATION
# ============================================================
try:
    if "gcp_service_account" in st.secrets:
        key_dict = dict(st.secrets["gcp_service_account"])
        creds = service_account.Credentials.from_service_account_info(
            key_dict,
            scopes=['https://www.googleapis.com/auth/earthengine']
        )
        ee.Initialize(credentials=creds, project='localqol')
        ee.data._credentials = creds
    else:
        ee.Initialize(project='localqol')
except Exception as e:
    st.error(f"Earth Engine authentication failed: {e}")
    st.stop()

# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def get_s1_collection(roi, start_date, end_date, orbit_pass, orbit_number=None):
    """
    Filtered Sentinel-1 IW GRD collection locked to a single orbit pass
    direction to eliminate geometric artifacts from mixed viewing angles.
    """
    col = (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.eq('orbitProperties_pass', orbit_pass))
    )
    if orbit_number is not None:
        col = col.filter(ee.Filter.eq('relativeOrbitNumber_start', int(orbit_number)))
    return col


def make_terrain_corrector(ref_angle_deg=40.0):
    """
    Cosine terrain correction: normalizes SAR backscatter to a reference
    incidence angle, removing slope-induced artifacts.
    Formula: sigma_corrected = sigma_linear * cos(ref) / cos(local_incidence)
    """
    cos_ref = math.cos(ref_angle_deg * math.pi / 180)

    def apply(image):
        angle_rad = image.select('angle').multiply(math.pi / 180)
        cos_inc   = angle_rad.cos()
        vv_tc = (
            ee.Image(10).pow(image.select('VV').divide(10))
            .multiply(cos_ref).divide(cos_inc)
            .log10().multiply(10).rename('VV_tc')
        )
        vh_tc = (
            ee.Image(10).pow(image.select('VH').divide(10))
            .multiply(cos_ref).divide(cos_inc)
            .log10().multiply(10).rename('VH_tc')
        )
        return image.addBands(vv_tc).addBands(vh_tc)
    return apply


def mask_s2_scl(image):
    """
    Per-pixel cloud/shadow mask using Sentinel-2 SCL band.
    Masks: cloud shadow (3), medium cloud (8), high cloud (9),
           cirrus (10), snow/ice (11).
    Superior to scene-level CLOUDY_PIXEL_PERCENTAGE alone because
    it removes residual cloud edges that scene filtering misses.
    """
    scl  = image.select('SCL')
    mask = (
        scl.neq(3).And(scl.neq(8)).And(scl.neq(9))
           .And(scl.neq(10)).And(scl.neq(11))
    )
    return image.updateMask(mask)


def get_s2_composite(roi, start, end, cloud_pct=10):
    """
    Clean Sentinel-2 L2A composite.
    Uses dual cloud filtering: scene-level threshold + pixel-level SCL mask.
    10% threshold is correct for Pune Nov–March dry season window.
    Old code used 40% which allowed cloud-contaminated scenes into composites.
    """
    return (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(roi)
        .filterDate(start, end)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_pct))
        .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'SCL'])
        .map(mask_s2_scl)
        .median()
    )


def compute_indices(img):
    """Compute spectral indices from a Sentinel-2 image."""
    return {
        'ndvi':  img.normalizedDifference(['B8',  'B4']).rename('NDVI'),
        'ndbi':  img.normalizedDifference(['B11', 'B8']).rename('NDBI'),
        'mndwi': img.normalizedDifference(['B3',  'B11']).rename('MNDWI'),
    }



def safe_pdf_text(text):
    """
    Replace non-Latin-1 Unicode characters with ASCII equivalents.
    fpdf2 core fonts (Courier, Helvetica, Times) only cover Latin-1 (ISO-8859-1).
    Characters like em-dash (U+2014), <= (U+2264), arrows (U+2192) will raise
    FPDFUnicodeEncodingException without this sanitisation.
    """
    return (
        str(text)
        .replace('\u2014', ' - ')    # em dash  - 
        .replace('\u2013', ' - ')    # en dash -
        .replace('\u2192', ' -> ')   # right arrow ->
        .replace('\u2190', ' <- ')   # left arrow <-
        .replace('\u2264', '<=')     # less-than-or-equal <=
        .replace('\u2265', '>=')     # greater-than-or-equal >=
        .replace('\u2018', "'")      # left single quote
        .replace('\u2019', "'")      # right single quote
        .replace('\u201c', '"')      # left double quote
        .replace('\u201d', '"')      # right double quote
        .replace('\u00b0', 'deg')    # degree sign (is in Latin-1 but safer as text)
    )

# ============================================================
# 4. LOCATION DICTIONARY
# ============================================================
pmrda_villages = {
    "Hinjewadi (Phase 1 & 2) [HNJ-1]": [18.5913, 73.7389],
    "Maan (Phase 3) [MAN-3]":           [18.5770, 73.6850],
    "Marunji [MRN-0]":                  [18.6010, 73.7220],
    "Mahalunge [MHL-0]":                [18.5675, 73.7460],
    "Sus Sector [SUS-0]":               [18.5435, 73.7435],
    "Wakad Node [WKD-0]":               [18.5987, 73.7688],
    "Ravet [RVT-0]":                    [18.6480, 73.7520],
    "Moshi [MSH-0]":                    [18.6780, 73.8530],
    "Chakan [CHK-0]":                   [18.7630, 73.8620],
    "Manual Override (Custom Coordinates)": [None, None],
}

# ============================================================
# 5. SIDEBAR  -  PIPELINE PARAMETERS
# ============================================================
with st.sidebar:
    st.header("PIPELINE PARAMETERS")

    # 5a. Target
    st.subheader("1. Target Acquisition")
    selected_location = st.selectbox(
        "Select Sector:", options=list(pmrda_villages.keys())
    )
    if selected_location == "Manual Override (Custom Coordinates)":
        lat = st.number_input("Latitude  (WGS84)", value=18.585000, format="%.6f")
        lon = st.number_input("Longitude (WGS84)", value=73.715000, format="%.6f")
    else:
        lat = pmrda_villages[selected_location][0]
        lon = pmrda_villages[selected_location][1]
        st.info(f"TARGET LOCKED: {lat:.4f}, {lon:.4f}")

    # 5b. Temporal
    st.subheader("2. Temporal Configuration")
    monthly_mode = st.toggle(
        "Monthly Monitoring Mode",
        value=False,
        help=(
            "Year-over-year same-month comparison cancels phenological and soil-moisture noise. "
            "Best for agricultural fringe areas like PMRDA boundary."
        )
    )

    if monthly_mode:
        today = datetime.date.today()
        monitor_month = st.selectbox(
            "Reference Month",
            options=list(range(1, 13)),
            index=today.month - 1,
            format_func=lambda m: datetime.date(2000, m, 1).strftime('%B')
        )
        monitor_year = st.number_input(
            "Reference Year (T1)", value=today.year,
            min_value=2017, max_value=today.year, step=1
        )
        t1_start = datetime.date(monitor_year, monitor_month, 1)
        t1_end   = (
            datetime.date(monitor_year, 12, 31) if monitor_month == 12
            else datetime.date(monitor_year, monitor_month + 1, 1) - datetime.timedelta(days=1)
        )
        t0_year  = monitor_year - 1
        t0_start = datetime.date(t0_year, monitor_month, 1)
        t0_end   = t0_start + datetime.timedelta(days=44)
        before_dates = [t0_start, t0_end]
        after_dates  = [t1_start, t1_end]
        st.info(f"T0: {t0_start} → {t0_end}\nT1: {t1_start} → {t1_end}")
    else:
        before_dates = st.date_input(
            "T0 Baseline (Pre-event)",
            [datetime.date(2025, 10, 1), datetime.date(2026, 1, 31)]
        )
        after_dates = st.date_input(
            "T1 Detection Window",
            [datetime.date(2026, 2, 1), datetime.date(2026, 3, 30)]
        )

    # 5c. Optical Thresholds  -  PRIMARY detector
    st.subheader("3. Optical Detection (Primary)")
    ndvi_loss_thresh = st.slider(
        "Min NDVI Loss",
        min_value=0.10, max_value=0.50,
        value=0.15 if monthly_mode else 0.20,
        step=0.01,
        help=(
            "NDVI must drop by this amount from T0 to T1. "
            "0.20 = significant permanent vegetation removal. "
            "Crop harvest in same-season comparison has NDVI loss ~0.05–0.10 (below threshold). "
            "Construction causes permanent 0.20–0.60 NDVI loss."
        )
    )
    ndbi_gain_thresh = st.slider(
        "Min NDBI Gain",
        min_value=0.05, max_value=0.40,
        value=0.10 if monthly_mode else 0.12,
        step=0.01,
        help=(
            "NDBI must increase by this amount. "
            "Wet soil raises NDBI by 0.03–0.07  -  safely below 0.12 threshold. "
            "New impervious surface raises NDBI by 0.15–0.40. "
            "Both conditions (NDVI loss AND NDBI gain) must be true simultaneously."
        )
    )
    cloud_thresh = st.slider(
        "Max Cloud Cover per Scene (%)",
        min_value=5, max_value=30, value=10, step=5,
        help=(
            "10% is correct for Pune Nov–March dry season. "
            "The previous 40% allowed cloud-contaminated scenes into composites. "
            "Relaxing this risks cloud-edge artifacts being detected as 'change'."
        )
    )

    # 5d. SAR Thresholds  -  CONFIRMATION only
    st.subheader("4. SAR Confirmation")
    st.caption("SAR confirms optical detections. It does NOT trigger them.")
    radar_thresh = st.slider(
        "Vertical Structure SAR Threshold (dB)",
        min_value=2.0, max_value=8.0, value=4.0, step=0.5,
        help=(
            "VV increase required for VERTICAL STRUCTURE class. "
            "Lowered from 5.5 to 4.0  -  optical gating already filters noise, "
            "so SAR confirmation threshold can be relaxed."
        )
    )
    sar_fallback = st.toggle(
        "SAR-Optional Mode",
        value=True,
        help=(
            "If SAR data is unavailable, fall back to optical-only. "
            "Recommended ON. Disabling requires SAR for every detection."
        )
    )

    # 5e. Orbit
    st.subheader("5. SAR Orbit Configuration")
    orbit_pass = st.selectbox(
        "Orbit Pass Direction",
        ["DESCENDING", "ASCENDING"],
        help="DESCENDING is the primary available pass over the PMRDA/PCMC region."
    )

    # 5f. Spatial / Quality
    st.subheader("6. Spatial Filters")
    buffer_radius = st.number_input(
        "Analysis Radius (m)", value=800, step=100, min_value=200, max_value=5000,
        help=(
            "800m = ~2 km² analysis zone = ~2,000 Sentinel-2 pixels. "
            "Previous default (3000m) = 28 km²  -  guarantees statistical false positives."
        )
    )
    min_area_sqm = st.slider(
        "Minimum Alert Area (sqm)",
        100, 2000,
        300 if monthly_mode else 200,
        step=50,
        help="Minimum contiguous pixel area to report. Eliminates isolated noise."
    )
    slope_thresh = st.slider(
        "Max Detection Slope (°)", 5, 30, 12, step=1,
        help="Exclude steep-slope pixels where SAR backscatter is unreliable."
    )

    with st.expander("ADVANCED"):
        if monthly_mode:
            persistence_required = 1
            st.info("Persistence locked to 1 (monthly window ~2–3 passes).")
        else:
            persistence_required = st.slider(
                "SAR Temporal Persistence (passes)", 1, 5, 2,
                help=(
                    "Number of independent S1 passes that must confirm the change. "
                    "Eliminates floods, fires, equipment, and crop burn events."
                )
            )
        ref_angle_deg = st.slider(
            "Terrain Correction Reference Angle (°)", 20, 50, 40,
            help="Cosine correction reference angle. 40° is standard for C-band."
        )
        fetch_limit = st.slider("Max Detection Polygons", 5, 50, 20, step=5)

    st.subheader("7. External APIs")
    try:
        gmaps_api_key = st.secrets["GMAPS_API_KEY"]
        st.success("Google Maps API: CONNECTED")
    except KeyError:
        gmaps_api_key = st.text_input("Google Maps API Key (optional)", type="password")

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button(
        "RUN DETECTION PIPELINE", type="primary", use_container_width=True
    )

# ============================================================
# 6. MAIN PIPELINE
# ============================================================
if run_btn:
    if len(before_dates) != 2 or len(after_dates) != 2:
        st.error("Date inputs require both start and end dates.")
        st.stop()

    b_start = before_dates[0].strftime('%Y-%m-%d')
    b_end   = before_dates[1].strftime('%Y-%m-%d')
    a_start = after_dates[0].strftime('%Y-%m-%d')
    a_end   = after_dates[1].strftime('%Y-%m-%d')
    mode_label = "MONTHLY YEAR-OVER-YEAR" if monthly_mode else "CUSTOM DATE RANGE"

    telemetry_placeholder = st.empty()
    logs = []

    def log(msg):
        logs.append(f"> {msg}")
        telemetry_placeholder.markdown(
            f"<div class='terminal-console'>{'<br>'.join(logs[-18:])}</div>",
            unsafe_allow_html=True
        )

    log(f"MODE: {mode_label}")
    log(f"TARGET: {lat:.5f}N, {lon:.5f}E | AOI radius: {buffer_radius}m")
    log(f"T0: {b_start} → {b_end}")
    log(f"T1: {a_start} → {a_end}")
    log(f"Optical triggers: NDVI loss > {ndvi_loss_thresh}, NDBI gain > {ndbi_gain_thresh}")
    log("Connecting to Earth Engine...")

    with st.spinner("Running detection pipeline..."):

        roi = ee.Geometry.Point([lon, lat]).buffer(buffer_radius)
        terrain_corrector = make_terrain_corrector(ref_angle_deg)

        # -------- 6A. SENTINEL-2 OPTICAL COMPOSITES (PRIMARY) --------
        log(f"Fetching S2 T0 composite (CLOUDY_PIXEL_PERCENTAGE <= {cloud_thresh}%)...")
        s2_before = get_s2_composite(roi, b_start, b_end, cloud_thresh)

        log(f"Fetching S2 T1 composite (CLOUDY_PIXEL_PERCENTAGE <= {cloud_thresh}%)...")
        s2_after  = get_s2_composite(roi, a_start, a_end, cloud_thresh)

        s2_before_count = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(roi).filterDate(b_start, b_end)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_thresh))
            .size().getInfo()
        )
        s2_after_count = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(roi).filterDate(a_start, a_end)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_thresh))
            .size().getInfo()
        )
        log(f"S2 scenes found: T0={s2_before_count}, T1={s2_after_count}")

        if s2_before_count == 0 or s2_after_count == 0:
            st.error(
                f"No Sentinel-2 scenes with <={cloud_thresh}% cloud cover found. "
                "Try: widen date range, raise cloud threshold, or use Monthly Mode."
            )
            st.stop()

        # -------- 6B. SPECTRAL INDICES --------
        log("Computing NDVI, NDBI, MNDWI indices...")
        idx_before = compute_indices(s2_before)
        idx_after  = compute_indices(s2_after)

        # Positive = vegetation LOST between T0 and T1
        ndvi_loss  = idx_before['ndvi'].subtract(idx_after['ndvi'])
        # Positive = built-up index INCREASED between T0 and T1
        ndbi_gain  = idx_after['ndbi'].subtract(idx_before['ndbi'])
        mndwi_after = idx_after['mndwi']

        # -------- 6C. STATIC MASKS --------
        log(f"Computing slope mask from NASADEM (threshold: {slope_thresh}°)...")
        dem       = ee.Image('NASA/NASADEM_HGT/001').select('elevation')
        slope_deg = ee.Terrain.slope(dem)
        flat_mask = slope_deg.lt(slope_thresh)

        # Water bodies: MNDWI > 0.2  -  exclude seasonal water/flood plains
        not_water = mndwi_after.lt(0.2)

        # -------- 6D. OPTICAL PRIMARY TRIGGER --------
        # BOTH conditions must be true simultaneously:
        #   (1) NDVI dropped by ndvi_loss_thresh → vegetation permanently removed
        #   (2) NDBI rose by ndbi_gain_thresh → impervious surface appeared
        # This dual requirement eliminates:
        #   - Crop harvest: NDVI drops but NDBI does NOT significantly rise
        #   - Soil moisture change: NDBI shifts only 0.03–0.07, below 0.12 threshold
        #   - Seasonal bare soil: NDVI drops but NDBI stays low or negative
        log(
            f"Applying optical primary trigger "
            f"(NDVI loss > {ndvi_loss_thresh} AND NDBI gain > {ndbi_gain_thresh})..."
        )
        optical_trigger = (
            ndvi_loss.gt(ndvi_loss_thresh)
            .And(ndbi_gain.gt(ndbi_gain_thresh))
            .And(flat_mask)
            .And(not_water)
        )

        # -------- 6E. SAR ORBIT LOCKING (v4 — auto-direction fallback) --------
        # Root cause of "No SAR data" when switching orbit direction:
        #   Old code discovered orbits from T0 ONLY. If T0 dominant orbit has
        #   0 passes in T1 for the chosen direction, sar_available = False with
        #   no explanation. Also gave no auto-recovery.
        #
        # Fix:
        #   1. Discover orbit numbers from BOTH T0 and T1
        #   2. Only lock to orbits that appear in BOTH (intersection)
        #   3. If chosen direction has 0 common orbits, auto-try opposite direction
        #   4. Emit orbit diagnostic table to log on every run
        log(f"Fetching Sentinel-1 ({orbit_pass}) — discovering orbits in T0 and T1...")
        before_s1_raw  = get_s1_collection(roi, b_start, b_end, orbit_pass)
        after_s1_raw   = get_s1_collection(roi, a_start, a_end, orbit_pass)
        before_orbits  = before_s1_raw.aggregate_array('relativeOrbitNumber_start').getInfo()
        after_orbits   = after_s1_raw.aggregate_array('relativeOrbitNumber_start').getInfo()
        common_orbits  = sorted(set(before_orbits) & set(after_orbits))
        active_pass    = orbit_pass   # may be updated by auto-fallback below
        sar_available  = len(common_orbits) > 0
        dominant_orbit = None
        after_count    = 0

        # Emit orbit diagnostic before any decision
        def _orbit_table(b_list, a_list):
            all_orbs = sorted(set(b_list) | set(a_list))
            bc = Counter(b_list); ac = Counter(a_list)
            rows = [f"  orbit #{o}: T0={bc.get(o,0)} T1={ac.get(o,0)}" for o in all_orbs]
            return ("\n").join(rows) if rows else "  (none found)"

        log(f"SAR orbit scan [{orbit_pass}]:\n{_orbit_table(before_orbits, after_orbits)}")

        # Auto-fallback: if no common coverage in chosen direction, try opposite
        if not sar_available and sar_fallback:
            alt_pass = "ASCENDING" if orbit_pass == "DESCENDING" else "DESCENDING"
            log(f"No common {orbit_pass} orbits. Auto-trying {alt_pass}...")
            st.warning(
                f"No Sentinel-1 **{orbit_pass}** passes with coverage in both T0 and T1. "
                f"Automatically switching to **{alt_pass}** — re-run with {alt_pass} "
                f"selected to make this permanent.",
                icon="🔄"
            )
            alt_before = get_s1_collection(roi, b_start, b_end, alt_pass)
            alt_after  = get_s1_collection(roi, a_start, a_end, alt_pass)
            alt_b_orbs = alt_before.aggregate_array('relativeOrbitNumber_start').getInfo()
            alt_a_orbs = alt_after.aggregate_array('relativeOrbitNumber_start').getInfo()
            alt_common = sorted(set(alt_b_orbs) & set(alt_a_orbs))
            log(f"SAR orbit scan [{alt_pass}]:\n{_orbit_table(alt_b_orbs, alt_a_orbs)}")
            if alt_common:
                before_s1_raw = alt_before
                after_s1_raw  = alt_after
                before_orbits = alt_b_orbs
                after_orbits  = alt_a_orbs
                common_orbits = alt_common
                active_pass   = alt_pass
                sar_available = True
                log(f"AUTO-FALLBACK SUCCESS: using {alt_pass} orbit(s) {alt_common}")

        if sar_available:
            # Pick dominant = orbit with most T1 passes among common set
            t1_counter    = Counter(o for o in after_orbits if o in common_orbits)
            dominant_orbit = t1_counter.most_common(1)[0][0]
            before_s1_col  = before_s1_raw.filter(
                ee.Filter.eq('relativeOrbitNumber_start', dominant_orbit)
            )
            after_s1_col   = after_s1_raw.filter(
                ee.Filter.eq('relativeOrbitNumber_start', dominant_orbit)
            )
            after_count    = after_s1_col.size().getInfo()
            sar_available  = after_count > 0
            log(
                f"SAR locked: {active_pass} orbit #{dominant_orbit} | "
                f"T0={Counter(before_orbits)[dominant_orbit]} passes | "
                f"T1={after_count} passes"
            )

        if not sar_available:
            if sar_fallback:
                log("WARNING: No SAR data in any orbit direction. Optical-only mode.")
                st.warning(
                    "No Sentinel-1 data found for this AOI/date range in either orbit "
                    "direction. Running optical-only detection (thresholds tightened)."
                )
            else:
                b_info = f"T0: {len(before_orbits)} {orbit_pass} passes"
                a_info = f"T1: {len(after_orbits)} {orbit_pass} passes"
                st.error(
                    f"No SAR data ({b_info} | {a_info}). "
                    "Enable SAR-Optional Mode or widen the date range."
                )
                st.stop()

        # -------- 6F. SAR TERRAIN-CORRECTED COMPOSITES --------
        if sar_available:
            log(f"Applying cosine terrain correction (ref angle: {ref_angle_deg}°)...")
            s1_before = (
                before_s1_col.map(terrain_corrector)
                .select(['VV_tc', 'VH_tc']).median()
            )
            s1_after = (
                after_s1_col.map(terrain_corrector)
                .select(['VV_tc', 'VH_tc']).median()
            )

            # -------- 6G. SAR TEMPORAL PERSISTENCE --------
            log(
                f"Building persistence stack "
                f"(require {persistence_required} of {after_count} passes)..."
            )
            # Each T1 pass is independently compared against T0 median.
            # A pixel must flag in >= persistence_required passes.
            # This eliminates: floods, fires, crop burns, parked vehicles.
            vv_persist_thresh = radar_thresh * 0.6
            before_vv         = s1_before.select('VV_tc')

            def flag_vv_change(image):
                return (
                    image.select('VV_tc')
                    .subtract(before_vv)
                    .gte(vv_persist_thresh)
                    .rename('changed')
                    .toFloat()
                )

            after_corrected   = after_s1_col.map(terrain_corrector)
            change_flags      = after_corrected.map(flag_vv_change)
            persistence_count = change_flags.sum()
            persistence_mask  = persistence_count.gte(persistence_required)

            # -------- 6H. SAR CHANGE METRICS --------
            vv_change    = s1_after.select('VV_tc').subtract(s1_before.select('VV_tc'))
            ratio_before = s1_before.select('VV_tc').subtract(s1_before.select('VH_tc'))
            ratio_after  = s1_after.select('VV_tc').subtract(s1_after.select('VH_tc'))
            ratio_change = ratio_after.subtract(ratio_before)

            # Road filter: paved roads show NDBI increase but low VV change
            # and no double-bounce ratio rise (flat surface, no vertical walls)
            is_road = (
                ndbi_gain.gt(ndbi_gain_thresh)
                .And(vv_change.abs().lt(2.0))
                .And(ratio_change.lt(0.5))
            )

            # Adaptive threshold: pixels that were crops (NDVI > 0.5) in T0
            # get a higher SAR threshold because crop stems create backscatter
            # that partially mimics construction signature.
            was_crop = idx_before['ndvi'].gt(0.50)
            smart_radar_thresh = ee.Image(radar_thresh).where(
                was_crop, ee.Image(radar_thresh).add(2.0)
            )

            # -------- 6I. DETECTION CLASSES --------
            log("Classifying detections: Land Clearing / Vertical Structure / Optical-Only...")

            # CLASS 1  -  LAND CLEARING
            # Optical trigger + moderate SAR increase (soil disturbance present)
            # No strong double-bounce ratio rise yet = no walls, foundation/slab stage
            is_clearing = (
                optical_trigger
                .And(vv_change.gte(2.0))
                .And(vv_change.lt(smart_radar_thresh))
                .And(ratio_change.lt(1.0))
                .And(is_road.Not())
                .And(persistence_mask)
            )

            # CLASS 2  -  VERTICAL STRUCTURE
            # Optical trigger + strong SAR increase + double-bounce ratio rise
            # Double-bounce (ratio > 1.0 dB rise) = vertical walls present
            is_vertical = (
                optical_trigger
                .And(vv_change.gte(smart_radar_thresh))
                .And(ratio_change.gt(1.0))
                .And(is_road.Not())
                .And(persistence_mask)
            )

            # CLASS 3  -  OPTICAL ONLY (SAR-unconfirmed construction)
            # When SAR is available but doesn't confirm, we still allow a
            # tightly-gated Class 3 for early-stage urban construction that
            # SAR C-band structurally misses:
            #   • Flat RCC slab (pre-wall stage)  — no double-bounce yet
            #   • Metal / polycarbonate sheet roof — specular, reduces VV
            #   • Single-storey low structures     — below double-bounce threshold
            #
            # Gate 1: tighter optical thresholds (+0.06 on both)
            # Gate 2: NDBI_BASELINE > 0.05 — must be inside existing built-up
            #         area, not a crop field going dry (kills Rabi FP noise)
            # Gate 3: NOT already in SAR-confirmed classes
            # Gate 4: NOT a road
            ndbi_before = compute_indices(s2_before)['ndbi']
            is_builtup_context = ndbi_before.gt(0.05)

            is_optical_only = (
                optical_trigger
                .And(is_clearing.Not())
                .And(is_vertical.Not())
                .And(is_road.Not())
                .And(is_builtup_context)
                .And(ndvi_loss.gt(ndvi_loss_thresh + 0.06))
                .And(ndbi_gain.gt(ndbi_gain_thresh + 0.06))
            )
            log("Class 3 (Optical-Only) active with built-up context guard + tightened thresholds.")

        else:
            # SAR unavailable  -  optical-only fallback
            # Tighten optical thresholds by 0.05 each to compensate
            log("Optical-only mode: tightening NDVI/NDBI thresholds by 0.05...")
            is_clearing    = optical_trigger.And(ndvi_loss.gt(ndvi_loss_thresh + 0.05))
            is_vertical    = optical_trigger.And(ndbi_gain.gt(ndbi_gain_thresh + 0.08))
            is_optical_only = ee.Image(0).selfMask()
            is_road         = ee.Image(0).selfMask()

        # -------- 6J. MINIMUM MAPPING UNIT (MMU) FILTER --------
        # connectedPixelCount: a pixel must have >= min_pixels spatially
        # contiguous neighbours also flagged. Eliminates isolated noise pixels.
        # At 10m resolution: 1 pixel = 100 sqm
        log(f"Applying MMU filter (minimum {min_area_sqm} sqm = {max(1, min_area_sqm//100)} pixels)...")
        min_pixels = max(1, min_area_sqm // 100)

        clearing_mmu = is_clearing.updateMask(
            is_clearing.connectedPixelCount(500, True).gte(min_pixels)
        )
        vertical_mmu = is_vertical.updateMask(
            is_vertical.connectedPixelCount(500, True).gte(min_pixels)
        )
        optical_mmu = is_optical_only.updateMask(
            is_optical_only.connectedPixelCount(500, True).gte(min_pixels)
        )

        # 1 = Land Clearing, 2 = Vertical Structure, 3 = Optical Only
        alert_img = (
            ee.Image(0)
            .where(clearing_mmu, 1)
            .where(vertical_mmu, 2)
            .where(optical_mmu,  3)
            .selfMask()
        )

        log("Alert layer compiled. Rendering...")

        # ============================================================
        # 7. VISUALIZATION  -  THREE TABS
        # ============================================================
        tab1, tab2, tab3 = st.tabs([
            "DETECTION MAP",
            "DETECTION TABLE + REPORT",
            "SPECTRAL DEBUG VIEW"
        ])

        # -------- TAB 1: MAP --------
        with tab1:
            st.markdown("#### Detection Map")
            st.caption(
                "🟠 Land Clearing (optical + moderate SAR)  "
                "🔴 Vertical Structure (optical + SAR double-bounce)  "
                "🟡 Optical Signal Only"
            )

            Map = geemap.Map(center=[lat, lon], zoom=14, ee_initialize=False)

            # Add Google Satellite basemap via folium TileLayer
            # (more reliable than geemap's add_tile_layer across versions)
            folium.TileLayer(
                tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
                attr="Google",
                name="Google Satellite",
                overlay=False,
                control=True,
            ).add_to(Map)

            # Sentinel-2 T1 False Colour composite (SWIR-NIR-Red)
            Map.addLayer(
                s2_after,
                {'bands': ['B11', 'B8', 'B4'], 'min': 0, 'max': 4000, 'gamma': 1.2},
                'S2 T1 False Colour (SWIR-NIR-Red)',
                shown=False
            )

            # Detection layers
            Map.addLayer(
                alert_img.eq(1).selfMask(),
                {'palette': '#ff8c00', 'opacity': 0.75},
                'Land Clearing (Class 1)'
            )
            Map.addLayer(
                alert_img.eq(2).selfMask(),
                {'palette': '#ff2200', 'opacity': 0.75},
                'Vertical Structure (Class 2)'
            )
            Map.addLayer(
                alert_img.eq(3).selfMask(),
                {'palette': '#ffee00', 'opacity': 0.65},
                'Optical Only (Class 3)'
            )

            # Debug layers  -  hidden by default, toggle on as needed
            Map.addLayer(
                ndvi_loss,
                {'min': -0.3, 'max': 0.5, 'palette': ['blue', 'white', 'red']},
                '[debug] NDVI Loss', shown=False
            )
            Map.addLayer(
                ndbi_gain,
                {'min': -0.2, 'max': 0.4, 'palette': ['blue', 'white', 'red']},
                '[debug] NDBI Gain', shown=False
            )
            if sar_available:
                Map.addLayer(
                    vv_change,
                    {'min': -5, 'max': 8, 'palette': ['blue', 'white', 'red']},
                    '[debug] VV Change (dB)', shown=False
                )
                Map.addLayer(
                    ratio_change,
                    {'min': -3, 'max': 3, 'palette': ['blue', 'white', 'red']},
                    '[debug] VV-VH Ratio Change', shown=False
                )
                Map.addLayer(
                    persistence_count,
                    {'min': 0, 'max': after_count, 'palette': ['black', 'yellow', 'white']},
                    f'[debug] SAR Persistence (0–{after_count})', shown=False
                )

            folium.LayerControl().add_to(Map)
            Map.to_streamlit(height=650)

        # -------- TAB 2: TABLE + REPORT --------
        with tab2:
            st.markdown("#### Detected Change Polygons")

            log("Extracting detection polygons (reduceToVectors)...")

            def extract_polygons(mask_img, alert_type_val, label):
                """
                Extract polygons not centroids.
                Centroids give meaningless dots  -  polygons show actual footprint + area.
                """
                return (
                    mask_img.selfMask()
                    .reduceToVectors(
                        geometry=roi,
                        crs='EPSG:4326',
                        scale=10,
                        geometryType='polygon',
                        eightConnected=True,
                        maxPixels=1e8,
                        reducer=ee.Reducer.countEvery()
                    )
                    .limit(fetch_limit)
                    .map(lambda f: f.set({
                        'alert_type':  alert_type_val,
                        'alert_label': label
                    }))
                )

            clearing_vec = extract_polygons(clearing_mmu, 1, 'Land Clearing')
            vertical_vec = extract_polygons(vertical_mmu, 2, 'Vertical Structure')
            optical_vec  = extract_polygons(optical_mmu,  3, 'Optical Only')

            all_vectors = clearing_vec.merge(vertical_vec).merge(optical_vec)

            # FIX 1 — MMU post-vectorisation guard.
            # 'count' is set by ee.Reducer.countEvery() inside extract_polygons.
            # Each pixel = 100 sqm, so min count = min_area_sqm / 100.
            # This catches fragments that slip past connectedPixelCount (known
            # GEE edge case when polygon dissolve produces isolated remnants).
            all_vectors = all_vectors.filter(
                ee.Filter.gte('count', max(1, min_area_sqm // 100))
            )

            try:
                vector_data = all_vectors.getInfo()
            except Exception as e:
                st.error(f"Vector extraction failed: {e}")
                vector_data = {'features': []}

            features   = vector_data.get('features', [])
            n_total    = len(features)
            n_clearing = sum(1 for f in features if f['properties'].get('alert_type') == 1)
            n_vertical = sum(1 for f in features if f['properties'].get('alert_type') == 2)
            n_optical  = sum(1 for f in features if f['properties'].get('alert_type') == 3)

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Total",            n_total)
            mc2.metric("Land Clearing",    n_clearing)
            mc3.metric("Vertical Struct.", n_vertical)
            mc4.metric("Optical Only",     n_optical)

            if n_total == 0:
                st.warning(
                    "No detections found. Suggestions: "
                    "(1) Widen date range. "
                    "(2) Reduce NDVI loss / NDBI gain thresholds. "
                    "(3) Reduce minimum area. "
                    "(4) Increase cloud threshold if insufficient clean scenes exist."
                )
                log("Pipeline complete: 0 detections.")
            else:
                log(f"Pipeline complete: {n_total} detections.")
                st.success(f"{n_total} detections found.")

                # Build DataFrame
                rows = []
                for i, feat in enumerate(features):
                    props  = feat.get('properties', {})
                    geom   = feat.get('geometry',   {})
                    coords = geom.get('coordinates', [[[]]])
                    try:
                        flat = coords[0]
                        cx   = sum(c[0] for c in flat) / len(flat)
                        cy   = sum(c[1] for c in flat) / len(flat)
                    except Exception:
                        cx, cy = lon, lat
                    pixel_count = props.get('count', 0)
                    area_sqm    = pixel_count * 100
                    rows.append({
                        'ID':          f"DET-{i+1:03d}",
                        'Type':        props.get('alert_label', '?'),
                        'Lat':         round(cy, 5),
                        'Lon':         round(cx, 5),
                        'Area (sqm)':  area_sqm,
                        'Pixels':      pixel_count,
                        'Class':       props.get('alert_type', 0),
                    })

                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)

                # GeoJSON download
                st.download_button(
                    label="Download GeoJSON",
                    data=json.dumps(vector_data, indent=2),
                    file_name=f"PMRDA_detections_{a_start}_{a_end}.geojson",
                    mime="application/json"
                )

                # ---- IMAGE CHIPS + PDF ----
                st.markdown("#### Detection Evidence Chips (Before / After  -  Sentinel-2 True Colour)")

                def get_s2_thumb_url(img, cy, cx, size_m=500):
                    """500m buffer = 1km×1km spatial context per chip."""
                    box = ee.Geometry.Point([cx, cy]).buffer(size_m)
                    return img.visualize(
                        bands=['B4', 'B3', 'B2'], min=0, max=3000
                    ).getThumbURL({
                        'region': box, 'dimensions': '320x320', 'format': 'png'
                    })

                # PDF Report class
                # ============================================================
                # PDF REPORT — Professional redesign
                # Layout:
                #   Page 1 : Cover / Summary (params table + detection register)
                #   Pages 2+: One page per detection
                #             - Class-coloured header bar
                #             - Metrics table
                #             - T0 | T1 Sentinel-2 side-by-side (top)
                #             - Google Satellite full-width (bottom)
                #             - Detection basis footnote
                # ============================================================

                # ---- Colour palette (RGB tuples) ----
                C_NAVY   = (15,  37,  84)
                C_TEAL   = (0,  140, 140)
                C_WHITE  = (255, 255, 255)
                C_BLACK  = (0,   0,   0)
                C_LGREY  = (245, 245, 245)
                C_MGREY  = (200, 200, 200)
                C_DGREY  = (100, 100, 100)
                C_ORANGE = (210,  90,   0)   # Land Clearing
                C_RED    = (180,  25,  25)   # Vertical Structure
                C_BLUE   = ( 30, 100, 200)   # Optical Only
                CLASS_COLOR = {1: C_ORANGE, 2: C_RED, 3: C_BLUE}
                CLASS_LABEL = {1: 'LAND CLEARING', 2: 'VERTICAL STRUCTURE', 3: 'OPTICAL ONLY'}

                # ---- Resolved orbit label (survives auto-fallback) ----
                _rpt_orbit = active_pass if 'active_pass' in dir() else orbit_pass
                _rpt_orbit_num = str(dominant_orbit) if dominant_orbit else 'n/a'
                _rpt_sar_mode  = 'Optional (fallback ON)' if sar_fallback else 'Required'
                _rpt_t1_passes = str(after_count) if after_count else 'n/a'

                class PMRDAReport(FPDF):

                    # ------ helpers ------
                    def _fill(self, rgb):
                        self.set_fill_color(*rgb)

                    def _text(self, rgb):
                        self.set_text_color(*rgb)

                    def _draw(self, rgb):
                        self.set_draw_color(*rgb)

                    def section_bar(self, title):
                        """Navy section header bar — full content width."""
                        self._fill(C_NAVY); self._text(C_WHITE)
                        self.set_font('Courier', 'B', 8)
                        self.cell(190, 7, f'  {safe_pdf_text(title)}', 0, 1, 'L', fill=True)
                        self._fill(C_WHITE); self._text(C_BLACK)
                        self.ln(1)

                    def two_col_kv(self, rows, lw=88, rw=102):
                        """Alternating-row key/value table."""
                        self._draw(C_MGREY)
                        self.set_line_width(0.2)
                        for idx, (k, v) in enumerate(rows):
                            bg = C_LGREY if idx % 2 == 0 else C_WHITE
                            self._fill(bg)
                            self.set_font('Courier', '', 8)
                            self._text(C_DGREY)
                            self.cell(lw, 6, f'  {safe_pdf_text(str(k))}', 'LB', 0, 'L', fill=True)
                            self.set_font('Courier', 'B', 8)
                            self._text(C_BLACK)
                            self.cell(rw, 6, f'  {safe_pdf_text(str(v))}', 'RB', 1, 'L', fill=True)
                        self._fill(C_WHITE); self._text(C_BLACK)
                        self._draw(C_BLACK); self.set_line_width(0.2)

                    def table_header_row(self, cols, widths):
                        """Dark-navy table column headers."""
                        self._fill(C_NAVY); self._text(C_WHITE)
                        self.set_font('Courier', 'B', 7)
                        for col, w in zip(cols, widths):
                            self.cell(w, 7, f' {col}', 0, 0, 'L', fill=True)
                        self.ln()
                        self._fill(C_WHITE); self._text(C_BLACK)

                    def img_label_bar(self, label, x, w, y):
                        """Teal label strip above each image."""
                        self._fill(C_TEAL); self._text(C_WHITE)
                        self.set_font('Courier', 'B', 7)
                        self.set_xy(x, y)
                        self.cell(w, 5, f' {label}', 0, 0, 'L', fill=True)
                        self._fill(C_WHITE); self._text(C_BLACK)

                    def placeholder_box(self, x, y, w, h, msg):
                        """Grey bordered placeholder when image unavailable."""
                        self._draw(C_MGREY); self._fill(C_LGREY)
                        self.set_line_width(0.3)
                        self.rect(x, y, w, h, 'FD')
                        self.set_font('Courier', '', 7)
                        self._text(C_DGREY)
                        self.set_xy(x, y + h/2 - 3)
                        self.cell(w, 4, safe_pdf_text(msg), 0, 1, 'C')
                        self._draw(C_BLACK); self._fill(C_WHITE)
                        self._text(C_BLACK); self.set_line_width(0.2)

                    # ------ header / footer ------
                    def header(self):
                        if self.page_no() == 1:
                            return   # Cover page has its own title block
                        self._fill(C_NAVY); self._text(C_WHITE)
                        self.set_font('Courier', 'B', 7)
                        loc_short = safe_pdf_text(selected_location.split('[')[0].strip())
                        self.cell(140, 6,
                            f'  PMRDA | {loc_short} | T1: {a_start} to {a_end}',
                            0, 0, 'L', fill=True)
                        self.set_font('Courier', '', 7)
                        self.cell(50, 6,
                            f'Page {self.page_no()}  ',
                            0, 1, 'R', fill=True)
                        self._fill(C_WHITE); self._text(C_BLACK)
                        self.ln(2)

                    def footer(self):
                        self.set_y(-12)
                        self._draw(C_TEAL); self.set_line_width(0.5)
                        self.line(10, self.get_y(), 200, self.get_y())
                        self.set_line_width(0.2); self._draw(C_BLACK)
                        self.ln(1)
                        self.set_font('Courier', '', 6)
                        self._text(C_DGREY)
                        self.cell(0, 4, safe_pdf_text(
                            f'Generated {datetime.date.today()} | '
                            f'PMRDA Geospatial Intelligence Portal v3 | '
                            f'Sentinel-2 L2A + Sentinel-1 IW GRD via Google Earth Engine | '
                            f'FOR OFFICIAL USE ONLY — Field verification required before enforcement.'
                        ), 0, 1, 'C')
                        self._text(C_BLACK)

                # ---- instantiate ----
                pdf = PMRDAReport()
                pdf.set_auto_page_break(auto=True, margin=14)
                pdf.set_margins(10, 10, 10)

                # ================================================================
                # PAGE 1 — COVER / SUMMARY
                # ================================================================
                pdf.add_page()

                # Title block — full-width navy
                pdf._fill(C_NAVY); pdf._text(C_WHITE)
                pdf.set_font('Courier', 'B', 15)
                pdf.cell(190, 11,
                    'PMRDA GEOSPATIAL INTELLIGENCE PORTAL', 0, 1, 'C', fill=True)
                pdf.set_font('Courier', 'B', 10)
                pdf.cell(190, 8,
                    'ILLEGAL CONSTRUCTION DETECTION REPORT', 0, 1, 'C', fill=True)
                pdf._fill(C_TEAL)
                pdf.set_font('Courier', 'B', 9)
                pdf.cell(190, 7,
                    safe_pdf_text(f'  Sector: {selected_location}'), 0, 1, 'L', fill=True)
                pdf._fill(C_NAVY)
                pdf.set_font('Courier', '', 7)
                pdf.cell(190, 5,
                    safe_pdf_text(
                        f'  Classification: FOR OFFICIAL USE ONLY  |  '
                        f'System: Optical-Primary Multi-Temporal Fusion Engine v3  |  '
                        f'Generated: {datetime.date.today()}'
                    ), 0, 1, 'L', fill=True)
                pdf._fill(C_WHITE); pdf._text(C_BLACK)
                pdf.ln(5)

                # ---- Two-column summary: Temporal | Detection counts ----
                y_sum = pdf.get_y()

                # Left column header (Temporal Coverage)
                pdf._fill(C_TEAL); pdf._text(C_WHITE)
                pdf.set_font('Courier', 'B', 8)
                pdf.cell(93, 7, '  TEMPORAL COVERAGE', 0, 0, 'L', fill=True)

                # Right column header (Detection Summary)
                pdf.cell(97, 7, '  DETECTION SUMMARY', 0, 1, 'L', fill=True)
                pdf._fill(C_WHITE); pdf._text(C_BLACK)

                # Left kv rows (temporal)
                temporal_rows = [
                    ('T0 Baseline Start',  b_start),
                    ('T0 Baseline End',    b_end),
                    ('T1 Detection Start', a_start),
                    ('T1 Detection End',   a_end),
                    ('Mode',               safe_pdf_text(mode_label)),
                    ('AOI Radius',         f'{buffer_radius} m'),
                ]
                y_kv = pdf.get_y()
                for idx, (k, v) in enumerate(temporal_rows):
                    bg = C_LGREY if idx % 2 == 0 else C_WHITE
                    pdf._fill(bg)
                    pdf.set_font('Courier', '', 7)
                    pdf._text(C_DGREY)
                    pdf.cell(55, 5.5, f'  {k}', 'LB', 0, 'L', fill=True)
                    pdf.set_font('Courier', 'B', 7)
                    pdf._text(C_BLACK)
                    pdf.cell(38, 5.5, f'  {v}', 'RB', 1, 'L', fill=True)

                # Right column — detection counts (go back up to y_kv)
                _det_rows = [
                    ('Total Detections',     str(n_total)),
                    ('Land Clearing',        str(n_clearing)),
                    ('Vertical Structure',   str(n_vertical)),
                    ('Optical Only',         str(n_optical)),
                    ('Total Flagged Area',   f'{sum(r["Area (sqm)"] for r in rows):,} sqm'),
                    ('SAR Confirmation',     'YES' if sar_available else 'OPTICAL-ONLY'),
                ]
                for idx, (k, v) in enumerate(_det_rows):
                    bg = C_LGREY if idx % 2 == 0 else C_WHITE
                    pdf._fill(bg)
                    pdf.set_font('Courier', '', 7)
                    pdf._text(C_DGREY)
                    pdf.set_xy(103, y_kv + idx * 5.5)
                    pdf.cell(57, 5.5, f'  {k}', 'LB', 0, 'L', fill=True)
                    pdf.set_font('Courier', 'B', 7)
                    pdf._text(C_BLACK)
                    _v_color = C_RED if (k == 'Total Detections' and n_total > 0) else C_BLACK
                    pdf._text(_v_color)
                    pdf.cell(40, 5.5, f'  {v}', 'RB', 0, 'L', fill=True)
                    pdf._text(C_BLACK)
                pdf.set_y(y_kv + len(_det_rows) * 5.5 + 1)

                pdf.ln(5)

                # ---- Pipeline Parameters — two-column layout ----
                pdf.section_bar('PIPELINE PARAMETERS')

                _opt_rows = [
                    ('NDVI Loss Threshold',   f'> {ndvi_loss_thresh}'),
                    ('NDBI Gain Threshold',   f'> {ndbi_gain_thresh}'),
                    ('Max Cloud Cover',       f'<= {cloud_thresh}%'),
                    ('Min Alert Area (MMU)',  f'{min_area_sqm} sqm'),
                    ('Max Detection Slope',   f'< {slope_thresh} deg'),
                    ('Cloud Mask Method',     'SCL per-pixel (L2A)'),
                ]
                _sar_rows = [
                    ('SAR Threshold',         f'{radar_thresh} dB'),
                    ('Orbit Direction',       _rpt_orbit),
                    ('Locked Orbit No.',      f'#{_rpt_orbit_num}'),
                    ('Temporal Persistence',  f'{persistence_required} pass(es) of {_rpt_t1_passes}'),
                    ('Terrain Correction',    f'{ref_angle_deg} deg reference angle'),
                    ('SAR Mode',              _rpt_sar_mode),
                    ('Max Output Polygons',   str(fetch_limit)),
                ]

                # Column headers
                pdf._fill(C_TEAL); pdf._text(C_WHITE)
                pdf.set_font('Courier', 'B', 8)
                pdf.cell(95, 6, '  OPTICAL DETECTION (PRIMARY)', 0, 0, 'L', fill=True)
                pdf.cell(95, 6, '  SAR CONFIRMATION (SECONDARY)', 0, 1, 'L', fill=True)
                pdf._fill(C_WHITE); pdf._text(C_BLACK)

                _param_y = pdf.get_y()
                _max_rows = max(len(_opt_rows), len(_sar_rows))
                for idx in range(_max_rows):
                    bg = C_LGREY if idx % 2 == 0 else C_WHITE
                    pdf._fill(bg)
                    # Optical column (left)
                    if idx < len(_opt_rows):
                        k, v = _opt_rows[idx]
                        pdf.set_font('Courier', '', 7); pdf._text(C_DGREY)
                        pdf.cell(52, 5.5, f'  {k}', 'LB', 0, 'L', fill=True)
                        pdf.set_font('Courier', 'B', 7); pdf._text(C_BLACK)
                        pdf.cell(43, 5.5, f'  {v}', 'RB', 0, 'L', fill=True)
                    else:
                        pdf.cell(95, 5.5, '', 'LRB', 0, 'L', fill=True)
                    # SAR column (right)
                    if idx < len(_sar_rows):
                        k, v = _sar_rows[idx]
                        pdf.set_font('Courier', '', 7); pdf._text(C_DGREY)
                        pdf.cell(52, 5.5, f'  {k}', 'LB', 0, 'L', fill=True)
                        pdf.set_font('Courier', 'B', 7); pdf._text(C_BLACK)
                        pdf.cell(43, 5.5, f'  {v}', 'RB', 1, 'L', fill=True)
                    else:
                        pdf.cell(95, 5.5, '', 'LRB', 1, 'L', fill=True)

                pdf._fill(C_WHITE); pdf._text(C_BLACK)
                pdf.ln(5)

                # ---- Detection Register Table ----
                pdf.section_bar('DETECTION REGISTER')
                _reg_cols  = ['ID',  'CLASS',             'LATITUDE',   'LONGITUDE',  'AREA (sqm)', 'PIXELS', 'SAR']
                _reg_widths = [18,    55,                  28,           28,           28,           18,       15]
                pdf.table_header_row(_reg_cols, _reg_widths)

                for idx, row in enumerate(rows):
                    c_idx = row.get('Class', 0)
                    bg = C_LGREY if idx % 2 == 0 else C_WHITE
                    pdf._fill(bg)
                    pdf.set_font('Courier', '', 7)
                    _vals = [
                        row['ID'],
                        CLASS_LABEL.get(c_idx, row['Type']),
                        f"{row['Lat']:.5f} N",
                        f"{row['Lon']:.5f} E",
                        f"{row['Area (sqm)']:,}",
                        str(row['Pixels']),
                        'YES' if sar_available else 'NO',
                    ]
                    for val, w in zip(_vals, _reg_widths):
                        pdf._text(CLASS_COLOR.get(c_idx, C_BLACK) if val == CLASS_LABEL.get(c_idx, '') else C_BLACK)
                        pdf.cell(w, 5.5, f' {safe_pdf_text(str(val))}', 'B', 0, 'L', fill=True)
                    pdf._text(C_BLACK)
                    pdf.ln()
                pdf._fill(C_WHITE)

                # ================================================================
                # PAGES 2+ — ONE PAGE PER DETECTION
                # ================================================================
                for i, feat in enumerate(features[:min(fetch_limit, n_total)]):
                    props  = feat.get('properties', {})
                    geom   = feat.get('geometry',   {})
                    coords = geom.get('coordinates', [[[]]])
                    try:
                        flat = coords[0]
                        cx   = sum(c[0] for c in flat) / len(flat)
                        cy   = sum(c[1] for c in flat) / len(flat)
                    except Exception:
                        cx, cy = lon, lat

                    pixel_count = props.get('count', 0)
                    area_sqm    = pixel_count * 100
                    alert_type  = props.get('alert_type', 0)
                    label       = props.get('alert_label', 'Unknown')
                    class_color = CLASS_COLOR.get(alert_type, C_NAVY)
                    _pword      = 'pixel' if pixel_count == 1 else 'pixels'

                    # ---- Image chips in Streamlit UI ----
                    col_a, col_b, col_c = st.columns(3)
                    url_before, url_after = None, None
                    try:
                        url_before = get_s2_thumb_url(s2_before, cy, cx)
                        url_after  = get_s2_thumb_url(s2_after,  cy, cx)
                        col_a.image(url_before,
                            caption=f"DET-{i+1:03d} T0 BASELINE ({b_start} to {b_end})",
                            use_container_width=True)
                        col_b.image(url_after,
                            caption=f"DET-{i+1:03d} T1 CHANGE ({a_start} to {a_end}) | {label} | ~{area_sqm} sqm",
                            use_container_width=True)
                    except Exception as e:
                        col_a.warning(f"Sentinel-2 chip unavailable: {e}")

                    try:
                        if gmaps_api_key:
                            col_c.image(
                                f"https://maps.googleapis.com/maps/api/staticmap"
                                f"?center={cy},{cx}&zoom=18&size=320x320"
                                f"&maptype=satellite&key={gmaps_api_key}",
                                caption=f"DET-{i+1:03d} Google Satellite (zoom 18)",
                                use_container_width=True)
                        else:
                            col_c.caption("Google Satellite: API key not configured.")
                    except Exception:
                        col_c.caption("Google Satellite: unavailable.")

                    # ---- Fetch images for PDF ----
                    before_path = f"/tmp/pmrda_before_{i}.png"
                    after_path  = f"/tmp/pmrda_after_{i}.png"
                    sat_path    = None

                    try:
                        if url_before: urllib.request.urlretrieve(url_before, before_path)
                        if url_after:  urllib.request.urlretrieve(url_after,  after_path)
                    except Exception as _e:
                        log(f"Sentinel chip download error: {_e}")

                    if gmaps_api_key:
                        # 640x320 = 2:1 aspect → renders as 180mm × 90mm in PDF
                        _sat_url = (
                            f"https://maps.googleapis.com/maps/api/staticmap"
                            f"?center={cy},{cx}&zoom=18&size=640x320"
                            f"&maptype=satellite&format=png&key={gmaps_api_key}"
                        )
                        try:
                            _r = requests.get(_sat_url, timeout=12)
                            _ct = _r.headers.get('content-type', '')
                            if _r.status_code == 200 and 'image' in _ct and len(_r.content) > 5000:
                                sat_path = f"/tmp/pmrda_sat_{i}.png"
                                with open(sat_path, 'wb') as _f:
                                    _f.write(_r.content)
                            else:
                                log(f"Google Sat DET-{i+1:03d}: HTTP {_r.status_code} ct={_ct} sz={len(_r.content)}")
                        except Exception as _e:
                            log(f"Google Sat DET-{i+1:03d} error: {_e}")

                    # ==============================================================
                    # PDF PAGE — Detection evidence
                    # ==============================================================
                    pdf.add_page()

                    # ---- Detection class banner ----
                    pdf._fill(class_color); pdf._text(C_WHITE)
                    pdf.set_font('Courier', 'B', 12)
                    pdf.cell(130, 10,
                        safe_pdf_text(f'  DET-{i+1:03d}  |  {CLASS_LABEL.get(alert_type, label.upper())}'),
                        0, 0, 'L', fill=True)
                    pdf.set_font('Courier', '', 8)
                    pdf.cell(60, 10,
                        f'  {cy:.5f} N  {cx:.5f} E',
                        0, 1, 'R', fill=True)
                    pdf._fill(C_WHITE); pdf._text(C_BLACK)
                    pdf.ln(3)

                    # ---- Metrics table (4 key/value pairs in a compact 2-col grid) ----
                    pdf.section_bar('DETECTION METRICS')
                    _metric_rows = [
                        ('Estimated Area',     f'{area_sqm:,} sqm ({pixel_count} {_pword} x 100 sqm)'),
                        ('Detection Period',   f'{a_start}  to  {a_end}'),
                        ('Baseline Period',    f'{b_start}  to  {b_end}'),
                        ('SAR Confirmation',
                            f'ASCENDING orbit #{_rpt_orbit_num} | {persistence_required}/{_rpt_t1_passes} passes'
                            if sar_available else 'NOT AVAILABLE (optical-only mode)'),
                        ('NDVI Loss Detected', f'> {ndvi_loss_thresh} threshold'),
                        ('NDBI Gain Detected', f'> {ndbi_gain_thresh} threshold'),
                    ]
                    pdf.two_col_kv(_metric_rows, lw=65, rw=125)
                    pdf.ln(4)

                    # ---- Sentinel-2 images — T0 left, T1 right ----
                    pdf.section_bar('SENTINEL-2 MULTI-TEMPORAL IMAGERY  (True Colour RGB — 500m AOI)')
                    IMG_W = 89   # mm each; gap = 190 - 89 - 89 = 12mm → x: 10 and 111
                    IMG_H = 89   # 1:1 aspect (320x320 request)

                    lbl_y  = pdf.get_y()
                    img_y  = lbl_y + 6

                    # T0 label
                    pdf.img_label_bar(
                        f'BASELINE  (T0: {b_start} to {b_end})', 10, IMG_W, lbl_y)
                    # T1 label
                    pdf.img_label_bar(
                        f'CHANGE DETECTED  (T1: {a_start} to {a_end})', 111, IMG_W, lbl_y)

                    # T0 image or placeholder
                    if url_before and os.path.exists(before_path):
                        pdf.image(before_path, x=10, y=img_y, w=IMG_W)
                    else:
                        pdf.placeholder_box(10, img_y, IMG_W, IMG_H, '[T0 IMAGE UNAVAILABLE]')

                    # T1 image or placeholder
                    if url_after and os.path.exists(after_path):
                        pdf.image(after_path, x=111, y=img_y, w=IMG_W)
                    else:
                        pdf.placeholder_box(111, img_y, IMG_W, IMG_H, '[T1 IMAGE UNAVAILABLE]')

                    # Advance cursor below images
                    pdf.set_y(img_y + IMG_H + 4)

                    # ---- Google Satellite — full width below ----
                    SAT_W = 190   # full content width
                    SAT_H = 95    # 640x320 = 2:1 → 190mm × 95mm

                    pdf.section_bar('GOOGLE SATELLITE VIEW  (Zoom 18 — Current Ground Truth)')
                    sat_lbl_y = pdf.get_y()
                    sat_img_y = sat_lbl_y + 5

                    if sat_path and os.path.exists(sat_path):
                        pdf.image(sat_path, x=10, y=sat_img_y, w=SAT_W)
                    else:
                        _msg = '[GOOGLE SATELLITE — API KEY NOT CONFIGURED]' if not gmaps_api_key else '[GOOGLE SATELLITE — FETCH FAILED]'
                        pdf.placeholder_box(10, sat_img_y, SAT_W, SAT_H, _msg)

                    pdf.set_y(sat_img_y + SAT_H + 4)

                    # ---- Detection basis ----
                    pdf.section_bar('DETECTION BASIS & LEGAL NOTICE')
                    _sar_note = (
                        f"SAR confirmed: {_rpt_orbit} orbit #{_rpt_orbit_num}, "
                        f"{persistence_required}/{_rpt_t1_passes} pass persistence, "
                        f"cosine terrain-corrected at {ref_angle_deg}deg reference angle."
                        if sar_available
                        else "SAR confirmation NOT AVAILABLE — optical-only detection mode applied."
                    )
                    pdf.set_font('Courier', '', 7)
                    pdf.multi_cell(190, 4, safe_pdf_text(
                        f"Detection basis: Sentinel-2 L2A optical change analysis. "
                        f"Primary trigger: NDVI loss > {ndvi_loss_thresh} AND NDBI gain > {ndbi_gain_thresh} "
                        f"(dual-condition optical gate). Built-up context guard: T0 NDBI > 0.05. "
                        f"MMU: {min_area_sqm} sqm. Slope mask: < {slope_thresh} deg. "
                        f"{_sar_note} "
                        f"This report is machine-generated. "
                        f"Field verification is MANDATORY before any enforcement action is taken."
                    ))



                # ----- IMPORTANT: fpdf2 output syntax -----
                # fpdf v1 (old):   pdf.output(dest='S').encode('latin-1')  <- BROKEN
                # fpdf2 (correct): bytes(pdf.output())                     <- CORRECT
                pdf_bytes = bytes(pdf.output())

                st.download_button(
                    label="Download PDF Evidence Report",
                    data=pdf_bytes,
                    file_name=f"PMRDA_Report_{selected_location.split('[')[0].strip()}_{a_start}_{a_end}.pdf",
                    mime="application/pdf"
                )

        # -------- TAB 3: SPECTRAL DEBUG --------
        with tab3:
            st.markdown("#### Spectral Index Debug View")
            st.info(
                "Use this tab to visually diagnose your detections. "
                "If NDVI Loss is red over farm fields -> raise the NDVI loss threshold or use Monthly Mode. "
                "If NDBI Gain is red over wet soil/water -> raise the NDBI gain threshold. "
                "The Optical Trigger layer shows ONLY pixels where both NDVI loss AND NDBI gain are above threshold  -  "
                "this is your actual signal layer."
            )
            Map2 = geemap.Map(center=[lat, lon], zoom=14, ee_initialize=False)

            folium.TileLayer(
                tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
                attr="Google",
                name="Google Satellite",
                overlay=False,
                control=True,
            ).add_to(Map2)

            Map2.addLayer(
                s2_before,
                {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000},
                'S2 T0 True Colour', shown=False
            )
            Map2.addLayer(
                s2_after,
                {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000},
                'S2 T1 True Colour'
            )
            Map2.addLayer(
                ndvi_loss,
                {'min': -0.3, 'max': 0.5, 'palette': ['#0000ff', '#ffffff', '#ff0000']},
                f'NDVI Loss (red = >{ndvi_loss_thresh})'
            )
            Map2.addLayer(
                ndbi_gain,
                {'min': -0.2, 'max': 0.4, 'palette': ['#0000ff', '#ffffff', '#ff0000']},
                f'NDBI Gain (red = >{ndbi_gain_thresh})', shown=False
            )
            Map2.addLayer(
                optical_trigger.selfMask(),
                {'palette': '#ff00ff', 'opacity': 0.75},
                'Optical Trigger (BOTH conditions met)', shown=False
            )
            if sar_available:
                Map2.addLayer(
                    vv_change,
                    {'min': -5, 'max': 8, 'palette': ['#0000ff', '#ffffff', '#ff0000']},
                    '[SAR] VV Change (dB)', shown=False
                )
                Map2.addLayer(
                    ratio_change,
                    {'min': -3, 'max': 3, 'palette': ['#0000ff', '#ffffff', '#ff0000']},
                    '[SAR] VV-VH Ratio Change', shown=False
                )

            folium.LayerControl().add_to(Map2)
            Map2.to_streamlit(height=580)

        log("DONE.")

