import streamlit as st
import ee
import geemap.foliumap as geemap
import osmnx as ox
import requests
from fpdf import FPDF
import datetime
import os
import urllib.request
import time
import math
from collections import Counter
from google.oauth2 import service_account

# ==========================================
# 1. SYSTEM INITIALIZATION & C2 STYLING
# ==========================================
st.set_page_config(page_title="PMRDA GEWS v2 | Enhanced Node", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Share Tech Mono', monospace !important;
    }

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

    [data-testid="stMetricValue"] {
        color: #00ffcc !important;
        font-size: 1.8rem !important;
    }

    .terminal-console {
        background-color: #000000;
        border: 1px solid #333333;
        padding: 15px;
        color: #00ff00;
        font-family: 'Share Tech Mono', monospace;
        height: 200px;
        overflow-y: hidden;
        margin-bottom: 20px;
    }

    .v2-badge {
        background-color: #003300;
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
    hr { border-color: #333333; }
    </style>
""", unsafe_allow_html=True)

st.title("PMRDA Geospatial Intelligence Portal")
st.markdown(
    "<span style='color:#888;'>NODE IDENT: ALPHA-ACTUAL-v2 | SUBSYSTEM: TERRAIN-CORRECTED MULTI-TEMPORAL FUSION ENGINE</span>"
    "<span class='v2-badge'>v2 ENHANCED ACCURACY</span>",
    unsafe_allow_html=True
)
st.markdown("---")

row1_col1, row1_col2 = st.columns(2)
row1_col1.metric("Uplink Status", "SECURE / TLS 1.3")
row1_col2.metric("Primary Sensor", "SAR C-Band (VV+VH)")

row2_col1, row2_col2 = st.columns(2)
row2_col1.metric("Optical Verification", "MSI (10m) + SCL Mask")
row2_col2.metric("Analysis Mode", "MULTI-TEMPORAL PERSISTENT")
st.markdown("---")

# ==========================================
# 2. EARTH ENGINE AUTHENTICATION
# ==========================================
try:
    if "gcp_service_account" in st.secrets:
        # CLOUD DEPLOYMENT ROUTE: Secure memory injection
        key_dict = dict(st.secrets["gcp_service_account"])
        creds = service_account.Credentials.from_service_account_info(
            key_dict,
            scopes=['https://www.googleapis.com/auth/earthengine']
        )
        ee.Initialize(credentials=creds, project='localqol')
        ee.data._credentials = creds  # MONKEY-PATCH: Forces geemap to recognize the cloud credentials
    else:
        # LOCAL MAC ROUTE: Fallback to terminal authentication
        ee.Initialize(project='localqol')
except Exception as e:
    st.error(f"SYSTEM HALTED: Earth Engine authentication protocol failed. {e}")
    st.stop()

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def get_s1_collection(roi, start_date, end_date, orbit_pass, orbit_number=None):
    """
    Filtered Sentinel-1 IW GRD collection.
    Locked to a single orbit pass direction (and optionally relative orbit number)
    to eliminate geometric artifacts from mixing ascending/descending passes.
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
    Returns a GEE map function that normalizes SAR backscatter to a reference
    incidence angle using cosine correction, removing slope-induced artifacts.

    Formula: sigma_corrected = sigma_linear * cos(ref) / cos(local_incidence)
    """
    cos_ref = math.cos(ref_angle_deg * math.pi / 180)

    def apply(image):
        angle_rad = image.select('angle').multiply(math.pi / 180)
        cos_inc = angle_rad.cos()

        vv_tc = (
            ee.Image(10).pow(image.select('VV').divide(10))
            .multiply(cos_ref).divide(cos_inc)
            .log10().multiply(10)
            .rename('VV_tc')
        )
        vh_tc = (
            ee.Image(10).pow(image.select('VH').divide(10))
            .multiply(cos_ref).divide(cos_inc)
            .log10().multiply(10)
            .rename('VH_tc')
        )
        return image.addBands(vv_tc).addBands(vh_tc)

    return apply


def mask_s2_scl(image):
    """
    Pixel-level cloud and shadow masking using the Scene Classification Layer (SCL).
    Masks: cloud shadow (3), cloud medium prob (8), cloud high prob (9),
           cirrus (10), snow/ice (11).
    Superior to scene-level CLOUDY_PIXEL_PERCENTAGE filtering alone.
    """
    scl = image.select('SCL')
    mask = (
        scl.neq(3).And(scl.neq(8)).And(scl.neq(9))
           .And(scl.neq(10)).And(scl.neq(11))
    )
    return image.updateMask(mask)


# ==========================================
# 4. SECTOR COORDINATE DICTIONARY
# ==========================================
pmrda_villages = {
    "Hinjewadi (Phase 1 & 2) [GRID: HNJ-1]": [18.5913, 73.7389],
    "Maan (Phase 3) [GRID: MAN-3]": [18.5770, 73.6850],
    "Marunji [GRID: MRN-0]": [18.6010, 73.7220],
    "Mahalunge [GRID: MHL-0]": [18.5675, 73.7460],
    "Sus Sector [GRID: SUS-0]": [18.5435, 73.7435],
    "Wakad Node [GRID: WKD-0]": [18.5987, 73.7688],
    "Manual Override (Custom Coordinates)": [None, None]
}

# ==========================================
# 5. CONTROL PANEL (SIDEBAR)
# ==========================================
with st.sidebar:
    st.header("PIPELINE PARAMETERS")

    st.subheader("1. Target Acquisition")
    selected_location = st.selectbox("Select Jurisdictional Sector:", options=list(pmrda_villages.keys()))

    if selected_location == "Manual Override (Custom Coordinates)":
        lat = st.number_input("Latitude (EPSG:4326)", value=18.585000, format="%.6f")
        lon = st.number_input("Longitude (EPSG:4326)", value=73.715000, format="%.6f")
    else:
        lat = pmrda_villages[selected_location][0]
        lon = pmrda_villages[selected_location][1]
        st.info(f"TARGET LOCKED: {lat}, {lon}")

    st.subheader("2. Temporal Baselines")
    before_dates = st.date_input("T0 Epoch (Pre-Analysis Baseline)",
                                 [datetime.date(2024, 1, 1), datetime.date(2024, 3, 31)])
    after_dates = st.date_input("T1 Epoch (Current State)",
                                [datetime.date(2026, 1, 1), datetime.date(2026, 4, 5)])

    st.subheader("3. Backscatter Calibration")
    radar_thresh = st.slider("Vertical Structure SAR Threshold (dB)",
                             min_value=2.0, max_value=8.0, value=5.5, step=0.1)

    with st.expander("ADVANCED CALIBRATION"):
        fetch_limit = st.slider("Vector Output Limit (Max Entities)", 5, 50, 10, step=1)
        ndbi_thresh = st.number_input("Minimum NDBI Variance Matrix", value=0.15, step=0.01)
        buffer_radius = st.number_input("Analysis Radius (Meters)", value=3000, step=500)

    st.subheader("4. SAR Orbit Configuration")
    orbit_pass = st.selectbox(
        "Orbit Pass Direction",
        ["ASCENDING", "DESCENDING"],
        help=(
            "Lock to a single orbit direction. Mixing ascending and descending passes "
            "produces phantom backscatter changes due to differing viewing geometries. "
            "ASCENDING is typical for the PMRDA region."
        )
    )

    st.subheader("5. Accuracy Enhancement")
    min_area_sqm = st.slider(
        "Minimum Alert Area (sqm)", 100, 2000, 400, step=50,
        help=(
            "Filters detections smaller than this footprint using connected component analysis. "
            "Eliminates single-pixel noise. Real construction is typically ≥200 sqm."
        )
    )
    persistence_required = st.slider(
        "Temporal Persistence (SAR passes required)", 1, 5, 2,
        help=(
            "Number of individual Sentinel-1 passes (not composites) that must independently "
            "confirm the change. Eliminates transient events: floods, fires, equipment, crop burns."
        )
    )
    ref_angle_deg = st.slider(
        "Terrain Correction Reference Angle (°)", 20, 50, 40,
        help=(
            "All SAR data is normalized to this incidence angle, removing slope-induced "
            "backscatter artifacts. 40° is the standard reference for C-band."
        )
    )

    st.subheader("6. External APIs")
    try:
        gmaps_api_key = st.secrets["GMAPS_API_KEY"]
        st.success("Google Static Optical API: CONNECTED")
    except KeyError:
        gmaps_api_key = st.text_input("Optical API Key (Optional)", type="password")

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("EXECUTE ANALYSIS PIPELINE", type="primary", use_container_width=True)

# ==========================================
# 6. MAIN PIPELINE EXECUTION
# ==========================================
if run_btn:
    if len(before_dates) != 2 or len(after_dates) != 2:
        st.error("SEQUENCE ABORTED: Temporal Baselines require exact Start/End dates.")
        st.stop()

    telemetry_placeholder = st.empty()
    telemetry_logs = [
        f"> Initiating connection to Earth Engine datacenters...",
        f"> Target coordinates locked: Lat {lat}, Lon {lon}",
        f"> Filtering Sentinel-1 to {orbit_pass} orbit pass only [orbit geometry lock]...",
        f"> Auto-detecting dominant relative orbit number from T0 epoch...",
        f"> Applying radiometric terrain correction (cosine model, ref: {ref_angle_deg}deg)...",
        f"> Building per-pass change flag stack for temporal persistence analysis...",
        f"> Requiring {persistence_required} independent SAR passes to confirm change...",
        f"> Requesting Sentinel-2 MSI optical data (relaxed scene-level threshold)...",
        f"> Applying pixel-level SCL cloud/shadow mask (classes 3, 8, 9, 10, 11)...",
        f"> Computing VV change, VH change, and VV-VH ratio change matrices...",
        f"> Applying improved phenological filter (NDVI threshold: 0.50)...",
        f"> Running minimum mapping unit filter ({min_area_sqm} sqm threshold)...",
        f"> Querying OpenStreetMap for authorized structure exclusion...",
        f"> Compiling verified anomaly vectors..."
    ]

    console_text = ""
    for log in telemetry_logs:
        console_text += log + "<br>"
        telemetry_placeholder.markdown(
            f"<div class='terminal-console'>{console_text}</div>",
            unsafe_allow_html=True
        )
        time.sleep(0.3)

    with st.spinner('FINALIZING RASTER COMPUTATIONS...'):
        b_start = before_dates[0].strftime('%Y-%m-%d')
        b_end   = before_dates[1].strftime('%Y-%m-%d')
        a_start = after_dates[0].strftime('%Y-%m-%d')
        a_end   = after_dates[1].strftime('%Y-%m-%d')

        roi = ee.Geometry.Point([lon, lat]).buffer(buffer_radius)
        terrain_corrector = make_terrain_corrector(ref_angle_deg)

        # ----------------------------------------------------------
        # 6a. Orbit Locking — detect dominant relative orbit number
        # ----------------------------------------------------------
        before_s1_raw = get_s1_collection(roi, b_start, b_end, orbit_pass)
        after_s1_raw  = get_s1_collection(roi, a_start, a_end, orbit_pass)

        orbit_numbers = before_s1_raw.aggregate_array('relativeOrbitNumber_start').getInfo()
        if not orbit_numbers:
            st.error(
                f"No Sentinel-1 {orbit_pass} data found for T0 epoch. "
                "Try the opposite orbit direction or widen the date range."
            )
            st.stop()

        dominant_orbit = Counter(orbit_numbers).most_common(1)[0][0]

        before_s1_col = before_s1_raw.filter(
            ee.Filter.eq('relativeOrbitNumber_start', dominant_orbit)
        )
        after_s1_col = after_s1_raw.filter(
            ee.Filter.eq('relativeOrbitNumber_start', dominant_orbit)
        )

        after_count = after_s1_col.size().getInfo()
        if after_count == 0:
            st.error(
                f"No Sentinel-1 data for T1 epoch on orbit #{dominant_orbit}. "
                "Adjust date range or switch orbit direction."
            )
            st.stop()

        st.info(
            f"SAR ORBIT LOCKED: {orbit_pass} | Relative Orbit #{dominant_orbit} "
            f"| {after_count} T1 pass(es) available for persistence analysis"
        )

        # ----------------------------------------------------------
        # 6b. Terrain-Corrected Composites
        # ----------------------------------------------------------
        s1_before = (
            before_s1_col.map(terrain_corrector)
            .select(['VV_tc', 'VH_tc'])
            .median()
        )
        s1_after = (
            after_s1_col.map(terrain_corrector)
            .select(['VV_tc', 'VH_tc'])
            .median()
        )

        # ----------------------------------------------------------
        # 6c. Multi-Temporal Persistence
        # Each individual T1 pass is independently checked against the
        # T0 median. A pixel must show significant change in at least
        # `persistence_required` passes to be considered a real detection.
        # This eliminates floods, fires, crop burns, parked equipment, etc.
        # ----------------------------------------------------------
        vv_persist_thresh = radar_thresh * 0.6   # per-pass threshold (lower than final)
        before_vv = s1_before.select('VV_tc')

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

        # ----------------------------------------------------------
        # 6d. Change Metrics: VV + VV-VH Ratio
        # VV/VH ratio discriminates double-bounce (buildings) from
        # volume scattering (vegetation). Ratio increasing means more
        # structural surface, not just bare soil or crops.
        # ----------------------------------------------------------
        vv_change    = s1_after.select('VV_tc').subtract(s1_before.select('VV_tc'))
        ratio_before = s1_before.select('VV_tc').subtract(s1_before.select('VH_tc'))
        ratio_after  = s1_after.select('VV_tc').subtract(s1_after.select('VH_tc'))
        ratio_change = ratio_after.subtract(ratio_before)

        # ----------------------------------------------------------
        # 6e. Sentinel-2 with Pixel-Level SCL Cloud Masking
        # Scene-level CLOUDY_PIXEL_PERCENTAGE is scene metadata —
        # clouds within the valid 90% still corrupt pixels. SCL provides
        # per-pixel classification we use to mask residual contamination.
        # ----------------------------------------------------------
        s2_bands_needed = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'SCL']

        s2_before = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(roi).filterDate(b_start, b_end)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
            .select(s2_bands_needed)
            .map(mask_s2_scl)
            .median()
        )
        s2_after = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(roi).filterDate(a_start, a_end)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
            .select(s2_bands_needed)
            .map(mask_s2_scl)
            .median()
        )

        # ----------------------------------------------------------
        # 6f. Spectral Indices
        # ----------------------------------------------------------
        ndbi_before = s2_before.normalizedDifference(['B11', 'B8'])
        ndbi_after  = s2_after.normalizedDifference(['B11', 'B8'])
        ndbi_change = ndbi_after.subtract(ndbi_before)

        ndvi_before = s2_before.normalizedDifference(['B8', 'B4'])
        ndvi_after  = s2_after.normalizedDifference(['B8', 'B4'])

        # ----------------------------------------------------------
        # 6g. Improved Discriminators
        # ----------------------------------------------------------

        # Road filter: NDBI increases (pavement) but radar is stable
        # and ratio doesn't rise (no vertical structure)
        is_road = (
            ndbi_change.gt(ndbi_thresh)
            .And(vv_change.abs().lt(2.0))
            .And(ratio_change.lt(0.5))
        )

        # Agricultural mask: stricter threshold (0.5 vs original 0.3)
        # Avoids crop-cycle transitions triggering false positives
        was_crop = ndvi_before.gt(0.5)
        smart_radar_thresh = ee.Image(radar_thresh).where(
            was_crop, ee.Image(radar_thresh).add(2.5)
        )

        # ----------------------------------------------------------
        # 6h. Classification
        # Land clearing: moderate VV increase + NDBI uptick + ratio
        #   increase (bare soil rougher than crops) + no post-vegetation
        # Vertical structure: high VV + strong ratio rise (double-bounce
        #   from building walls) + high NDBI change
        # Both require persistence_mask — change must persist across
        #   multiple independent S1 passes.
        # ----------------------------------------------------------
        is_clearing = (
            vv_change.gte(3.0)
            .And(vv_change.lt(smart_radar_thresh))
            .And(ndbi_change.gt(0.05))
            .And(ndvi_after.lt(0.3))
            .And(ratio_change.gt(0.5))
            .And(is_road.Not())
            .And(persistence_mask)
        )

        is_vertical = (
            vv_change.gte(smart_radar_thresh)
            .And(ratio_change.gt(1.0))
            .And(ndbi_change.gt(ndbi_thresh))
            .And(ndvi_after.lt(0.3))
            .And(is_road.Not())
            .And(persistence_mask)
        )

        # ----------------------------------------------------------
        # 6i. Minimum Mapping Unit Filter
        # connectedPixelCount eliminates isolated pixels that pass
        # all spectral thresholds but have no spatial coherence.
        # Real construction has a minimum contiguous footprint.
        # ----------------------------------------------------------
        min_pixels = max(1, min_area_sqm // 100)  # 10m pixels → 100 sqm each

        clearing_mmu = is_clearing.updateMask(
            is_clearing.connectedPixelCount(500, True).gte(min_pixels)
        )
        vertical_mmu = is_vertical.updateMask(
            is_vertical.connectedPixelCount(500, True).gte(min_pixels)
        )

        alert_img = (
            ee.Image(0)
            .where(clearing_mmu, 1)
            .where(vertical_mmu, 2)
            .selfMask()
        )

        # ----------------------------------------------------------
        # 6j. OSM Authorized Structure Masking
        # ----------------------------------------------------------
        try:
            osm_gdf = ox.features_from_point((lat, lon), {"building": True}, dist=buffer_radius)
            osm_ee = geemap.gdf_to_ee(osm_gdf[['geometry']])
            buffered_osm = osm_ee.map(lambda f: f.buffer(2))
            osm_raster_mask = ee.Image.constant(0).paint(buffered_osm, 1)
            final_alerts = alert_img.updateMask(osm_raster_mask.eq(0))
        except Exception as e:
            st.warning(f"OSM Database unreachable ({e}). Proceeding with unmasked analysis.")
            final_alerts = alert_img
            osm_ee = None

        console_text += "> MULTI-TEMPORAL COMPUTATIONS COMPLETE. RENDERING DATA LAYERS...<br>"
        telemetry_placeholder.markdown(
            f"<div class='terminal-console'>{console_text}</div>",
            unsafe_allow_html=True
        )

        # ==========================================
        # 7. DATA VISUALIZATION & OUTPUT
        # ==========================================
        tab1, tab2 = st.tabs(["GEOSPATIAL RENDER", "VECTOR EXTRACTION LOG"])

        with tab1:
            # ee_initialize=False OVERRIDE: Prevents the map from triggering the legacy auth check
            Map = geemap.Map(center=[lat, lon], zoom=14, ee_initialize=False)
            Map.addLayer(
                final_alerts.eq(1).selfMask(),
                {'palette': 'orange'},
                'Pre-Construction (Land Clearing)'
            )
            Map.addLayer(
                final_alerts.eq(2).selfMask(),
                {'palette': 'red'},
                'Confirmed Vertical Structure'
            )
            # Debug layers (hidden by default — toggle in map legend)
            Map.addLayer(
                ratio_change,
                {'min': -3, 'max': 3, 'palette': ['blue', 'white', 'red']},
                '[debug] VV-VH Ratio Change',
                shown=False
            )
            Map.addLayer(
                persistence_count,
                {'min': 0, 'max': after_count, 'palette': ['black', 'yellow', 'white']},
                f'[debug] Persistence Count (0-{after_count} passes)',
                shown=False
            )
            if osm_ee:
                Map.addLayer(osm_ee, {'color': 'blue'}, 'Authorized OSM Structures')
            Map.to_streamlit(height=650)

        # ==========================================
        # 8. DOSSIER GENERATION
        # ==========================================
        with tab2:
            st.markdown("### COMPILING INTELLIGENCE DOSSIER")

            clearing_vectors = (
                final_alerts.eq(1).selfMask()
                .reduceToVectors(
                    geometry=roi, crs='EPSG:4326', scale=10,
                    geometryType='centroid', maxPixels=1e8
                )
                .limit(fetch_limit)
                .map(lambda f: f.set('alert_type', 1))
            )
            vertical_vectors = (
                final_alerts.eq(2).selfMask()
                .reduceToVectors(
                    geometry=roi, crs='EPSG:4326', scale=10,
                    geometryType='centroid', maxPixels=1e8
                )
                .limit(fetch_limit)
                .map(lambda f: f.set('alert_type', 2))
            )

            combined_vectors = clearing_vectors.merge(vertical_vectors)
            points_data = combined_vectors.getInfo()

            if 'features' in points_data and len(points_data['features']) > 0:
                total_alerts = len(points_data['features'])
                st.success(f"PIPELINE COMPLETE: {total_alerts} ANOMALIES SECURED.")

                with st.expander("VIEW RAW GEOJSON TELEMETRY"):
                    st.json(points_data)

                class PMRDAReport(FPDF):
                    def header(self):
                        self.set_font('Courier', 'B', 14)
                        self.cell(0, 10, 'GEOSPATIAL INTELLIGENCE REPORT - PMRDA v2 [RESTRICTED]', 0, 1, 'C')
                        self.set_font('Courier', '', 9)
                        self.cell(0, 5, f'Generated: {datetime.date.today()} | Sector: {selected_location}', 0, 1, 'C')
                        self.cell(
                            0, 5,
                            f'Orbit: {orbit_pass} #{dominant_orbit} | '
                            f'Terrain Correction: {ref_angle_deg}deg ref | '
                            f'Persistence: {persistence_required}/{after_count} passes | '
                            f'MMU: {min_area_sqm} sqm',
                            0, 1, 'C'
                        )
                        self.line(10, 28, 200, 28)
                        self.ln(10)

                pdf = PMRDAReport()

                def get_s2_thumb(img, lat, lon, filename):
                    box = ee.Geometry.Point([lon, lat]).buffer(150)
                    url = img.visualize(bands=['B4', 'B3', 'B2'], min=0, max=3000).getThumbURL({
                        'region': box, 'dimensions': '300x300', 'format': 'png'
                    })
                    urllib.request.urlretrieve(url, filename)

                for idx, feature in enumerate(points_data['features']):
                    lon_feat, lat_feat = feature['geometry']['coordinates']
                    alert_type = feature['properties']['alert_type']

                    tag   = "CLASS 1: LAND CLEARING ANOMALY" if alert_type == 1 else "CLASS 2: VERTICAL STRUCTURE ANOMALY"
                    color = (200, 100, 0) if alert_type == 1 else (150, 0, 0)

                    pdf.add_page()
                    pdf.set_font('Courier', 'B', 12)
                    pdf.set_text_color(*color)
                    pdf.cell(0, 10, f"TARGET ID #{idx + 1} of {total_alerts} - {tag}", 0, 1)

                    pdf.set_text_color(0, 0, 0)
                    pdf.set_font('Courier', '', 10)
                    pdf.cell(0, 8, f"COORDINATES: {lat_feat:.6f}, {lon_feat:.6f}", 0, 1)
                    pdf.cell(0, 7, f"ORBIT: {orbit_pass} | RELATIVE ORBIT #{dominant_orbit} | T1 PASSES: {after_count}", 0, 1)
                    pdf.cell(0, 7, f"PERSISTENCE: {persistence_required}/{after_count} passes confirmed change", 0, 1)
                    pdf.cell(0, 7, f"MIN MAPPING UNIT: {min_area_sqm} sqm | TERRAIN CORRECTION: {ref_angle_deg}deg ref angle", 0, 1)

                    before_file = f"before_{idx}.png"
                    after_file  = f"after_{idx}.png"
                    get_s2_thumb(s2_before, lat_feat, lon_feat, before_file)
                    get_s2_thumb(s2_after,  lat_feat, lon_feat, after_file)

                    pdf.ln(5)
                    pdf.set_font('Courier', 'B', 10)
                    pdf.cell(90, 10, "EPOCH T0 (Sentinel-2 MSI)", 0, 0)
                    pdf.cell(90, 10, "EPOCH T1 (Sentinel-2 MSI)", 0, 1)
                    pdf.image(before_file, x=10, w=80)
                    pdf.image(after_file, x=105, y=pdf.get_y() - 80, w=80)
                    os.remove(before_file)
                    os.remove(after_file)

                    if gmaps_api_key:
                        pdf.ln(5)
                        pdf.cell(0, 10, "OPTICAL VERIFICATION SENSOR (High-Res API):", 0, 1)
                        img_url = (
                            f"https://maps.googleapis.com/maps/api/staticmap"
                            f"?center={lat_feat},{lon_feat}&zoom=19&size=500x500"
                            f"&maptype=satellite&markers=color:red|{lat_feat},{lon_feat}"
                            f"&key={gmaps_api_key}"
                        )
                        img_path = f"proof_{idx}.png"
                        resp = requests.get(img_url)
                        if resp.status_code == 200:
                            with open(img_path, 'wb') as f:
                                f.write(resp.content)
                            pdf.image(img_path, x=55, w=100)
                            os.remove(img_path)
                        else:
                            pdf.cell(0, 10, "[VERIFICATION ERROR: Target acquisition failed]", 0, 1)

                pdf_output_path = "temp_report_v2.pdf"
                pdf.output(pdf_output_path)

                with open(pdf_output_path, "rb") as f:
                    pdf_bytes = f.read()

                st.download_button(
                    label="DOWNLINK SECURE INTELLIGENCE DOSSIER v2 (PDF)",
                    data=pdf_bytes,
                    file_name=f"PMRDA_v2_Report_{selected_location.split('[')[0].strip().replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )

                os.remove(pdf_output_path)

            else:
                st.info("SCAN COMPLETE: No targets meeting defined parameters detected in this sector.")
