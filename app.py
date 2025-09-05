
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import folium
import requests
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# Load Gradient Boosting model and preprocessors
try:
    with open('best_gradient_boosting_model_no_log.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler_no_log.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('ordinal_encoder_no_log.pkl', 'rb') as f:
        ordinal_encoder = pickle.load(f)
    with open('one_hot_encoder_no_log.pkl', 'rb') as f:
        ohe = pickle.load(f)
    st.success("Model dan preprocessor berhasil dimuat!")
except FileNotFoundError:
    st.error("File model/preprocessor tidak ditemukan. Pastikan Anda sudah menjalankan kode pickling.")
    st.stop()

# OSRM route function
def get_osrm_route(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    url = f"http://router.project-osrm.org/route/v1/driving/{pickup_lon},{pickup_lat};{dropoff_lon},{dropoff_lat}"
    params = {"overview": "full", "geometries": "geojson"}
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if "routes" in data and len(data["routes"]) > 0:
            coords = data["routes"][0]["geometry"]["coordinates"]
            coords = [(lat, lon) for lon, lat in coords]
            distance_km = data["routes"][0]["distance"] / 1000
            return distance_km, coords
        else:
            return None, None
    except Exception as e:
        print("OSRM error:", e)
        return None, None

# Streamlit config
st.set_page_config(page_title="Uber Fare Dashboard", layout="wide")
st.header("🚖 Estimasi Tarif Uber")

# Session state for locations
if "pickup" not in st.session_state:
    st.session_state.pickup = None
if "dropoff" not in st.session_state:
    st.session_state.dropoff = None

# Input features
st.markdown("🟢 Klik pertama = Pickup, 🔴 Klik kedua = Dropoff")

passenger_count = st.number_input("Jumlah Penumpang", min_value=1, max_value=6, value=1)
hour = st.number_input("Jam Pickup (0-23)", min_value=0, max_value=23, value=12)
day_of_week = st.number_input("Hari dalam Minggu (0=Senin, 6=Minggu)", min_value=0, max_value=6, value=0)
pickup_year = st.number_input("Tahun Pickup", min_value=2000, max_value=2030, value=2023)
pickup_season = st.selectbox("Pickup Season", ordinal_encoder.categories_[0], index=0)
pickup_month = st.number_input("Pickup Month", min_value=1, max_value=12, value=1)
pickup_weekday = day_of_week
pickup_hour = hour
pickup_hour_category = st.selectbox("Pickup Hour Category", ohe.categories_[3], index=0)

# Manual coordinate input
st.markdown("**Atau masukkan koordinat pickup dan dropoff secara manual:**")
manual_pickup_lat = st.text_input("Pickup Latitude (manual)", "")
manual_pickup_lon = st.text_input("Pickup Longitude (manual)", "")
manual_dropoff_lat = st.text_input("Dropoff Latitude (manual)", "")
manual_dropoff_lon = st.text_input("Dropoff Longitude (manual)", "")

st.markdown("🟢 Klik pertama = Pickup, 🔴 Klik kedua = Dropoff")

# Single map
base_map = folium.Map(location=[40.730610, -73.935242], zoom_start=12)
base_map.add_child(folium.LatLngPopup())
map_data = st_folium(base_map, height=500, width=800)

# Capture clicks
if map_data and map_data["last_clicked"]:
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

    if st.session_state.pickup is None:
        st.session_state.pickup = (lat, lon)
        st.success(f"✅ Pickup ditetapkan: {st.session_state.pickup}")
    elif st.session_state.dropoff is None:
        st.session_state.dropoff = (lat, lon)
        st.success(f"✅ Dropoff ditetapkan: {st.session_state.dropoff}")

# Prediction and route drawing

# Use manual coordinates if provided, else use map pick
def get_coordinate(val):
    try:
        return float(val)
    except:
        return None

pickup_lat = get_coordinate(manual_pickup_lat) if manual_pickup_lat else None
pickup_lon = get_coordinate(manual_pickup_lon) if manual_pickup_lon else None
dropoff_lat = get_coordinate(manual_dropoff_lat) if manual_dropoff_lat else None
dropoff_lon = get_coordinate(manual_dropoff_lon) if manual_dropoff_lon else None

if pickup_lat is None or pickup_lon is None:
    if st.session_state.pickup:
        pickup_lat, pickup_lon = st.session_state.pickup
if dropoff_lat is None or dropoff_lon is None:
    if st.session_state.dropoff:
        dropoff_lat, dropoff_lon = st.session_state.dropoff

if pickup_lat is not None and pickup_lon is not None and dropoff_lat is not None and dropoff_lon is not None:
    distance_km, route_coords = get_osrm_route(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    if distance_km is None or distance_km < 0.1:
        distance_km = np.sqrt((dropoff_lon - pickup_lon) ** 2 + (dropoff_lat - pickup_lat) ** 2) * 111
        st.warning("⚠️ Routing gagal atau jarak terlalu kecil, gunakan jarak garis lurus.")

    # Build input dictionary with all required features (adjust as needed)
    input_dict = {
        "passenger_count": passenger_count,
        "dist": distance_km,
        "pickup_year": pickup_year,
        "pickup_season": pickup_season,
        "pickup_month": pickup_month,
        "pickup_weekday": pickup_weekday,
        "pickup_hour": pickup_hour,
        "pickup_hour_category": pickup_hour_category,
        "pickup_lat": pickup_lat,
        "pickup_lon": pickup_lon,
        "dropoff_lat": dropoff_lat,
        "dropoff_lon": dropoff_lon,
        # ...add any other features required by your model
    }
    input_df = pd.DataFrame([input_dict])

    # Apply encoders and scaler
    # Ensure pickup_season type matches encoder expectation
    input_df["pickup_season"] = input_df["pickup_season"].astype(type(ordinal_encoder.categories_[0][0]))
    input_df[["pickup_season"]] = ordinal_encoder.transform(input_df[["pickup_season"]])

    # One-hot encode categorical columns
    ohe_cols = ["pickup_month", "pickup_weekday", "pickup_hour", "pickup_hour_category"]
    ohe_array = ohe.transform(input_df[ohe_cols])
    ohe_feature_names = ohe.get_feature_names_out(ohe_cols)
    ohe_df = pd.DataFrame(ohe_array, columns=ohe_feature_names, index=input_df.index)
    input_df = pd.concat([input_df.drop(columns=ohe_cols), ohe_df], axis=1)

    # Scale numerical columns
    input_df[["passenger_count", "dist", "pickup_year"]] = scaler.transform(
        input_df[["passenger_count", "dist", "pickup_year"]]
    )

    # Ensure columns are in the same order as during training
    input_df = input_df[model.feature_names_in_]

    prediction = model.predict(input_df)
    fare = max(prediction[0], 3.0)

    # Draw route on same map
    route_map = folium.Map(location=[pickup_lat, pickup_lon], zoom_start=13)
    folium.Marker([pickup_lat, pickup_lon], popup="Pickup", icon=folium.Icon(color="green")).add_to(route_map)
    folium.Marker([dropoff_lat, dropoff_lon], popup="Dropoff", icon=folium.Icon(color="red")).add_to(route_map)
    coords = route_coords or [[pickup_lat, pickup_lon], [dropoff_lat, dropoff_lon]]
    folium.PolyLine(coords, color="blue", weight=5).add_to(route_map)
    st_folium(route_map, height=500, width=800)

    st.success(f"💰 Estimasi Tarif: ${fare:.2f} (Jarak: {distance_km:.2f} km)")

# Reset button
if st.button("🔄 Reset Lokasi"):
    st.session_state.pickup = None
    st.session_state.dropoff = None
    st.rerun()
