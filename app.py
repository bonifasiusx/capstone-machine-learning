import os, pickle
import numpy as np
import pandas as pd
import streamlit as st

# CONFIG
st.set_page_config(
    page_title="California Housing Predictor",
    page_icon="🏡",
    layout="centered"
)

OCEAN_OPTIONS = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']

# FEATURE ENGINEERING
class FeatureEngineer:
    def fit(self, X, y=None):
        self.rooms_95 = X['totalRooms'].quantile(0.95)
        self.age_max = X['houseMedAge'].max()
        self.bin_edges = [0, 10, 25, self.age_max]
        return self

    def transform(self, X):
        X = X.copy()
        X['roomsPerHouseholds'] = X['totalRooms'] / X['households']
        X['bedroomsPerRoom'] = X['totalBedrooms'] / X['totalRooms']
        X['popPerHouseholds'] = X['population'] / X['households']
        X['isManyRooms'] = (X['totalRooms'] > getattr(self, 'rooms_95', 2016)).astype(int)
        X['housingAgeBin'] = pd.cut(
            X['houseMedAge'],
            bins=getattr(self, 'bin_edges', [0, 10, 25, 52]),
            labels=['young', 'middle', 'old'],
            include_lowest=True,
            right=True
        )
        return X
    
# LOAD PICKLED MODEL
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "CA_housing_price_regressor.sav")
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'CA_housing_price_regressor.sav' not found in app directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

model = load_model()

# UI
st.title("🏡 California Housing Price Predictor")

with st.form("predict_form"):
    st.subheader("Enter Property Details")

    # Location
    col1, col2 = st.columns(2)
    with col1:
        longitude = st.slider("Longitude", -125.0, -113.0, -118.0)
        latitude  = st.slider("Latitude",   32.0,    42.0,   34.0)
    with col2:
        ocean = st.selectbox("Ocean Proximity", OCEAN_OPTIONS, index=1)

    # Property details
    col3, col4 = st.columns(2)
    with col3:
        age      = st.number_input("House Age (years)", 1, 60, 25)
        rooms    = st.number_input("Total Rooms", 10, 50000, 2500)
        bedrooms = st.number_input("Total Bedrooms", 1, 15000, 500)

    with col4:
        population = st.number_input("Population", 3, 60000, 1500)
        households = st.number_input("Households", 1, 10000, 450)
        income     = st.number_input("Median Income ($)", 5000, 200000, 50000, 1000)

    predict = st.form_submit_button("🎯 Predict Price", type="primary")

# PREDICT
if predict:
    data = pd.DataFrame([{
        "longitude":     float(longitude),
        "latitude":      float(latitude),
        "houseMedAge":   int(age),
        "totalRooms":    int(rooms),
        "totalBedrooms": int(bedrooms),
        "population":    int(population),
        "households":    int(households),
        # dataset uses income in 10k units
        "medIncome":     float(income) / 10000.0,
        "oceanProx":     str(ocean)
    }])

    try:
        price = float(model.predict(data)[0])

        # Pretty card
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem; border-radius: 15px; color: white;
            text-align: center; margin: 2rem 0;">
            <h2 style="margin:0; font-size:3rem;">${price:,.0f}</h2>
            <p style="margin:.5rem 0 0; font-size:1.2rem;">Estimated House Value</p>
        </div>
        """, unsafe_allow_html=True)

        # Category hint
        if price < 100_000:
            st.success("💰 Budget-Friendly")
        elif price < 300_000:
            st.info("🏠 Moderate")
        elif price < 500_000:
            st.warning("💎 Premium")
        else:
            st.error("🌟 Luxury")

        # Quick stats
        with st.expander("📊 Details"):
            rooms_per_house   = rooms / households
            people_per_house  = population / households
            bedroom_ratio     = bedrooms / rooms

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Rooms/Household", f"{rooms_per_house:.1f}")
                st.metric("People/Household", f"{people_per_house:.1f}")
            with col_b:
                st.metric("Bedroom Ratio", f"{bedroom_ratio:.1%}")
                st.metric("Income Level", f"${income:,}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# BATCH PREDICTION
st.markdown("---")
with st.expander("📊 Batch Processing"):
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df)} records")

        needed = {"longitude","latitude","houseMedAge","totalRooms","totalBedrooms",
                  "population","households","medIncome","oceanProx"}
        missing = needed - set(df.columns)
        if missing:
            st.error(f"Missing required columns: {sorted(missing)}")
        else:
            if st.button("Process All"):

                preds = model.predict(df)
                df['predicted_price'] = preds

                st.success("✅ Done!")
                st.dataframe(df[['longitude', 'latitude', 'predicted_price']])

                st.download_button(
                    "📥 Download",
                    df.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv"
                )

# HELP
with st.expander("ℹ️ Help"):
    st.markdown("""
**How to use**
1. Adjust location for California coordinates
2. Enter area-level property details (rooms, households, etc.)
3. Median income is in dollars; the app converts it for the model
4. Click **Predict Price**

**Tips**
- Use realistic values for your area/block
- `oceanProx` should match the dataset categories
""")

st.caption("Built with Streamlit • California Housing Dataset")
