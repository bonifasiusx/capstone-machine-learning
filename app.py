import streamlit as st
import joblib
import pandas as pd
import numpy as np

# CONFIGURATION
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

# LOAD MODEL
@st.cache_resource
def load_model():
    try:
        return joblib.load("CA_housing_price_regressor.joblib")
    except:
        st.error("❌ Model file 'CA_housing_price_regressor.joblib' not found")
        st.stop()

# MAIN APP
st.title("🏡 California Housing Price Predictor")

model = load_model()

# Input Form
with st.form("predict_form"):
    st.subheader("Enter Property Details")
    
    # Location
    col1, col2 = st.columns(2)
    with col1:
        longitude = st.slider("Longitude", -125.0, -113.0, -118.0)
        latitude = st.slider("Latitude", 32.0, 42.0, 34.0)
    with col2:
        ocean = st.selectbox("Ocean Proximity", OCEAN_OPTIONS, index=1)
    
    # Property Details
    col3, col4 = st.columns(2)
    with col3:
        age = st.number_input("House Age (years)", 1, 60, 25)
        rooms = st.number_input("Total Rooms", 10, 50000, 2500)
        bedrooms = st.number_input("Total Bedrooms", 1, 15000, 500)
    
    with col4:
        population = st.number_input("Population", 3, 60000, 1500)
        households = st.number_input("Households", 1, 10000, 450)
        income = st.number_input("Median Income ($)", 5000, 200000, 50000, 1000)
    
    predict = st.form_submit_button("🎯 Predict Price", type="primary")

# Make Prediction
if predict:
    # Prepare data
    data = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "houseMedAge": age,
        "totalRooms": rooms,
        "totalBedrooms": bedrooms,
        "population": population,
        "households": households,
        "medIncome": income / 10000,  # Convert to model scale
        "oceanProx": ocean
    }])
    
    # Predict
    try:
        price = model.predict(data)[0]
        
        # Display Result
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin: 2rem 0;
        ">
            <h2 style="margin: 0; font-size: 3rem;">
                ${price:,.0f}
            </h2>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">
                Estimated House Value
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Price Category
        if price < 100000:
            st.success("💰 Budget-Friendly")
        elif price < 300000:
            st.info("🏠 Moderate")
        elif price < 500000:
            st.warning("💎 Premium")
        else:
            st.error("🌟 Luxury")
        
        # Quick Stats
        with st.expander("📊 Details"):
            rooms_per_house = rooms / households
            people_per_house = population / households
            bedroom_ratio = bedrooms / rooms
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Rooms/Household", f"{rooms_per_house:.1f}")
                st.metric("People/Household", f"{people_per_house:.1f}")
            with col_b:
                st.metric("Bedroom Ratio", f"{bedroom_ratio:.1%}")
                st.metric("Income Level", f"${income:,}")
    
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")

# Batch Processing
st.markdown("---")
with st.expander("📊 Batch Processing"):
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df)} records")
        
        if st.button("Process All"):
            df['medIncome'] = df['medIncome'] / 10000
            predictions = model.predict(df)
            df['medIncome'] = df['medIncome'] * 10000
            df['predicted_price'] = predictions
            
            st.success("✅ Done!")
            st.dataframe(df[['longitude', 'latitude', 'predicted_price']])
            
            csv = df.to_csv(index=False)
            st.download_button("📥 Download", csv, "predictions.csv")

# Help
with st.expander("ℹ️ Help"):
    st.markdown("""
    **How to use:**
    1. Adjust location sliders for California coordinates
    2. Enter property details (rooms, bedrooms, etc.)
    3. Enter median income in actual dollars
    4. Click "Predict Price"
    
    **Tips:**
    - All values should be for the area/block, not individual houses
    - Use realistic California coordinates
    - Income is median household income for the area
    """)

st.caption("Built with Streamlit • California Housing Dataset")