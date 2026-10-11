"""California home value estimator: Streamlit front end for the final AVM (game plan Week 9).

Runs locally. Needs the cleaned data, the school-district shapefile and the trained final model:
    python src/cleaning.py && python src/geo.py && python src/models.py --tuned && python src/evaluate_final.py
    streamlit run app.py

All valuation logic lives in src/predict.py; this file is only the form and the result display.
"""

import os
import sys
from datetime import date

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import predict  # noqa: E402

st.set_page_config(page_title="CA Home Value Estimator", page_icon="🏠", layout="centered")


@st.cache_resource(show_spinner="Loading the model and school-district boundaries…")
def load_valuer():
    return predict.Valuer()


def money(x):
    return f"${x / 1e6:,.2f}M" if x >= 1e6 else f"${x / 1e3:,.0f}k"


YES_NO = {"Unknown": None, "Yes": True, "No": False}

valuer = load_valuer()
pre = valuer.model.named_steps["preprocess"]
flooring_options = list(pre.named_transformers_["multi_hot_Flooring"].vocabulary_)
summary = pd.read_csv(os.path.join(predict.DOCS_DIR, "metrics_summary.csv"))
test_mdape = summary[summary["final_model"] & (summary["stage"] == "test") & (summary["segment_type"] == "overall")]["MdAPE"].iloc[0]

st.title("California Home Value Estimator")
st.caption(f"{valuer.model_label} model trained on CRMLS single-family sales from {valuer.train_info['train_start']} "
           f"to {valuer.train_info['train_end']}. Typical error on the April 2026 test month: {test_mdape:.1f}% "
           "(median absolute percentage error). An estimate, not an appraisal.")

with st.form("home"):
    st.subheader("Location")
    c1, c2, c3 = st.columns(3)
    zip_code = c1.text_input("ZIP code", value="92618", max_chars=5)
    lat = c2.number_input("Latitude (optional)", value=None, format="%.6f", help="Leave blank to use the ZIP code's center.")
    lon = c3.number_input("Longitude (optional)", value=None, format="%.6f")

    st.subheader("The home")
    c1, c2, c3 = st.columns(3)
    living = c1.number_input("Living area (sq ft)", min_value=0, value=2000, step=50)
    beds = c2.number_input("Bedrooms", min_value=0, max_value=20, value=3)
    baths = c3.number_input("Bathrooms", min_value=0, max_value=20, value=2)
    c1, c2, c3 = st.columns(3)
    year_built = c1.number_input("Year built", min_value=1850, max_value=date.today().year + 1, value=1990)
    lot = c2.number_input("Lot size (sq ft, optional)", min_value=0, value=None, step=500)
    levels = c3.selectbox("Levels", ["Unknown"] + valuer.levels, index=1 + valuer.levels.index("One") if "One" in valuer.levels else 0)
    c1, c2, c3 = st.columns(3)
    stories = c1.number_input("Stories (optional)", min_value=0, max_value=10, value=None)
    main_beds = c2.number_input("Bedrooms on main level (optional)", min_value=0, max_value=20, value=None)
    flooring = c3.multiselect("Flooring", flooring_options)

    st.subheader("Parking and amenities")
    c1, c2, c3 = st.columns(3)
    garage = c1.number_input("Garage spaces (optional)", min_value=0, max_value=20, value=None)
    parking = c2.number_input("Total parking spaces (optional)", min_value=0, max_value=50, value=None)
    attached = c3.selectbox("Attached garage", list(YES_NO))
    c1, c2, c3 = st.columns(3)
    pool = c1.selectbox("Private pool", list(YES_NO))
    view = c2.selectbox("View", list(YES_NO))
    fireplace = c3.selectbox("Fireplace", list(YES_NO))
    c1, c2, c3 = st.columns(3)
    basement = c1.selectbox("Basement", list(YES_NO))
    waterfront = c2.selectbox("Waterfront", list(YES_NO))
    new_build = c3.selectbox("New construction", list(YES_NO))

    st.subheader("HOA")
    c1, c2 = st.columns(2)
    hoa = c1.number_input("HOA fee (optional, $)", min_value=0, value=None, step=25)
    hoa_freq = c2.selectbox("Paid", ["Monthly", "Quarterly", "SemiAnnually", "Annually"])

    valuation_date = st.date_input("Valuation date", value=date.today(),
                                   help="Sets the season; the price level is the latest in the training data.")
    submitted = st.form_submit_button("Estimate value", type="primary")

if submitted:
    home = {
        "PostalCode": zip_code, "Latitude": lat, "Longitude": lon,
        "LivingArea": living, "LotSizeSquareFeet": lot, "BedroomsTotal": beds, "BathroomsTotalInteger": baths,
        "YearBuilt": year_built, "GarageSpaces": garage, "ParkingTotal": parking, "Stories": stories,
        "MainLevelBedrooms": main_beds, "AssociationFee": hoa, "AssociationFeeFrequency": hoa_freq if hoa else None,
        "Levels": None if levels == "Unknown" else levels, "Flooring": flooring or None,
        "ViewYN": YES_NO[view], "PoolPrivateYN": YES_NO[pool], "NewConstructionYN": YES_NO[new_build],
        "FireplaceYN": YES_NO[fireplace], "AttachedGarageYN": YES_NO[attached], "BasementYN": YES_NO[basement],
        "WaterfrontYN": YES_NO[waterfront],
    }
    result = valuer.value(home, valuation_date)

    st.divider()
    lo80, hi80 = result["range_80"]
    lo90, hi90 = result["range_90"]
    st.metric("Estimated value", money(result["price"]))
    st.markdown(f"**80% likely range:** {money(lo80)} – {money(hi80)}  \n"
                f"**90% likely range:** {money(lo90)} – {money(hi90)}")
    cov = valuer.intervals["test_coverage_80"]
    st.caption("Ranges come from the model's errors on March 2026 sales at a similar predicted price; "
               f"in testing on April 2026 sales, the 80% range contained {100 * cov.min():.0f}–{100 * cov.max():.0f}% "
               "of homes in each price band.")
    for w in result["warnings"]:
        st.warning(w)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**School districts** (from coordinates)")
        d = result["districts"]
        if pd.isna(d.get("school_district_k8")):
            st.write("None found")
        else:
            st.write(f"K-8: {d['school_district_k8'].rsplit(' [', 1)[0]}")
            st.write(f"High school: {d['school_district_hs'].rsplit(' [', 1)[0]}")
    with c2:
        if result["filled"]:
            st.markdown("**Filled in from the ZIP code**")
            for k, val in result["filled"].items():
                if k in ("Latitude", "Longitude"):
                    continue
                st.write(f"{k}: {val}")
    point = {"lat": lat if lat is not None else result["filled"].get("Latitude"),
             "lon": lon if lon is not None else result["filled"].get("Longitude")}
    if point["lat"] is not None and point["lon"] is not None:
        st.map(pd.DataFrame([point]), zoom=11, size=60)
