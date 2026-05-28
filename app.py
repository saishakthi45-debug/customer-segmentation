import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from kneed import KneeLocator

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Customer Insight Studio", layout="wide")
st.title("📊 Customer Insight & Segmentation Studio")
st.subheader("Turn customer data into meaningful business insights!")

file = st.file_uploader("Upload CSV", type=["csv"])

# ---------------- PREPROCESS FUNCTION ----------------
def preprocess(df):
    df = df.copy()
    encoders = {}

    for col in df.columns:
        if df[col].dtype == "object":
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
            encoders[col] = encoder

    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    return df_scaled, scaler, encoders

# ---------------- ELBOW METHOD ----------------
def find_k(data):
    inertia = []
    k_range = range(2, 10)

    try:
        for k in k_range:
            model = KMeans(n_clusters=k, n_init=10, random_state=42)
            model.fit(data)
            inertia.append(model.inertia_)

    except ValueError as e:
        st.error("⚠️ Error in Elbow Method: Non-numeric data detected. Please ensure preprocessing is applied correctly.")
        return 3  # fallback safe value

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(k_range, inertia, marker='o')
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method")
    st.pyplot(fig)

    kl = KneeLocator(k_range, inertia, curve="convex", direction="decreasing")
    return kl.elbow if kl.elbow else 3

# ---------------- MAIN ----------------
if file is not None:
    df = pd.read_csv(file)

    # -------- PREVIEW --------
    st.subheader("Dataset Preview")
    st.write(df.head())

    # -------- DATA ANALYSIS --------
    st.subheader("📊 Data Analysis")

    st.write("### Summary Statistics")
    st.write(df.describe())

    st.write("### Missing Values")
    st.write(df.isnull().sum())

    numeric_df = df.select_dtypes(include="number")

    if not numeric_df.empty:
        st.write("### Correlation")
        st.write(numeric_df.corr())

        st.write("### Distributions")
        for col in numeric_df.columns[:2]:
            # Skip ID-like or high-unique columns
            try:
                if numeric_df[col].nunique() > 0.9 * len(numeric_df):
                    st.warning(f"Skipping {col} (likely ID or unique values, not meaningful for distribution)")
                    continue

                st.write(col)
                st.bar_chart(numeric_df[col].value_counts().sort_index())

            except Exception as e:
                st.error(f"Error plotting {col}: {e}")

    # -------- FEATURE SELECTION --------
    default_cols = [col for col in ["Age", "Annual Income"] if col in df.columns]

    features = st.multiselect(
        "Select Features",
        df.columns,
        default=default_cols
    )

    if len(features) < 2: 
        st.error("Select at least 2 features") 
        st.stop() 
    df_selected = df[features]

    # -------- PREPROCESS --------
    df_scaled, scaler, encoders = preprocess(df_selected)

    # -------- FIND K --------
    st.subheader("Find Optimal Clusters")
    k = find_k(df_scaled)
    st.write(f"Optimal K: {k}")

    # -------- CLUSTERING --------
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    clusters = model.fit_predict(df_scaled)

    df_result = df_selected.copy()
    df_result["Cluster"] = clusters

    # -------- OUTPUT --------
    st.subheader("Clustered Data")
    st.write(df_result.head())

    # -------- VISUALIZATION --------
    st.subheader("Cluster Visualization")

    num_cols = df_result.select_dtypes(include="number").columns

    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("X-axis", num_cols)
    with col2:
        y_axis = st.selectbox("Y-axis", num_cols, index=1)

    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.scatter(df_result[x_axis], df_result[y_axis], c=df_result["Cluster"])
    ax2.set_xlabel(x_axis)
    ax2.set_ylabel(y_axis)
    ax2.set_title("Customer Segments")
    st.pyplot(fig2)

    # -------- CLUSTER INSIGHTS --------
    st.subheader("Cluster Insights")
    cluster_means = df_result.groupby("Cluster").mean(numeric_only=True)
    st.write(cluster_means)

    # -------- INTERPRETATION --------
    st.subheader("Customer Profiles")

    for i in cluster_means.index:
        st.write(f"### Segment {i}")

        row = cluster_means.loc[i]
        insights = []

        for col in row.index:
            if row[col] > df_selected[col].mean():
                insights.append(f"High {col}")
            else:
                insights.append(f"Low {col}")

        st.write(", ".join(insights))

    # -------- PREDICTION --------
    
    st.subheader("🔮 Predict Cluster for New Customer")

    input_data = {}

    for col in df_selected.columns:

        # If numeric → number input
        if df_selected[col].dtype in ["int64"]:
            value = st.number_input(
            f"Enter {col}",
            min_value=int(df_selected[col].min()),
            max_value=int(df_selected[col].max()),
            value=int(df_selected[col].median())
        )

        # If categorical → dropdown
        else:
            options = df[col].unique()
            value = st.selectbox(f"Select {col}", options)

        input_data[col] = value

    if st.button("Predict Cluster"):
        new_df = pd.DataFrame([input_data])

        # Apply SAME encoding as training
        for col in new_df.columns:
            if col in encoders:
                try:
                    new_df[col] = encoders[col].transform(new_df[col])
                except ValueError:
                    st.error(f"Unknown category in {col}. Please select a valid option.")
                    st.stop()

        # Scale
        new_scaled = scaler.transform(new_df)

        prediction = model.predict(new_scaled)[0]

        st.success(f"Predicted Cluster: {prediction}")

else:
    st.info("Upload a dataset to begin")