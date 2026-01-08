import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Bitcoin Price Prediction", layout="wide")

st.title("📈 Bitcoin Price Prediction using Machine Learning")

df = pd.read_csv("bitcoin.csv")

df["Date"] = pd.to_datetime(df["Date"])
df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month
df["day"] = df["Date"].dt.day
df["is_quarter_end"] = (df["month"] % 3 == 0).astype(int)

df.drop(columns=["Date"], inplace=True)

df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
df.dropna(inplace=True)

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Correlation Heatmap")

heatmap_type = st.radio(
    "Select Heatmap Type",
    ["Binary Correlation (0/1)", "Pearson Correlation (-1 to +1)"]
)

corr = df.drop(columns=["target"]).corr()

fig, ax = plt.subplots(figsize=(12, 6))

if heatmap_type == "Binary Correlation (0/1)":
    binary_corr = (corr.abs() > 0.5).astype(int)
    sns.heatmap(binary_corr, annot=True, cmap="Blues", cbar=False, ax=ax)
else:
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)

st.pyplot(fig)

X = df.drop(columns=["target"])
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

st.subheader("🔮 Predict Next Day Bitcoin Movement")

col1, col2, col3 = st.columns(3)

with col1:
    open_price = st.number_input("Open Price", float(df["Open"].mean()))
    high_price = st.number_input("High Price", float(df["High"].mean()))
    low_price = st.number_input("Low Price", float(df["Low"].mean()))

with col2:
    close_price = st.number_input("Close Price", float(df["Close"].mean()))
    adj_close = st.number_input("Adj Close", float(df["Adj Close"].mean()))
    volume = st.number_input("Volume", float(df["Volume"].mean()))

with col3:
    year = st.number_input("Year", int(df["year"].mean()))
    month = st.number_input("Month", int(df["month"].mean()))
    day = st.number_input("Day", int(df["day"].mean()))
    is_quarter_end = st.selectbox("Is Quarter End", [0, 1])

if st.button("Predict"):
    input_data = np.array([
        open_price,
        high_price,
        low_price,
        close_price,
        adj_close,
        volume,
        year,
        month,
        day,
        is_quarter_end
    ]).reshape(1, -1)

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("📈 Bitcoin price is likely to go UP tomorrow")
    else:
        st.error("📉 Bitcoin price is likely to go DOWN tomorrow")

