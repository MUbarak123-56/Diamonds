import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st 
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Lasso
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline


df = pd.read_csv("diamonds.csv")
ord2 = OrdinalEncoder(categories = [["Fair", "Good", "Very Good", "Premium", "Ideal"],
                                    ['D', 'E', 'F', 'G', 'H', 'I', 'J'],
                                    ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']])
std = StandardScaler()
##ord_new = ord2.fit_transform(df[["cut", "color", "clarity"]])
##std_new = std.fit_transform(df[["carat","x", "y", "z"]])
column_pipeline = ColumnTransformer([("ord", ord2, ["cut", "color", "clarity"]), ("std", std, ["carat", "x", "y", "z"])])
##full_pipeline = Pipeline([("col", column_pipeline),("rid", RFECV(Ridge(alpha = 1.0), cv = 5)) ,("reg", RandomForestRegressor(n_estimators=46, max_features=4))])
full_pipeline = make_pipeline(column_pipeline, RFECV(Ridge(alpha = 1.0), cv = 5), RandomForestRegressor(max_features=4, n_estimators=46))
#X = df.drop(["depth", "table", "price"], axis = 1)
X = df[["carat","cut", "color", "clarity", "x", "y", "z"]]
y = df["price"]
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 42069)
full_pipeline.fit(x_train, y_train)

def run():
    from PIL import Image
    image = Image.open("diamonds.jpg")
    st.image(image, caption = "Diamonds", use_column_width = True)
    st.title("Diamond Pricing Project")

    st.header("Visualizing diamond properties")
    st.subheader("Diamond prices' distribution")
    fig, ax = plt.subplots(figsize = (16,12))
    sns.kdeplot(x = "price", data = df, fill = False, hue = "cut", palette = "viridis")
    ax.set_title("Distribution of prices based on different cuts", fontsize = 18)
    ax.set_xlabel("Price", fontsize = 12)
    ax.set_ylabel("Density", fontsize = 12)
    st.pyplot(fig)
    st.subheader("Diamond Carat distribution")
    fig2, ax2 = plt.subplots(figsize = (16,12))
    sns.kdeplot(x = 'carat', data = df, fill = False, hue = "cut", palette = "deep")
    ax2.set_title("Distribution of carats based on different cuts", fontsize = 18)
    ax2.set_xlabel("Carat", fontsize = 12)
    ax2.set_ylabel("Density", fontsize = 12)
    st.pyplot(fig2)

    st.header("Diamond Price Predictor")
    carat = st.number_input("Carat", min_value = 0.0, value = 0.0, step =0.01)
    st.write("Options for cut are: Fair, Good, Very Good, Premium, Ideal. From left to right, Fair is worst and Ideal is best.")
    cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
    st.write("Options for color are: D, E, F, G, H, I, J. From left to right, D is worst and J is best.")
    color = st.selectbox("Color", ['D', 'E', 'F', 'G', 'H', 'I', 'J'])
    st.write("Options for clarity are: I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF. From left to right, I1 is worst and IF is best.")
    clarity = st.selectbox("Clarity", ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
    x = st.number_input("x", min_value = 0.0, value = 0.0, step = 0.01)
    y = st.number_input("y", min_value = 0.0, value = 0.0, step = 0.01)
    z = st.number_input("z", min_value = 0.0, value = 0.0, step = 0.01)
    new_df = pd.DataFrame({"carat":[carat], "cut": [cut], "color": [color], "clarity": [clarity], "x": [x], "y": [y], "z": [z]})
    new_price = 0

    if st.button("Estimate price"):
        new_price = full_pipeline.predict(new_df)
        st.success("$%.2f"%(float(new_price)))
        st.write("The price of this diamond is: $%2.f"%(float(new_price)))



if __name__ == "__main__":
    run()