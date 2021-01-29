import pandas as pd 
import numpy as np 
import matplotlib as plt
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


df = pd.read_csv("diamonds.csv")
ord2 = OrdinalEncoder(categories = [["Fair", "Good", "Very Good", "Premium", "Ideal"],
                                    ['D', 'E', 'F', 'G', 'H', 'I', 'J'],
                                    ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']])
std = StandardScaler()
column_pipeline = ColumnTransformer([("ord", ord2, ["cut", "color", "clarity"]), ("std", std, ["carat", "x", "y", "z"])])
full_pipeline = Pipeline([("col", column_pipeline),("rid", RFECV(Ridge(alpha = 1.0), cv = 5)) ,("reg", RandomForestRegressor(n_estimators=46, max_features=4))])
X = df.drop(["depth", "table", "price"], axis = 1)
y = df["price"]
x_train, y_train, x_test, y_test = train_test_split(X, y, train_size = 0.8, random_state = 42069)


def run():
    from PIL import Image
    image = Image.open("diamonds.jpg")
    st.image(image, caption = "Diamonds", use_column_width = True)
    st.title("Diamond Pricing Project")
    








if __name__ == "__main__":
    run()