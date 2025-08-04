import streamlit as st
from st_pages import add_page_title, get_nav_from_toml

st.set_page_config(layout="wide")

st.title("Главная страница")
st.write("Выберите приложение в меню слева.")