import streamlit as st
import tfm_models
import tfm_exploratory
import tfm_features


pages = {
    'Explore Data': tfm_exploratory,
    'Explore Models': tfm_models,
    'Explore Features': tfm_features
}

st.sidebar.title('Navegación')
selection = st.sidebar.radio('Ir a', list(pages.keys()))
page = pages[selection]
page.app()
    