import streamlit as st
from dialobot.app.frontend.static.css import style


def css():
    st.markdown(
        f'<style>{style}</style>',
        unsafe_allow_html=True,
    )


def remote_css(url):
    st.markdown(
        f'<link href="{url}" rel="stylesheet">',
        unsafe_allow_html=True,
    )


def icon(icon_name):
    st.markdown(
        f'<i class="material-icons">{icon_name}</i>',
        unsafe_allow_html=True,
    )


def paginate(page):
    pass
