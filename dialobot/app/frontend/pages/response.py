import streamlit as st


def page():
    st.title('Response Generation')
    st.markdown("<br>", unsafe_allow_html=True)
    st.write(
        "Most of the people who use chatbot builder make handcrafted answer using result of the API. "
        "However, this reduces the diversity of the answers and makes conversations that are less lively. "
        "Therefore, Dialobot supports a Response Generation feature that generates lively answers using the result of the API. "
        "(Note: This feature may be incomplete because it is beta version)"
    )
