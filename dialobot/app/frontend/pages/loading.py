import streamlit as st


def page():
    st.image(
        "https://user-images.githubusercontent.com/38183241/118511978-5d537180-b76d-11eb-89bd-055cb9227725.png"
    )

    st.markdown("""
        ## **What is Dialobot ?**
        - **Opensource chatbot framework** available for free.
        - **Neural chatbot framework** using the latest models (RoBERTa, DistillUSE, mBART)
        - **Multilingual chatbot framework** that supports English, Korean, Chinese.
        - **Zero-shot chatbot framework** that can be used immediately without training.
        - **Chatbot builder** that supports web application and RESTful API for services.
        """)

    st.markdown(
        """
    ## **License**
    - Dialobot project is licensed under the terms of the Apache License 2.0.
    - Copyright 2021 <a href='https://github.com/hyunwoongko' style='color: black; text-decoration-line: none;'>Hyunwoong Ko</a>. 
    All Rights Reserved.
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<br>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style=\"text-align: center;\">"
        "<br>"
        "<h3>"
        "<a title='' href='javascript:location.reload();' style='color: black; text-decoration-line: none;'>"
        "Click here to start builder application"
        "</a>"
        "</h3>"
        "</div>",
        unsafe_allow_html=True,
    )
