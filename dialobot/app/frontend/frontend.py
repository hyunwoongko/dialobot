from dialobot.app.frontend.pages import intent, entity, qa, para, response
from dialobot.app.frontend.utils import css
import streamlit as st


class Frontend:

    def __init__(self):
        css()
        self.pages = {
            "Intent Classification": intent,
            "Entity Recognition": entity,
            "Question Answering": qa,
            "Paraphrase Generation": para,
            "Response Generation": response,
        }
        self.build_sidebar()

    def build_sidebar(self):
        # add sidebar image
        st.sidebar.image(
            "https://user-images.githubusercontent.com/38183241/118511978-5d537180-b76d-11eb-89bd-055cb9227725.png"
        )

        current_page = "Intent Classification"
        page_names = list(self.pages.keys())
        buttons = [st.sidebar.button(key) for key in page_names]

        for i, button in enumerate(buttons):
            if button:
                current_page = page_names[i]

        self.pages[current_page].page()
