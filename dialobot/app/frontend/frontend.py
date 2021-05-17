import streamlit as st


class Frontend:

    def __init__(self):
        import streamlit as st
        from dialobot.app.frontend.utils import local_css
        local_css("dialobot/app/frontend/static/style.css")

        self.header = st.empty()
        self.description = st.empty()
        self.build_home()
        self.build_sidebar()

    def build_home(self):
        self.header.header(
            "Dialobot: Opensource Multilingual Chatbot Framework")
        self.description.markdown("- Dialobot is blahblah")

    def build_sidebar(self):
        from dialobot.app.frontend.pages.deploy import load_deploy
        from dialobot.app.frontend.pages.entity import load_entity
        from dialobot.app.frontend.pages.intent import load_intent
        from dialobot.app.frontend.pages.qa import load_qa

        st.sidebar.image(
            "https://user-images.githubusercontent.com/38183241/118511978-5d537180-b76d-11eb-89bd-055cb9227725.png"
        )

        st.sidebar.write("")

        if st.sidebar.button("Intent Classification"):
            load_intent(self.header, self.description)
        if st.sidebar.button("Entity Recognition"):
            load_entity(self.header, self.description)
        if st.sidebar.button("Question Answering"):
            load_qa(self.header, self.description)
        if st.sidebar.button("Application Deployment"):
            load_deploy(self.header, self.description)
