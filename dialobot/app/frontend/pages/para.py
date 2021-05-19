import streamlit as st


def page():
    st.title('Paraphrase Generation')
    st.markdown("<br>", unsafe_allow_html=True)
    st.write(
        "Adding sentences to analyze user's intention is a very laborious task. "
        "Therefore, Dialobot supports the Parapharse Generation feature which automatically generates a similar sentence. "
        "With just a few sentences, you can add dozens of similar sentences to your desired intent. "
        "However, this feature is only applied to intents with more than 10 sentences registered."
    )
