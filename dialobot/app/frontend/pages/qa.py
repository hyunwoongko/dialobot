# Copyright (c) 2021, Hyunwoong Ko. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st


def page():
    st.title('Question Answering')
    st.markdown("<br>", unsafe_allow_html=True)
    st.write(
        "It is very inefficient to deal with all questions as Intent-Entity. "
        "Therefore, Dialobot supports Question Answering feature. If you put some documents in advance, "
        "the bot checks the documents and answers automatically without Intent-Entity. "
        "Click on the title of the document to edit an document. "
        "And if you want to add a new document, click the 'Add Document' button."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("***")

    col1, col2 = st.beta_columns(2)
    with col1:
        st.markdown("<h4 style='text-align: center'>Document Title</h4>",
                    unsafe_allow_html=True)
    with col2:
        st.markdown("<h4 style='text-align: center'>Document Contents</h4>",
                    unsafe_allow_html=True)

    st.markdown("***")

    col1, col2 = st.beta_columns(2)
    with col1:
        st.button("Galaxy S20 Manual")

    with col2:
        st.markdown("<p style='text-align: left; padding: 0.25rem 0.25rem;'> It is one of the 2020 models and 11th models of the Galaxy S series, Samsung Android flagship smartphone series ...</p>",
                    unsafe_allow_html=True)

    st.markdown("***")
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("Add Document")
    st.markdown("<br>", unsafe_allow_html=True)
