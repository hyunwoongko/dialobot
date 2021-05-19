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
    st.title('Entity Recognition')
    st.markdown("<br>", unsafe_allow_html=True)
    st.write(
        "Entity recognition is a feature that extracts the entity in the user's utterance. "
        "Dialobot can predict the entity in the user's utterances with just a few examples without training. "
        "You can edit by clicking name of entity, "
        "and if you want to add a new entity, click the 'Add Entity' button."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("***")

    col1, col2 = st.beta_columns(2)
    with col1:
        st.markdown("<h4 style='text-align: center'>Entity Name</h4>",
                    unsafe_allow_html=True)
    with col2:
        st.markdown("<h4 style='text-align: center'>Registered Words</h4>",
                    unsafe_allow_html=True)

    st.markdown("***")

    col1, col2 = st.beta_columns(2)
    with col1:
        st.button("Pizza")
        st.button("Location")

    with col2:
        st.markdown("<p style='text-align: center; padding: 0.25rem 0.25rem;'> 25 </p>",
                    unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; padding: 0.25rem 0.25rem;'> 104 </p>",
                    unsafe_allow_html=True)

    st.markdown("***")
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("Add Entity")
    st.markdown("<br>", unsafe_allow_html=True)
