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
    st.title('Intent Classification')
    st.markdown("<br>", unsafe_allow_html=True)
    st.write(
        "Intent classification is a feature that analyzes the intention of the user's utterance. "
        "Dialobot can predict the user's intention with just a few examples without training. "
        "You can edit by clicking name of intent, "
        "and if you want to add a new intent, click the 'Add Intent' button.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("***")

    col1, col2, col3 = st.beta_columns(3)
    with col1:
        st.markdown(
            "<h4 style='text-align: center'>Intent Name</h4>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            "<h4 style='text-align: center'>Sentences</h4>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            "<h4 style='text-align: center'>Replies</h4>",
            unsafe_allow_html=True,
        )

    st.markdown("***")

    col1, col2, col3 = st.beta_columns(3)
    with col1:
        st.button("Default")
        st.button("Weather")
        st.button("Restaurant")
        st.button("Time")

    with col2:
        st.markdown(
            "<p style='text-align: center; padding: 0.25rem 0.25rem;'> N/A </p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align: center; padding: 0.25rem 0.25rem;'> 12 </p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align: center; padding: 0.25rem 0.25rem;'> 2 </p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align: center; padding: 0.25rem 0.25rem;'> 16 </p>",
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            "<p style='text-align: center; padding: 0.25rem 0.25rem;'> 4 </p>",
            unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align: center; padding: 0.25rem 0.25rem;'> 5 </p>",
            unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align: center; padding: 0.25rem 0.25rem;'> 12 </p>",
            unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align: center; padding: 0.25rem 0.25rem;'> 10 </p>",
            unsafe_allow_html=True)

    st.markdown("***")
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("Add Intent")
    st.markdown("<br>", unsafe_allow_html=True)
