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
    st.title('Response Generation')
    st.markdown("<br>", unsafe_allow_html=True)
    st.write(
        "Most of the people who use chatbot builder make handcrafted answer using result of the API. "
        "However, this reduces the diversity of the answers and makes conversations that are less lively. "
        "Therefore, Dialobot supports a Response Generation feature that generates lively answers using the result of the API. "
        "(Note: This feature may be incomplete because it is beta version)")
