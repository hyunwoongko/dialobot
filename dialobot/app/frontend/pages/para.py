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
    st.title('Paraphrase Generation')
    st.markdown("<br>", unsafe_allow_html=True)
    st.write(
        "Adding sentences to analyze user's intention is a very laborious task. "
        "Therefore, Dialobot supports the Parapharse Generation feature which automatically generates similar sentences. "
        "With just a few sentences, you can add dozens of similar sentences to your desired intent. "
        "However, this feature is only applied to intents with more than 10 sentences registered."
    )
