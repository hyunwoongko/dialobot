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

LANGUAGE_ALIAS = {
    "english": "en",
    "eng": "en",
    "korean": "ko",
    "kor": "ko",
    "japanese": "ja",
    "jp": "ja",
    "chinese": "zh",
    "cn": "zh",
}

MODEL_ALIAS = {
    "classifier": "clf",
    "retriever": "rtv",
}

RETRIEVER_MODELS_DIMENSION = {
    "distiluse-base-multilingual-cased-v1": 512,
    "distiluse-base-multilingual-cased-v2": 512,
    "paraphrase-multilingual-MiniLM-L12-v2": 384,
    "paraphrase-multilingual-mpnet-base-v2": 768
}