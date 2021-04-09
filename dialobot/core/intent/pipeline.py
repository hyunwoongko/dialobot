# Copyright (c) 2021, Dialobot. All rights reserved.
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

import os
from typing import List, Union, Dict, Any, Tuple

from dialobot.core.base import IntentBase
from dialobot.core.utils.const import MODEL_ALIAS
from dialobot.core.intent.classifier import IntentClassifier
from dialobot.core.intent.retriever import IntentRetriever


class Intent(IntentBase):

    def __init__(
            self,
            model: str,
            lang: str,
            device: str = "cuda",
            clf_fallback_threshold: float = 0.7,
            rtv_fallback_threshold: float = 0.7,
            rtv_model: str = "distiluse-base-multilingual-cased-v2",
            rtv_dim: int = 512,
            idx_path: str = os.path.join(
                os.path.expanduser('~'),
                ".dialobot",
                "intent/",
            ),
            idx_file: str = "intent.idx",
            dataset_file: str = "dataset.pkl",
            topk: int = 5,
    ):

        model = model.lower()
        if model not in self.availabel_models():
            model = MODEL_ALIAS[model]

        assert model in self.availabel_models(), \
            "currently we support only Classifier, Retriever, Both Model. \n"\
            "So, param `model` must be one of ['clf', 'rtv', 'both']"

        self.model = model

        if model == "clf":
            self.clf = IntentClassifier(lang=lang, fallback_threshold=clf_fallback_threshold)
            self.rtv = None

        elif model == "rtv":
            self.clf = None
            self.rtv = IntentRetriever(
                model=rtv_model,
                dim=rtv_dim,
                idx_path=idx_path,
                idx_file=idx_file,
                dataset_file=dataset_file,
                topk=topk,
                fallback_threshold=rtv_fallback_threshold
            )

        elif model == "both":
            self.clf = IntentClassifier(lang=lang, fallback_threshold=clf_fallback_threshold)
            self.rtv = IntentRetriever(
                model=rtv_model,
                dim=rtv_dim,
                idx_path=idx_path,
                idx_file=idx_file,
                dataset_file=dataset_file,
                topk=topk,
                fallback_threshold=rtv_fallback_threshold
            )
        else:
            raise Exception(f"wrong models: {model}")

        self.device = device

    @staticmethod
    def availabel_models():
        return ["clf", "rtv", "both"]

    def add(self, data: Union[Tuple[str, str], List[Tuple[str, str]]], exist_ok=True) -> None:

        assert self.model not in ["clf"], f"You do not need to remove data in classifier models."

        return self.rtv.add(data=data, exist_ok=exist_ok)

    def remove(self, data: Tuple[str, str]) -> None:

        assert self.model not in ["clf"], f"You do not need to remove data in classifier models."

        return self.rtv.remove(data)

    def clear(self) -> None:

        assert self.model not in ["clf"], f"You do not need to remove data in classifier models."

        return self.rtv.clear()

    def recognize(
        self,
        text: str,
        detail: bool = False,
        intents: List[str] = None,
        voting: str = "soft",
    ) -> Union[str, Dict[str, Any]]:

        assert self.model not in ["clf"] or intents is not None, \
            "In classifier model, you must put intents(List[str])."

        if self.model == "clf":
            return self.clf.recognize(text=text, intents=intents, detail=detail)

        elif self.model == "rtv":
            return self.rtv.recognize(text=text, detail=detail, voting=voting)

        elif self.model == 'both':
            rtv_intents = self.rtv.intents()
            if intents is None:
                clf_out = self.clf.recognize(text=text, intents=rtv_intents, detail=detail)
                rtv_out = self.rtv.recognize(text=text, detail=detail, voting=voting)
            else:
                for input_intent in intents:
                    assert input_intent in rtv_intents, \
                        "`{}` is an intent that has not been trained in the retriever model.".format(input_intent)
                clf_out = self.clf.recognize(text=text, intents=intents, detail=detail)
                rtv_out = self.rtv.recognize(text=text, detail=detail, voting=voting)
            intent = 'fallback' if clf_out['intent'] != rtv_out['intent'] else clf_out['intent']
            if not detail:
                return intent

            return {
                "intent": intent,
                "classifier": clf_out["logits"],
                "retriever": rtv_out["distances"],
            }
