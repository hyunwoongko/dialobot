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
            lang: str,
            model: str = 'both',
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
        """

        Args:
            lang (str): language
            model (str): select model [classifier(clf), retriever(rtv), both]
            device (str): choose 1 between cpu and gpu
            clf_fallback_threshold (str): classifier threshold for fallback checking
            rtv_fallback_threshold (str): retriever threshold for fallback checking
            rtv_model (str): retriever model name for sentence transformers
            rtv_dim (int): retriever dimension of vector
            idx_path (str): path to save retriever dataset
            idx_file (str): file name of trained faiss
            dataset_file (str): file name of retriever dataset
            topk (int): number of distances to return

        Examples:
            >>>> # 1. create classifier
            >>>> intent = Intent(lang="en")
            >>>> # 2. add retriever data and batch data
            >>> intent.add(("They do really good food at that restaurant and it's not very expensive either.", "restaurant"))
            >>> intent.add(("Tell me today's weather", "weather"))
            >>> intent.add([("How will the weather be tomorrow?", "weather"),
                ("A lot of new restaurants have started up in the region.", "restaurant")])
            >>> # 3. remove data
            >>> intent.remove(("Tell me today's weather", "weather"))
            >>> # 4. recognize intent
            >>> intent.recognize("Tell me today's weather", intents=["weather", "restaurant"])
            'weather'
            >>> intent.recognize("Tell me today's weather", intents=["weather", "restaurant"], detail=True)
            {'intent': 'weather',
            'classifier': {'weather': 0.75165, 'restaurant': 0.0004},
            'retriever': [(0.99999976, 'weather'), (0.6623127, 'weather'), ...]}

        """
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
                rtv_out = self.rtv.recognize(text=text, voting=voting, detail=detail)
            else:
                for input_intent in intents:
                    assert input_intent in rtv_intents, \
                        "`{}` is an intent that has not been trained in the retriever model.".format(input_intent)
                clf_out = self.clf.recognize(text=text, intents=intents, detail=detail)
                rtv_out = self.rtv.recognize(text=text, voting=voting, detail=detail)
            if detail:
                intent = 'fallback' if clf_out['intent'] != rtv_out['intent'] else clf_out['intent']
                return {
                    "intent": intent,
                    "classifier": clf_out["logits"],
                    "retriever": rtv_out["distances"],
                }
            else:
                intent = 'fallback' if clf_out != rtv_out else clf_out
                return intent
