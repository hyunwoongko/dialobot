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
import contextlib
import torch

from typing import Union, Dict, Any, List
from dialobot.modules.base import IntentBase
from dialobot.modules.utils.const import LANGUAGE_ALIAS
from dialobot.modules.utils.tokenizer import BrainBertTokenizer
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    BertTokenizer,
    BertJapaneseTokenizer,
)


class IntentClassifier(IntentBase):

    def __init__(self, lang: str) -> None:
        """
        Zero-shot intent classifier using RoBERTa models.

        Args:
            lang (str): language
        """

        if lang not in self.available_languages():
            lang = LANGUAGE_ALIAS[lang.lower()]

        self.lang = lang
        assert lang in self.available_languages(), \
            "currently we support only English, Korean, Japanese, Chinese.\n" \
            " So, param `lang` must be one of ['en', 'ko', 'ja', 'zh']"

        if lang == "en":
            self.model_name = "hyunwoongko/roberta-base-en-mnli"
            self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)

        elif lang == "ja":
            self.model_name = "hyunwoongko/jaberta-base-ja-xnli"
            self.tokenizer = BertJapaneseTokenizer.from_pretrained(
                self.model_name,
                unk_token="<unk>",
                cls_token="<s>",
                sep_token="</s>",
                mask_token="<mask>",
                pad_token="<pad>",
            )

        elif lang == "zh":
            self.model_name = "hyunwoongko/zhberta-base-zh-xnli"
            self.tokenizer = BertTokenizer.from_pretrained(
                self.model_name,
                unk_token="<unk>",
                cls_token="<s>",
                sep_token="</s>",
                mask_token="<mask>",
                pad_token="<pad>",
            )

        elif lang == "ko":
            root_path = os.path.expanduser("~")
            dir_path = os.path.join(root_path, ".dialobot", "intent")
            vocab_json = os.path.join(dir_path, "brainbert.vocab.json")
            merges_txt = os.path.join(dir_path, "brainbert.merges.txt")
            self.model_name = "hyunwoongko/brainbert-base-ko-kornli"

            if not os.path.exists(vocab_json) or not os.path.exists(merges_txt):
                # download vocab.json and merges.txt for using huggingface tokenizers
                self.download_brainbert_tokenizer(
                    dir_path=dir_path,
                    vocab_json=vocab_json,
                    merges_txt=merges_txt,
                )

            self.tokenizer = BrainBertTokenizer.from_file(
                vocab_filename=vocab_json,
                merges_filename=merges_txt,
            )

        else:
            raise Exception(f"wrong language: {lang}")

        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_name)

    @staticmethod
    def available_languages():
        return ["en", "ko", "ja", "zh"]

    @staticmethod
    def hypothesises(lang: str, intent: str):
        templates = {
            "ko": f"이 문장은 {intent}에 관한 것이다.",
            "ja": f"この文は、{intent}に関するものである。",
            "zh": f"这句话是关于{intent}的。",
            "en": f"This sentence is about {intent}.",
        }

        return templates[lang]

    def recognize(
        self,
        text: str,
        intents: List[str],
        detail: bool = False,
    ) -> Union[str, Dict[str, Any]]:

        results = []
        for intent in intents:
            hypothesis = self.hypothesises(self.lang, intent)
            text = f"{text}</s></s>{hypothesis}"
            tokens = self.tokenizer(text, return_tensors="pt")

            if self.lang != "ko":
                tokens = tokens["input_ids"]

            output = self.model(tokens).logits[0]
            output = torch.softmax(output, dim=0)
            results.append(output[self.labels()])

        f = lambda i: results[i]
        argmax = max(range(len(results)), key=f)

        if not detail:
            return intents[argmax]

        return {
            "intent": intents[argmax],
            "logits": {k: round(v.item(), 5) for k, v in zip(intents, results)}
        }

    def labels(self):
        if "xnli" in self.model_name:
            return 0
        elif "mnli" in self.model_name:
            return 1
        elif "kornli" in self.model_name:
            return 2

    def download_brainbert_tokenizer(
        self,
        dir_path: str,
        vocab_json: str,
        merges_txt: str,
    ) -> None:

        import requests
        import json

        os.makedirs(dir_path, exist_ok=True)
        with contextlib.suppress(FileNotFoundError):
            os.remove(vocab_json)
            os.remove(merges_txt)

        url = f"https://huggingface.co/{self.model_name}/raw/main/tokenizer.json"
        tokenizer = requests.get(url).json()["model"]
        json.dump(tokenizer["vocab"], open(vocab_json, "w"), ensure_ascii=False)

        merges_txt_fp = open(merges_txt, "w")
        for line in tokenizer["merges"]:
            merges_txt_fp.write(line + "\n")
