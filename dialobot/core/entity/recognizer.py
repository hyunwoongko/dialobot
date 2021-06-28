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

import os
import contextlib
import nltk
from nltk.tokenize import word_tokenize
import torch
from torch.nn import functional as F
from pynori.korean_analyzer import KoreanAnalyzer

from typing import Union, Dict, Any, List
from dialobot.core.base import NerBase
from dialobot.core.utils import LANGUAGE_ALIAS, BrainBertTokenizer
from transformers import RobertaTokenizer, RobertaForSequenceClassification


class Ner(NerBase):

    def __init__(self, lang: str, merge=True, device="cpu") -> None:
        lang = lang.lower()

        if lang not in self.available_languages():
            lang = LANGUAGE_ALIAS[lang]

        assert lang in self.available_languages(), \
            "We support only English, Korean.\n" \
            "So, param `lang` must be one of ['en', 'ko']"

        if lang == "en":
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')
            self.model_name = "hyunwoongko/roberta-base-en-mnli"
            self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)

        elif lang == "ko":
            root_path = os.path.expanduser("~")
            dir_path = os.path.join(root_path, ".dialobot", "entity")
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

        self.lang = lang
        self.merge = merge
        self.device = device
        self.model = RobertaForSequenceClassification.from_pretrained(self.model_name).to(self.device)

    @staticmethod
    def available_languages():
        return ["en", "ko"]

    @staticmethod
    def hypothesises(lang: str, noun: str, entity: str):
        templates = {
            "en": f"This word is {noun}.</s></s>this word is related with {entity}",
            "ko": f"이 단어는 {noun}이다.</s></s>이 단어는 {entity}와 관련이 있다.",
        }
        return templates[lang]

    def recognize(self, text: str, entities: List, threshold: float =0.825) -> Union[str, List[str], float]:
        if self.lang == "en":
            tokens = word_tokenize(text)
            nouns = [w for w, p in nltk.pos_tag(tokens) if "NN" in p]
            entities += [e.lower() for e in entities]
            entities += [e.capitalize() for e in entities]
            entities = list(set(entities))

        elif self.lang == "ko":
            nori = KoreanAnalyzer(decompound_mode='None',
                                  infl_decompound_mode='DISCARD',
                                  discard_punctuation=True,
                                  output_unknown_unigrams=False,
                                  pos_filter=False,
                                  synonym_filter=False)
            nori_tokenize = nori.do_analysis(text)
            tokens = nori_tokenize['termAtt']
            nouns = [w for w, p in zip(nori_tokenize['termAtt'], nori_tokenize['posTagAtt']) if "NN" in p]

        ner_output, ner_dict = [], {}
        for n in nouns:
            scores = []
            for e in entities:
                text = self.hypothesises(lang=self.lang, noun=n, entity=e)
                nli_input = self.tokenizer(text, return_tensors="pt")
                if self.lang == "ko":
                    attention_mask = torch.tensor([1 for _ in range(len(nli_input[0]))]).unsqueeze(0).long()
                    nli_input = {'input_ids': nli_input, 'attention_mask': attention_mask}
                output = self.model(**nli_input).logits[0]
                output = torch.softmax(output, dim=0)[1]
                scores.append(output.unsqueeze(0))

            output = F.softmax(torch.cat(scores, dim=0))
            argmax = output.argmax(-1)
            result = entities[argmax]
            ner_dict[n] = (result, scores[argmax])

        for token in tokens:
            if token in ner_dict:
                score = round(ner_dict[token][1].item(), 3)
                name = ner_dict[token][0].upper()
                if score >= threshold:
                    entity = (token, (name, score))
                else:
                    entity = (token, "O")
            else:
                entity = (token, "O")
            ner_output.append(entity)

        if self.merge:
            merged_output = []
            prev_w, (prev_e, prev_s) = "", ("O", 0.0)
            merge_count = 0

            for i, output in enumerate(ner_output):
                if output[1] == "O":
                    if merge_count != 0:
                        prev_s /= (merge_count + 1)
                        merged_output.append((prev_w, (prev_e, prev_s)))

                    merged_output.append(output)
                    prev_w = output[0]
                    prev_e = "O"
                    merge_count = 0

                else:
                    if output[1][0] == prev_e:
                        if merge_count == 0:
                            merged_output.pop()

                        merge_count += 1
                        prev_w += " " + output[0]
                        perv_e = output[1][0]
                        prev_s += output[1][1]

                    if output[1][0] != prev_e or i == len(ner_output) - 1:
                        if merge_count != 0:
                            prev_s /= (merge_count + 1)
                            merged_output.append((prev_w, (prev_e, prev_s)))

                        else:
                            merged_output.append(output)

                        prev_w, (prev_e, prev_s) = output
                        merge_count = 0

            return merged_output

        return ner_output

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
