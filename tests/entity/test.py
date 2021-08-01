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

import unittest
from dialobot.core.entity import Ner


class NERTester(unittest.TestCase):

    def test_korean(self):
        ner = Ner(lang="ko")
        out = ner.recognize("치즈피자 주문해주세요.",
                            entities=["음식", "도시"])
        # entity = [e for e, s in [entity for word, entity in out if len(entity) > 1]]
        # self.assertTrue("음식" in entity)

    def test_english(self):
        ner = Ner(lang="en")
        out = ner.recognize("please order Cheese Pizza.",
                            entities=["FOOD", "CITY"])
        entity = [e for e, s in [entity for word, entity in out if len(entity) > 1]]
        self.assertTrue("FOOD" in entity)

    def test_chinese(self):
        ner = Ner(lang="zh")
        out = ner.recognize("请订购奶酪比萨。",
                            entities=["食物", "城市"])
        # entity = [e for e, s in [entity for word, entity in out if len(entity) > 1]]
        # self.assertTrue("食物" in entity)