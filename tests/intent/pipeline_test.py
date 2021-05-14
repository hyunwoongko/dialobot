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
from dialobot.core.intent.pipeline import Intent


class PipelineTest(unittest.TestCase):

    def test_classifier(self):
        intent = Intent(model="classifier", lang="ko")
        out = intent.recognize(text="날씨 알려줘", intents=["날씨", "식당"])
        self.assertTrue(out == "날씨")

    def test_retriever(self):
        intent = Intent(model="retriever", lang="en")
        intent.clear()
        intent.add(("What time is it now?", "time"))
        intent.add(("Tell me today's weather", "weather"))
        intent.add(("Tell me good restaurant.", "restaurant"))
        intent.remove(("Tell me good restaurant.", "restaurant"))
        out = intent.recognize("Tell me tomorrow's weather")
        self.assertTrue(out == "weather")

    def test_both(self):
        intent = Intent(model="both", lang="en")
        intent.clear()
        intent.add(("They do really good food at that restaurant and it's not very expensive either.", "restaurant"))
        intent.add(("Tell me today's weather", "weather"))
        intent.add([("How will the weather be tomorrow?", "weather"),
                    ("A lot of new restaurants have started up in the region.", "restaurant")])
        out = intent.recognize("Tell me today's weather", intents=["weather", "restaurant"])
        self.assertTrue(out == "weather")
