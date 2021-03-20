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
from dialobot.modules.intent import IntentRetriever


class TestRetriever(unittest.TestCase):

    def test_add(self):
        retriever = IntentRetriever()
        retriever.clear()
        retriever.add(("Tell me tomorrow's weather", "weather"))
        retriever.add(("Tell me today's weather", "weather"))
        retriever.add(("Tell me good restaurant.", "restaurant"))
        retriever.add(("I'm hungry. Tell me restaurants", "restaurant"))

        self.assertTrue(len(retriever) == 4)
        retriever.clear()

    def test_remove(self):
        retriever = IntentRetriever()
        retriever.clear()
        retriever.add(("Tell me today's weather", "weather"))
        retriever.remove(("Tell me today's weather", "weather"))

        self.assertTrue(len(retriever) == 0)
        retriever.clear()

    def test_search(self):
        retriever = IntentRetriever()
        retriever.clear()
        retriever.add(("Tell me today's weather", "weather"))
        retriever.add(("Tell me good restaurant.", "restaurant"))

        cls = retriever.recognize("Tell me great restaurant")
        self.assertTrue(cls == "restaurant")
        retriever.clear()


if __name__ == '__main__':
    testcase = TestRetriever()
    testcase.test_add()
    testcase.test_remove()
    testcase.test_search()
