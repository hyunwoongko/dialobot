import unittest
from dialobot.modules.intent import IntentClassifier


class ClassifierTester(unittest.TestCase):

    def test_korean(self):
        clf = IntentClassifier(lang="ko")
        out = clf.recognize("날씨 알려줘", intents=["날씨", "식당"])
        self.assertTrue(out == "날씨")

    def test_english(self):
        clf = IntentClassifier(lang="ko")
        out = clf.recognize("Tell me today's weather",
                            intents=["weather", "restaurant"])
        self.assertTrue(out == "weather")

    def test_chinese(self):
        clf = IntentClassifier(lang="zh")
        out = clf.recognize("告诉我天气", intents=["天气", "饭厅"])
        self.assertTrue(out == "天气")

    def test_japanese(self):
        clf = IntentClassifier(lang="ja")
        out = clf.recognize("天気を教えて。", intents=["空合い", "食堂"])
        self.assertTrue(out == "空合い")
