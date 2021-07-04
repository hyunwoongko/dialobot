from asian_bart import AsianBartTokenizer, AsianBartForConditionalGeneration
import torch.nn as nn

"""
Examples:
>>>  # 1. create PG Model
>>> pg = Paraphrase(lang="ko")
>>>  # 2. generate 
>>> pg.generate("오늘 날씨 알려줘!")
"""


class Paraphrase(nn.Module):
    """
    model
    """
    def __init__(self, lang: str, device = "cpu") -> None:
        self.model = AsianBartForConditionalGeneration.from_pretrained("model_name").to(device)
        self.tokenizer = AsianBartTokenizer.from_pretrained("hyunwoongko/asian-bart-ecjk")
        if lang == "ko":
            self.lang_code = "ko_KR"
        elif lang == "en":
            self.lang_code = "en_XX"
        elif lang == "zh":
            self.lang_code = "zh_XX"
        else :
            raise NotImplementedError(f"wrong language code : {lang}")

    def generate(self, user_input: str):
        inputs = self.tokenizer.prepare_seq2seq_batch(
            src_texts=user_input, src_langs=self.lang_code
        )
        gen_token = self.model.generate(
            **inputs, forced_bos_token_id=self.tokenizer.lang_code_to_id[self.lang_code]
        )

        print(
            f"result : {self.tokenizer.decode(gen_token[0][2:], skip_special_tokens=True)}"
        )
