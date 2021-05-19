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

from typing import List, Optional, Union, Tuple
from tokenizers import Tokenizer, decoders, pre_tokenizers, AddedToken
from tokenizers.implementations import BaseTokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
import torch


class BrainBertTokenizer(BaseTokenizer):

    def __init__(
        self,
        vocab: Union[str, List],
        merges: List[Tuple[str, str]],
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        sep_token: str = "</s>",
        cls_token: str = "<s>",
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        replacement: str = "▁",
        add_prefix_space: bool = True,
        dropout: Optional[float] = None,
        normalize: bool = True,
    ):
        bpe = BPE(
            vocab=vocab,
            merges=merges,
            unk_token=unk_token,
            fuse_unk=True,
        )

        tokenizer = Tokenizer(bpe)

        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
            replacement=replacement,
            add_prefix_space=add_prefix_space,
        )

        tokenizer.decoder = decoders.Metaspace(
            replacement=replacement,
            add_prefix_space=add_prefix_space,
        )

        if normalize:
            tokenizer.normalizer = NFKC()

        parameters = {
            "model": "SentencePieceBPE",
            "unk_token": unk_token,
            "replacement": replacement,
            "add_prefix_space": add_prefix_space,
            "dropout": dropout,
        }

        super().__init__(tokenizer, parameters)
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False)
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False)
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False)
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False)
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False)
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False)

        self.add_special_tokens([
            bos_token,
            eos_token,
            sep_token,
            cls_token,
            unk_token,
            pad_token,
        ])

    @staticmethod
    def from_file(
        vocab_filename: str,
        merges_filename: Union[str, None],
        **kwargs,
    ):
        vocab, merges = BPE.read_file(vocab_filename, merges_filename)
        return BrainBertTokenizer(vocab, merges, **kwargs)

    def __call__(
        self,
        text: str,
        return_tensors: bool = True,
        add_special_tokens: str = "pt",
    ) -> Union[List[int], torch.Tensor]:
        """
        encode text for brainbert.

        Args:
            text (str): input sentence
            return_tensors (str): whether convert list of int to `torch.Tensor` or not
            add_special_tokens (bool): whether add <s>, </s> to encoding or not

        Returns:
            (List[str]): list of token ids
            (torch.Tensor): tensor of token ids
        """

        if add_special_tokens:
            text = f"<s>{text}</s>"

        input_ids = self.encode(text).ids

        if return_tensors == "pt":
            input_ids = torch.tensor(input_ids).unsqueeze(0).long()

        return input_ids
