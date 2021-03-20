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

from typing import Union, Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from dialobot.modules.base import IntentBase

import os
import numpy as np
import pickle

try:
    import faiss
except ImportError as e:
    raise ImportError(
        "Can not import `faiss`, please install package using below instructions.\n"
        "- CPU user: `pip install faiss-cpu`\n"
        "- GPU user: `pip install faiss-gpu`\n")


class IntentRetriever(IntentBase):

    def __init__(
        self,
        model: str = "distiluse-base-multilingual-cased-v2",
        dim: int = 512,
        idx_path: str = f"{os.path.expanduser('~')}/.dialobot/intent/",
        idx_file: str = "intent.idx",
    ) -> None:
        """
        IntentRetriever using USE and faiss.
        Dialobot conducts fallback checking through vector retrieval.

        Args:
            model (str): model name for sentence transformers
            dim (int): dimension of vector.
            idx_path (str): path to save dataset
            idx_file (str): file name of dataset

        References:
            Universal Sentence Encoder (Cer et al., 2018)
            https://arxiv.org/abs/1803.11175

            Billion-scale similarity search with GPUs (Johnson et al., 2017)
            https://arxiv.org/abs/1702.08734

        Examples:
            >>> # 1. create retriever and append dataset
            >>> retriever = IntentRetriever()
            >>> retriever.add(("What time is it now?", "time"))
            >>> retriever.add(("Tell me today's weather", "weather"))
            >>> # 2. remove dataset
            >>> retriever.remove(("What time is it now?", "time"))
            >>> # 3. recognize intent by retrieval
            >>> retriever.recognize("Tell me tomorrow's weather")
            'weather'
            >>> # 4. set `True` param `detail` if you want more information
            >>> retriever.recognize("Tell me tomorrow's weather", detail=True)
            {'intent': 'weather', distances: [0.988, 0.693, ...]}
            >>> # 5. clear all dataset
            >>> retriever.clear()
        """

        self.model = SentenceTransformer(model)
        self.index = faiss.IndexFlatL2(dim)
        self.dim = dim
        self.idx_path = idx_path
        self.idx_file = idx_file

        if os.path.exists(idx_path + idx_file):
            with open(idx_path + idx_file, mode="rb") as f:
                self.dataset: List[Tuple[str, np.ndarray, str]] = pickle.load(f)

                if len(self.dataset) != 0:
                    vectors = [
                        vec.reshape(1, -1) for sent, vec, idx in self.dataset
                    ]
                    self.index.add(np.concatenate(vectors, axis=0))
        else:
            os.makedirs(idx_path, exist_ok=True)
            self.dataset: List[Tuple[str, np.ndarray, str]] = []
            # list of (sentence, vector, intent)

    def add(self, data: Tuple[str, str]) -> None:
        """
        Add data to dataset.

        Args:
            data (Tuple[str, str]): tuple of (sentence, intent)

        Examples:
            >>> retriever = IntentRetriever()
            >>> retriever.add(("What time is it now?", "time"))
            >>> retriever.add(("Tell me today's weather", "weather"))

        Raises:
            Raises Exceptoin when you try to add existed data.
        """

        for d in self.dataset:
            if data[0] == d[0] and data[1] == d[2]:
                raise Exception(f"This data is already existed: {data}")

        vector = self._vectorize(data[0])
        data = (data[0], vector, data[1])
        self.index.add(vector)
        self.dataset.append(data)

        with open(self.idx_path + self.idx_file, mode="wb") as f:
            pickle.dump(self.dataset, f, pickle.HIGHEST_PROTOCOL)

    def remove(self, data: Tuple[str, str]) -> None:
        """
        Remove data from dataset.

        Args:
            data (Tuple[str, str]): tuple of (sentence, intent)

        Examples:
            >>> retriever = IntentRetriever()
            >>> retriever.remove(("What time is it now?", "time"))

        Raises:
            Raises Exceptoin when you try to remove non-existed data.

        """

        find = False
        new_dataset = []
        new_vectors = []
        new_index = faiss.IndexFlatL2(self.dim)

        for d in self.dataset:
            if d[0] != data[0] or d[2] != data[1]:
                new_dataset.append(d)
                new_vectors.append(d[1].reshape(1, -1))
            else:
                find = True

        if not find:
            raise Exception(f"This data does not exist: {data}")

        if len(new_vectors) != 0:
            new_index.add(np.concatenate(new_vectors, axis=0))

        self.dataset = new_dataset
        self.index = new_index

        with open(self.idx_path + self.idx_file, mode="wb") as f:
            pickle.dump(self.dataset, f, pickle.HIGHEST_PROTOCOL)

    def clear(self) -> None:
        """
        Clear all dataset.

        Examples:
            >>> retriever = IntentRetriever()
            >>> retriever.clear()

        """

        self.dataset = []
        self.index.reset()

        with open(self.idx_path + self.idx_file, mode="wb") as f:
            pickle.dump(self.dataset, f, pickle.HIGHEST_PROTOCOL)

    def recognize(
        self,
        text: str,
        detail: bool = False,
        topk: int = 5,
    ) -> Union[str, Dict[str, Union[str, float, int]]]:
        """
        Recognize intent by data.

        Args:
            text (str): target string
            detail (bool): whether to return details or not
            topk (int): number of distances to return

        Returns:
            (str): intent of input sentence (detail=False)
            (Dict[str, Union[str, float]]): intent and distances (detail=True)

        Examples:
            >>> retriever = IntentRetriever()
            >>> retriever.recognize("Tell me tomorrow's weather")
            'weather'
            >>> retriever.recognize("Tell me tomorrow's weather", detail=True)
            {'intent': 'weather', distances: [0.988, 0.693, ...]}

        """

        assert self.index.ntotal != 0, \
            "empty index. please add new data using below codes.\n" \
            ">>> retriever = IntentRetriver()\n" \
            ">>> retriever.add((sentence, intent))"

        topk = min(topk, self.index.ntotal)
        vector = self._vectorize(text)
        dists, indices = self.index.search(vector, topk)
        intent = self.dataset[int(indices[0][0])][2]

        if not detail:
            return intent

        return {
            "inetnt": intent,
            "distances": dists[0].tolist(),
        }

    def ntotal(self) -> int:
        """
        Return number of data in dataset

        Returns:
            (int): number of data in dataset

        Examples:
            >>> retriever = IntentRetriever()
            >>> retriever.ntotal()
            20
        """

        return self.index.ntotal

    def __len__(self) -> int:
        """
        Return number of data in dataset

        Returns:
            (int): number of data in dataset

        Examples:
            >>> retriever = IntentRetriever()
            >>> len(retriever)
            20
        """
        return self.ntotal()

    def _vectorize(self, text: str) -> np.ndarray:
        """
        Create vector from text.

        Args:
            text (str): target string

        Returns:
            (np.ndarray): vector from text

        """

        vector = self.model.encode(text)
        vector = np.array(vector, dtype=np.float32)
        return vector.reshape(1, -1)
