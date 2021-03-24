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
from dialobot.core.base import IntentBase

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
        idx_path: str = os.path.join(
            os.path.expanduser('~'),
            ".dialobot",
            "intent",
        ),
        idx_file: str = "intent.idx",
        dataset_file: str = "dataset.pkl",
        fallback_threshold: float = 0.7,
        nlist: int = 100,
    ) -> None:
        """
        IntentRetriever using USE and faiss.
        Dialobot conducts fallback checking through vector retrieval.

        Args:
            model (str): model name for sentence transformers
            dim (int): dimension of vector.
            idx_path (str): path to save dataset
            idx_file (str): file name of dataset
            fallback_threshold (float): thershold for fallback checking

        References:
            Universal Sentence Encoder (Cer et al., 2018)
            https://arxiv.org/abs/1803.11175

            Billion-scale similarity search with GPUs (Johnson et al., 2017)
            https://arxiv.org/abs/1702.08734

        Examples:
            >>> # 1. create retriever
            >>> retriever = IntentRetriever()
            >>> # 2. add data
            >>> retriever.add(("What time is it now?", "time"))
            >>> retriever.add(("Tell me today's weather", "weather"))
            >>> # 3. remove data
            >>> retriever.remove(("What time is it now?", "time"))
            >>> # 4. recognize intent
            >>> retriever.recognize("Tell me tomorrow's weather")
            'weather'
            >>> # 5. set `True` param `detail` if you want more information
            >>> retriever.recognize("Tell me tomorrow's weather", detail=True)
            {'intent': 'weather', distances: [(0.988, weather), (0.693, greeting), ...]}
            >>> # 6. clear all dataset
            >>> retriever.clear()
        """

        # todo : nlist, nprobe 기본값 설정
        # nprobe : the number of cells (out of nlist) that are visited to perform a search
        # nlist : the number of cells
        # nlist =100으로 잡고 문장을 추가하면 다음과 같은 에러발생
        # Error: 'nx >= k' failed: Number of training points (1) should be at least as large as number of clusters (100)
        # Solution 1 : nlist 크기만큼 데이터셋을 한번에 학습을 해야함 => default dataset이 필요
        # Solution 2 : 데이터가 100개 이하일 때는 문장이 추가될때마다 문장 개수와 동일한 nlist를 가진 index를 새로 생성,
        #              100개 이후에는 기존의 index에 추가 데이터를 그대로 학습

        self.model = SentenceTransformer(model)
        self.quantizer = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIVFFlat(
            self.quantizer,
            dim,
            nlist,
            faiss.METRIC_INNER_PRODUCT,
        )

        self.dim = dim
        self.nlist = nlist
        self.idx_path = idx_path
        self.idx_file = idx_file
        self.dataset_file = dataset_file
        self.fallback_threshold = fallback_threshold

        if os.path.exists(idx_path + dataset_file):
            with open(idx_path + dataset_file, mode="rb") as f:
                self.dataset: List[Tuple[str, np.ndarray, str]] = pickle.load(f)
        else:
            os.makedirs(idx_path, exist_ok=True)
            self.dataset: List[Tuple[str, np.ndarray, str]] = []
            # list of (sentence, vector, intent)

        if os.path.exists(idx_path + idx_file):
            self.index = faiss.read_index(idx_path + idx_file)

    def add(self, data: Tuple[str, str], exist_ok=True) -> None:
        """
        Add data to dataset.

        Args:
            data (Tuple[str, str]): tuple of (sentence, intent)
            exist_ok (bool): ignore exception when you inputted duplicates data

        Examples:
            >>> retriever = IntentRetriever()
            >>> retriever.add(("What time is it now?", "time"))
            >>> retriever.add(("Tell me today's weather", "weather"))

        Raises:
            Raises exceptoin when you try to add existed data.
        """

        for d in self.dataset:
            if data[0] == d[0] and data[1] == d[2]:
                if exist_ok:
                    return
                else:
                    raise Exception(f"This data is already existed: {data}")

        vector = self._vectorize(data[0])
        data = (data[0], vector, data[1])
        self.index.train(vector)
        assert self.index.is_trained

        self.index.add(vector)
        self.dataset.append(data)

        with open(self.idx_path + self.dataset_file, mode="wb") as f:
            pickle.dump(self.dataset, f, pickle.HIGHEST_PROTOCOL)

        faiss.write_index(self.index, self.idx_path + self.idx_file)

    def remove(self, data: Tuple[str, str]) -> None:
        """
        Remove data from dataset.

        Args:
            data (Tuple[str, str]): tuple of (sentence, intent)

        Examples:
            >>> retriever = IntentRetriever()
            >>> retriever.remove(("What time is it now?", "time"))

        Raises:
            Raises exceptoin when you try to remove non-existed data.
        """

        find = False
        new_dataset = []
        new_vectors = []
        new_index = faiss.IndexIVFFlat(self.quantizer, self.dim, self.nlist,
                                       faiss.METRIC_INNER_PRODUCT)

        for d in self.dataset:
            if d[0] != data[0] or d[2] != data[1]:
                new_dataset.append(d)
                new_vectors.append(d[1].reshape(1, -1))
            else:
                find = True

        if not find:
            raise Exception(f"This data does not exist: {data}")

        if len(new_vectors) != 0:
            new_index.train(np.concatenate(new_vectors, axis=0))
            new_index.add(np.concatenate(new_vectors, axis=0))

        self.dataset = new_dataset
        self.index = new_index

        with open(self.idx_path + self.dataset_file, mode="wb") as f:
            pickle.dump(self.dataset, f, pickle.HIGHEST_PROTOCOL)

        faiss.write_index(self.index, self.idx_path + self.idx_file)

    def clear(self) -> None:
        """
        Clear all dataset.

        Examples:
            >>> retriever = IntentRetriever()
            >>> retriever.clear()
        """

        self.dataset = []
        self.index.reset()

        with open(self.idx_path + self.dataset_file, mode="wb") as f:
            pickle.dump(self.dataset, f, pickle.HIGHEST_PROTOCOL)

        faiss.write_index(self.index, self.idx_path + self.idx_file)

    def recognize(
        self,
        text: str,
        detail: bool = False,
        topk: int = 5,
        voting: str = "soft",
    ) -> Union[str, Dict[str, Union[str, List[Tuple[float, str]]]]]:
        """
        Recognize intent by input sentence.

        Args:
            text (str): input sentence
            detail (bool): whether to return details or not
            topk (int): number of distances to return
            voting (str): voting method for kNN search.
                must be one of ['soft', 'hard'].

        Returns:
            (str): intent of input sentence (detail=False)
            (Dict[str, Union[str, List[Tuple[float, str]]]]): intent and distances (detail=True)

        Examples:
            >>> retriever = IntentRetriever()
            >>> retriever.recognize("Tell me tomorrow's weather")
            'weather'
            >>> retriever.recognize("Tell me tomorrow's weather", detail=True)
            {'intent': 'weather', distances: [(0.988, weather), (0.693, greeting), ...]}
        """

        voting = voting.lower()
        assert voting in ['soft', 'hard'], \
            "param `voting` must be one of ['soft', 'hard']."

        assert self.index.ntotal != 0, \
            "empty index. please add new data using below codes.\n" \
            ">>> retriever = IntentRetriver()\n" \
            ">>> retriever.add((sentence, intent))"

        topk = min(topk, self.index.ntotal)
        vector = self._vectorize(text)
        dists, indices = self.index.search(vector, topk)
        dists, indices = dists[0], indices[0]
        is_fallback = False

        if max(dists) < self.fallback_threshold:
            is_fallback = True

        scores: Dict[str, float] = {}
        for idx, d in zip(indices, dists):
            intent = self.dataset[idx][2]
            if intent not in scores:
                scores[intent] = 0.0
            if voting == "soft":
                scores[intent] += d
            else:
                scores[intent] += 1

        scores: Dict[float, str] = {v: k for k, v in scores.items()}
        intent = scores[sorted(scores, reverse=True)[0]]
        intent = 'fallback' if is_fallback else intent

        if not detail:
            return intent

        return {
            "intent":
                intent,
            "distances": [
                (d, self.dataset[i][2]) for i, d in zip(indices, dists)
            ],
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
        Create vector from input sentence.

        Args:
            text (str): input sentence

        Returns:
            (np.ndarray): vector from input sentence
        """

        vector = self.model.encode(text)
        vector = np.array(vector, dtype=np.float32)
        vector = vector.reshape(1, -1)
        faiss.normalize_L2(vector)

        return vector
