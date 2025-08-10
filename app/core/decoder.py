# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------------
# NOTICE: This file has been modified from its original version.

import numpy as np


class Decoder:
    BOS = "[B]"
    EOS = "[E]"
    PAD = "[P]"

    def __init__(self, charset: str):
        specials_first = (self.EOS,)
        specials_last = (self.BOS, self.PAD)

        self._itos = specials_first + tuple(charset) + specials_last
        self._stoi = {s: i for i, s in enumerate(self._itos)}

        self.eos_id, self.bos_id, self.pad_id = [self._stoi[s] for s in specials_first + specials_last]

    def __len__(self):
        return len(self._itos)

    def _tok2ids(self, tokens: str) -> list[int]:
        return [self._stoi[s] for s in tokens]

    def _ids2tok(self, token_ids: list[int], join: bool = True) -> str:
        tokens = [self._itos[i] for i in token_ids]
        return "".join(tokens) if join else tokens

    def _filter(self, probs: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, list[int]]:
        ids = ids.tolist()
        try:
            eos_idx = ids.index(self.eos_id)
        except ValueError:
            eos_idx = len(ids)  # Nothing to truncate.
        # Truncate after EOS
        ids = ids[:eos_idx]
        probs = probs[: eos_idx + 1]  # but include prob. for EOS (if it exists)
        return probs, ids

    def __call__(self, token_dists: np.ndarray, raw: bool = False) -> tuple[list[str], list[np.ndarray]]:
        """Decode a batch of token distributions.

        Args:
            token_dists: softmax probabilities over the token distribution. Shape: N, L, C
            raw: return unprocessed labels (will return list of list of strings)

        Returns:
            list of string labels (arbitrary length) and
            their corresponding sequence probabilities as a list of ndarrays
        """
        batch_tokens = []
        batch_probs = []
        for dist in token_dists:
            probs = np.max(dist, axis=-1)
            ids = np.argmax(dist, axis=-1)

            if not raw:
                probs, ids = self._filter(probs, ids)
            tokens = self._ids2tok(ids, not raw)
            batch_tokens.append(tokens)
            batch_probs.append(probs)
        return batch_tokens, batch_probs
