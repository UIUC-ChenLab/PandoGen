# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import numpy as np
import sqlite3
import random
from utils import SpecialTokens, AMINO_ACIDS, _DEFAULT_SPECIAL_TOKENS, concat_lists
from typing import Union, Callable, Generator, Optional, List, Dict
import logging
from collections.abc import Iterable
from collections import namedtuple, defaultdict
import os
import json
import math
from functools import partial
import torch
from dataclasses import dataclass, asdict
import pickle
import pysam
import tqdm

logger = logging.getLogger(name=__file__)


def split_sequence(
    sequence: str,
    mask_start: int,
    mask_length: int,
    special_tokens: SpecialTokens
) -> tuple:
    """
    Split a sequence into encoder input (unmasked segment) and decoder output (masked segment).
    Note that mask_start is removed from the encoder input.
    """
    left_segment = sequence[:mask_start]
    masked_segment = sequence[mask_start: mask_start + mask_length]
    right_segment = sequence[mask_start + mask_length: ]

    if len(left_segment) + len(masked_segment) + len(right_segment) < len(sequence):
        raise ValueError(
        f"Bad masking arguments: seq={sequence}, mask_start={mask_start}, mask_length={mask_length}")

    assert(left_segment + masked_segment + right_segment == sequence), "Bad split"

    encoder_input_to_be_tokenized = concat_lists(
        [special_tokens.start_of_sequence],
        list(left_segment),
        [special_tokens.masked_segment_indicator],
        list(right_segment),
        [special_tokens.end_of_sequence],
    )

    decoder_output_to_be_tokenized = concat_lists(
        list(masked_segment),
        [special_tokens.masked_segment_indicator]
    )

    return encoder_input_to_be_tokenized, decoder_output_to_be_tokenized


def is_substring_repeating(string: str, subst: str, start: int, stop: int) -> bool:
    length = len(subst)

    for i in range(start, stop):
        if string[i: i + length] == subst:
            return True

    return False


def find_minimal_non_repeating_presuffix(
    string: str,
    start_length: int = 1,
    max_length: int = 21,
    suffix: bool = False,
) -> Union[None, int]:
    """
    Find minimal non-repeating prefix
    """
    length_range = (start_length, max_length + 1)
    current_length = math.ceil(sum(length_range) / 2)
    flag = True
    decisions = defaultdict(lambda: None)

    def repeat_checker(length: int) -> bool:
        if not suffix:
            return is_substring_repeating(
                string, string[:length], 1, len(string) - length + 1
            )
        else:
            return is_substring_repeating(
                string, string[-length: ], 0, len(string) - length
            )

    if repeat_checker(max_length):
        return None

    loop_counter = 0

    while length_range[0] < length_range[1]:
        """
        As with any binary search, we want to see the stuck condition.
        If we reach (n, n + 1), current_length becomes n.

        if flag is True at this stage, we go to (n + 1, n + 1)
        if flag is False at this stage, we go to (n, n)

        However, (a, a) is not acceptable as we assume length[1] is excluded
        because we assume pythonic indexing
        """
        flag = repeat_checker(current_length)
        decisions[current_length] = flag

        if decisions[current_length] is True and decisions[current_length + 1] is False:
            return current_length + 1
        
        if decisions[current_length - 1] is True and decisions[current_length] is False:
            return current_length

        if flag:
            length_range = (current_length + 1, length_range[1])
        else:
            length_range = (length_range[0], current_length)

        current_length = sum(length_range) // 2

        loop_counter += 1

        if loop_counter > 100:
            raise ValueError(
                f"Loop counter failed, arguments: string={string}, "
                f"start_length={start_length}, max_length={max_length}, "
                f"suffix={suffix}")

    if decisions[start_length] is False:
        return start_length
    else:
        raise ValueError(
            f"Cannot find correct length, arguments: string={string}, "
            f"start_length={start_length}, max_length={max_length}, "
            f"suffix={suffix}")


def create_example(
    sequence: str,
    min_masked_segment: int = 1,
    max_masked_segment: int = 50,
    min_unmasked_length: int = 2,
    include_extremities: bool = False,
    special_tokens: SpecialTokens = _DEFAULT_SPECIAL_TOKENS,
    randint_functor: Callable = random.randint,
    randsample_functor: Callable = random.sample,
    silent: bool = False,
) -> Union[tuple, None]:
    """
    Creates an example for training the model

    sequence: The sequence from which to create training example
    min_masked_segment: Minimum length of masked segment
    max_masked_segment: Maximum length of masked segment
    min_unmasked_length: Minimum length of unmasked segment (Also we make sure minimum length is
        at least half the length of the sequence)
    include_extremities: Whether the left and right extremeties can be included in the
        masked segment
    special_tokens: Special tokens to punctuate the sequence
    """
    # Make sure that at least half the sequence is left after masking.
    # Make sure that at least one AA is left on either extremity if
    # include_extremities is False.
    left_guard_band = 0 if include_extremities else \
        find_minimal_non_repeating_presuffix(sequence, suffix=False)
    right_guard_band = 0 if include_extremities else \
        find_minimal_non_repeating_presuffix(sequence, suffix=True)

    if left_guard_band is None or right_guard_band is None:
        return None

    try:
        mask_length = randint_functor(
            min_masked_segment,
            min(
                len(sequence) // 2,
                max_masked_segment,
                len(sequence) - min_unmasked_length,
                len(sequence) - (left_guard_band + right_guard_band),
            )
        )
        left_mask_location = randsample_functor(
            range(left_guard_band, len(sequence) - mask_length - right_guard_band + 1), 1)[0]
    except ValueError:
        funcargs = f"seq={sequence}, min_masked_segment={min_masked_segment}, "\
            f"max_masked_segment={max_masked_segment}, include_extremities={include_extremities}"
        if not silent:
            logger.warning(f"Cannot create masked sequence under specifications {funcargs}")
        return None

    return split_sequence(sequence, left_mask_location, mask_length, special_tokens)


def sample_random_sequence_segment(
    seq: str, length: int, randsample_functor: Callable = random.sample):
    start_pos = randsample_functor(range(0, len(seq) - length + 1), 1)[0]
    return seq[start_pos: start_pos + length]


class Tokenizer:
    def __init__(
        self,
        max_sequence_length: int = 1024,
        max_masked_segment: int = 50,
        include_extremities: bool = False,
        special_tokens: SpecialTokens = _DEFAULT_SPECIAL_TOKENS,
        randsample_functor: Callable = random.sample,
        randint_functor: Callable = random.randint,
        silent: bool = False,
        min_masked_segment: int = 1,
    ):
        self.max_sequence_length = max_sequence_length
        self.max_masked_segment = max_masked_segment
        self.include_extremities = include_extremities
        self.special_tokens = special_tokens
        self.mapper = {}
        self._set_tokenizer(special_tokens)
        self.randsample_functor = randsample_functor
        self.randint_functor = randint_functor
        self.silent = silent
        self.min_masked_segment = min_masked_segment

    def _set_tokenizer(self, special_tokens: SpecialTokens):
        self.mapper = {i: j for j, i in enumerate(AMINO_ACIDS)}
        self.mapper[special_tokens.start_of_sequence] = len(self.mapper)
        self.mapper[special_tokens.end_of_sequence] = len(self.mapper)
        self.mapper[special_tokens.masked_segment_indicator] = len(self.mapper)
        self.mapper["*"] = self.mapper[special_tokens.end_of_sequence]

    @property
    def max_encoder_length(self):
        """
        This happens when the sequence length is max length, one location gets
        masked and a [CLS] and [SEP] token are applied on both ends
        """
        return self.max_sequence_length + 2

    @property
    def max_decoder_length(self):
        """
        This happens when masked segment reaches max length. That plus a single [MASK]
        token at the end causes this situation
        """
        return self.max_masked_segment + 1

    def tokenize(self, sequence: str) -> list:
        if len(sequence) > self.max_sequence_length:
            segment = sample_random_sequence_segment(
                sequence,
                self.max_sequence_length,
                randsample_functor=self.randsample_functor,
            )
        else:
            segment = sequence

        result = create_example(
            segment,
            max_masked_segment=self.max_masked_segment,
            include_extremities=self.include_extremities,
            special_tokens=self.special_tokens,
            randsample_functor=self.randsample_functor,
            randint_functor=self.randint_functor,
            silent=self.silent,
            min_masked_segment=self.min_masked_segment,
        )

        if result is None:
            return None

        encoder_input, decoder_input = result

        return [self.mapper[i] for i in encoder_input], [self.mapper[i] for i in decoder_input]


class MemmapDataset:
    """
    Stores a single memory map of items with the corresponding mask
    """
    def __init__(self, fileprefix: str, num_items: int, max_length: int, mode: str = "r"):
        self.num_items = num_items
        self.max_length = max_length
        self.data = np.memmap(
            fileprefix + ".data.memmap",
            dtype=np.ubyte,
            mode=mode,
            shape=(num_items, max_length)
        )
        self.mask = np.memmap(
            fileprefix + ".mask.memmap",
            dtype=np.ubyte,
            mode=mode,
            shape=(num_items, max_length)
        )
        self.mode = mode

    def _initialize(self) -> None:
        self.data.fill(0)
        self.mask.fill(0)

    def __setitem__(self, idx: int, item: Union[np.ndarray, list]) -> None:
        if self.mode == "r":
            raise ValueError(f"Cannot write when mode is {self.mode}")

        length = len(item) if type(item) is list else item.shape[0]
        mask = np.zeros(shape=(self.max_length, ), dtype=np.ubyte)
        mask[:length] = 1
        self.data[idx][:length] = item
        self.mask[idx][:] = mask

    def __getitem__(self, idx) -> tuple:
        return (self.data[idx], self.mask[idx])

    def __len__(self) -> int:
        return self.num_items


class SqLiteData:
    """
    SqLite based list storage for strings
    """
    def __init__(self, prefix: str, mode: str = "r"):
        self.db = sqlite3.connect(prefix + ".db")
        if mode != "r" and mode != "r+":
            self.db.execute("CREATE TABLE data (row INTEGER PRIMARY KEY, data TEXT)")

    def __getitem__(self, idx: int) -> str:
        return self.db.execute("SELECT data FROM data WHERE row = ?", (idx, )).fetchone()[0]

    def __setitem__(self, idx: int, item: str) -> None:
        self.db.execute("INSERT INTO data VALUES (?, ?)", (idx, item))

    def close(self):
        self.db.commit()
        self.db.close()


class CharStorage:
    """
    A character array storage for strings
    """
    def __init__(
        self,
        fileprefix: str,
        num_items: int,
        max_length: int = 128,
        mode: str = "r",
        special_char: str = ".",
    ):
        self.num_items = num_items
        self.max_length = max_length
        self.data = np.memmap(
            fileprefix + ".memmap",
            dtype=np.ubyte,
            mode=mode,
            shape=(num_items, max_length),
        )
        self.special_char = special_char

    def _initialize(self) -> None:
        self.data.fill(0)

    def __setitem__(self, idx: int, item: str) -> None:
        if self.special_char in item:
            raise ValueError(f"Item {item} contains special char {self.special_char}")
        item = item + self.special_char
        if len(item) > self.max_length:
            item = item[:self.max_length]
        array = [ord(x) for x in item]
        np_array = np.zeros(shape=(self.max_length, ), dtype=np.ubyte)
        np_array[: len(array)] = array
        self.data[idx] = np_array

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, idx: int) -> str:
        item = self.data[idx]
        res = []
        for i in item:
            c = chr(i)
            if c == self.special_char:
                break
            res.append(c)
        return "".join(res)


@dataclass
class DiskStorageDataItem:
    itokens: Union[np.memmap, np.ndarray, torch.Tensor]
    otokens: Union[np.memmap, np.ndarray, torch.Tensor]
    metadata: str

    def tensorize(self) -> None:
        def helper(tdict: dict) -> dict:
            tdict = {key: torch.Tensor(np.array(value)).byte() for key, value in tdict.items()}
            return tdict

        self.itokens = helper(self.itokens)
        self.otokens = helper(self.otokens)


class DiskStorage:
    def __init__(
        self,
        max_length_encoder: int,
        max_length_decoder: int,
        num_sequences: int,
        datadir: str,
        mode: str = "r",
        length: Optional[int] = None,
    ):
        if mode not in ["r", "r+", "w+"]:
            raise ValueError(f"Only modes r, r+, w+ supported. Received {mode}")

        if mode == "w+" and os.path.exists(datadir):
            raise ValueError(f"Directory {datadir} already exists!")

        if mode == "w+":
            os.makedirs(datadir)

        self.encoder_memmap_data = MemmapDataset(
            os.path.join(datadir, "encoder_data"),
            num_items=num_sequences,
            max_length=max_length_encoder,
            mode=mode,
        )

        self.decoder_memmap_data = MemmapDataset(
            os.path.join(datadir, "decoder_data"),
            num_items=num_sequences,
            max_length=max_length_decoder,
            mode=mode,
        )

        self.metadata = CharStorage(
            os.path.join(datadir, "metadata"),
            num_items=num_sequences,
            mode=mode,
        )

        self.num_sequences = num_sequences
        self._length = 0

        # Remaining data to enable load
        self.max_length_encoder = max_length_encoder
        self.max_length_decoder = max_length_decoder
        self.datadir = datadir
        self.mode = mode

        if length is not None:
            self._length = length

        self._tokenizer = None

    def __len__(self) -> int:
        return self._length

    def _initialize(self) -> None:
        """
        Initialize all memmap arrays so that disk storage is reserved
        """
        self.encoder_memmap_data._initialize()
        self.decoder_memmap_data._initialize()
        self.metadata._initialize()
        self._length = self.num_sequences

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value: Tokenizer):
        assert(type(value) is Tokenizer)
        self._tokenizer = value

    def __setitem__(
        self,
        idx: int,
        items: tuple,
    ) -> None:
        encoder_data, decoder_data, metadata = items
        self.encoder_memmap_data[idx] = encoder_data
        self.decoder_memmap_data[idx] = decoder_data
        self.metadata[idx] = metadata

    def __getitem__(self, idx: int) -> DiskStorageDataItem:
        encoder_return_items = self.encoder_memmap_data[idx]
        decoder_return_items = self.decoder_memmap_data[idx]
        metadata = self.metadata[idx]
        def array_ize(x: dict) -> dict:
            return {key: np.array(value) for key, value in x.items()}
        encoder_data = array_ize({
            "input_ids": encoder_return_items[0], "attention_mask": encoder_return_items[1]})
        decoder_data = array_ize({
            "input_ids": decoder_return_items[0], "attention_mask": decoder_return_items[1]})

        return DiskStorageDataItem(itokens=encoder_data, otokens=decoder_data, metadata=metadata)

    def append(
        self,
        encoder_data: Union[np.ndarray, list],
        decoder_data: Union[np.ndarray, list],
        metadata: str
    ) -> None:
        if self._length >= self.num_sequences:
            raise ValueError("Exceeded memory size, exiting")

        self[self._length] = (encoder_data, decoder_data, metadata)
        self._length += 1

    def close(self) -> None:
        if self.mode != "w+":
            return

        with open(os.path.join(self.datadir, "config.json"), "w") as fhandle:
            json.dump(
                {
                    "max_length_encoder": self.max_length_encoder,
                    "max_length_decoder": self.max_length_decoder,
                    "num_sequences": self.num_sequences,
                    "length": self._length,
                },
                fhandle
            )
        if self.tokenizer is not None:
            try:
                with open(os.path.join(self.datadir, "tokenizer.pkl"), "wb") as fhandle:
                    pickle.dump(self.tokenizer, fhandle)
            except AttributeError:
                del self.tokenizer.randsample_functor
                del self.tokenizer.randint_functor
                with open(os.path.join(self.datadir, "tokenizer.pkl"), "wb") as fhandle:
                    pickle.dump(self.tokenizer, fhandle)

    @classmethod
    def load(cls, datadir: str, mode: str = "r"):
        with open(os.path.join(datadir, "config.json"), "r") as fhandle:
            config = json.load(fhandle)

        obj = cls(
            **config,
            datadir=datadir,
            mode=mode,            
        )

        tokenizer_file = os.path.join(datadir, "tokenizer.pkl")

        if os.path.exists(tokenizer_file):
            with open(tokenizer_file, "rb") as fhandle:
                obj.tokenizer = pickle.load(fhandle)

        return obj


class Dataset(torch.utils.data.Dataset):
    def __init__(self, datadir: str):
        super().__init__()
        self.data = DiskStorage.load(datadir, mode="r")
        self._length = len(self.data)

    def __len__(self) -> int:
        return self._length

    def set_length(self, max_idx: int) -> None:
        self._length = max_idx

    def __getitem__(self, idx: int) -> DiskStorageDataItem:
        res = self.data[idx]
        res.tensorize()
        return res


class CompoundDiskStorageReader:
    """
    When distributed launch is used to dump data, use this class
    """
    def __init__(self, datadir: str, data_type: str, max_thread_count: int = 50):
        self.data = []

        for i in range(max_thread_count):
            dirname = os.path.join(datadir, str(i))
            if os.path.exists(dirname):
                data_of_type = os.path.join(dirname, data_type)
                if os.path.exists(data_of_type):
                    self.data.append(DiskStorage.load(data_of_type, mode="r"))
            else:
                break
        else:
            raise ValueError(
                f"Potentially more than {max_thread_count} fragments. Cannot process.")

    @property
    def tokenizer(self):
        return self.data[0].tokenizer

    def __len__(self):
        return sum(len(i) for i in self.data)

    def __getitem__(self, idx: int) -> DiskStorageDataItem:
        for i, d in enumerate(self.data):
            if idx < len(d):
                return d[idx]
            idx -= len(d)
        raise ValueError(f"Index {idx} out of bounds")


class CompoundDataset(torch.utils.data.Dataset):
    def __init__(self, datadir: str, data_type: str):
        super().__init__()
        self.data = CompoundDiskStorageReader(datadir, data_type)
        self._length = len(self.data)

    def __len__(self) -> int:
        return self._length

    def set_length(self, max_idx: int) -> None:
        self._length = max_idx

    def __getitem__(self, idx: int) -> DiskStorageDataItem:
        res = self.data[idx]
        res.tensorize()
        return res


def collate_single(items: List[dict]) -> dict:
    """
    Just stack into a single output dictionary
    """
    all_input_ids = [item["input_ids"] for item in items]
    all_attn_masks = [item["attention_mask"] for item in items]
    return {
        "input_ids": torch.stack(all_input_ids, dim=0),
        "attention_mask": torch.stack(all_attn_masks, dim=0),
    }


def prepare_label_input(decoder_inputs: Dict[str, torch.Tensor]) -> torch.LongTensor():
    """
    According to
    https://github.com/huggingface/transformers/blob/v4.16.2-release/src/transformers/models/encoder_decoder/modeling_encoder_decoder.py#L508,
    when labels are provided and decoder_inputs are not provided, the model left zero-pads
    labels to produce decoder_inputs. Since decoder outputs are "causally" generated, we
    can simply put the value -100 wherever attention mask is zero.
    """
    labels = decoder_inputs["input_ids"].long()
    labels = labels.masked_fill(decoder_inputs["attention_mask"] == 0, -100)
    return labels


def collate_function(batch: List[DiskStorageDataItem]) -> tuple:
    """
    Check EncoderDecoderModel forward specification @
    https://github.com/huggingface/transformers/blob/v4.16.2-release/src/transformers/models/encoder_decoder/modeling_encoder_decoder.py#L508
    """
    encoder_inputs = collate_single([i.itokens for i in batch])
    decoder_inputs = collate_single([i.otokens for i in batch])
    labels = prepare_label_input(decoder_inputs)
    return {
        "input_ids": encoder_inputs["input_ids"].long(),
        "attention_mask": encoder_inputs["attention_mask"].byte(),
        "labels": labels.long(),
        # "metadata": [i.metadata for i in batch]
    }


def get_random_section(
    seq: list,
    slice_length: int,
    randint_functor: Callable = random.randint
) -> list:
    position = randint_functor(0, len(seq) - slice_length)
    return seq[position: position + slice_length]


class SimpleSequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        seq_file: str,
        mapper: Optional[dict] = None,
        max_length: int = 1400,
        randint_functor: Callable = random.randint,
        ignore_too_long: bool = False,
    ):
        super().__init__()
        # self.seq_list = [x for x in seq_list if "stop" not in x]
        self.references = []

        logger.info("Reading input and doing checks")

        with pysam.FastaFile(seq_file) as fhandle:
            for r in tqdm.tqdm(fhandle.references, "Fasta read"):
                seq = fhandle.fetch(r)

                if "stop" in seq:
                    continue

                if ignore_too_long:
                    if len(seq) > max_length - 1:
                        continue

                self.references.append(r)

        self.seq_file = seq_file        

        if not mapper:
            mapper = Tokenizer().mapper
        self.mapper = mapper
        self.max_length = max_length
        self.randint_functor = randint_functor
        self._handle = None

    @property
    def seq_list(self):
        if self._handle is None:
            self._handle = pysam.FastaFile(self.seq_file)

        return self._handle

    def __len__(self):
        return len(self.references)

    def __getitem__(self, idx: int) -> dict:
        ref = self.references[idx]
        seq = self.seq_list.fetch(ref)
        seq = list(seq)
        if len(seq) > self.max_length - 1:
            seq = get_random_section(
                seq, slice_length=self.max_length - 1, randint_functor=self.randint_functor)
        seq = ["[CLS]"] + seq
        res = torch.LongTensor([self.mapper[i] for i in seq])
        input_ids = torch.zeros(self.max_length).long()
        attn_mask = torch.zeros(self.max_length).byte()
        input_ids[: res.shape[0]] = res
        attn_mask[: res.shape[0]] = 1
        return {"input_ids": input_ids, "attention_mask": attn_mask}


def collate_function_for_decoder(batch: List[torch.LongTensor]) -> dict:
    """
    Note that BertLMHead model auto-shifts labels to make sure predictions
    match the inputs. Code reference:
    https://github.com/huggingface/transformers/blob/v4.16.2-release/src/transformers/models/bert/modeling_bert.py#L1246
    """
    results = collate_single(batch)
    labels = results["input_ids"].masked_fill(results["attention_mask"] == 0, -100)
    results["labels"] = labels
    return results
