# Copyright (c) 2023 University of Illinois Board of Trustees. All Rights Reserved.
# Developed at the ES|CAD group (http://dchen.ece.illinois.edu)
# This file is released under specific terms. See LICENSE.txt or go to https://opensource.org/license/mit/
import re
import numpy as np
import math
import logging

logger = logging.getLogger(__file__)


MISMATCH_PENALTY = float("inf")
DELETION_PENALTY = 1
INSERTION_PENALTY = 1

variant_pattern = re.compile("^([A-Za-z]+)([0-9]+)([A-Za-z]+)$")


def group_consecutive_deletions(variants: list) -> list:
    grouped = []
    result = []

    def _fix_group(group):
        return tuple([v[1] for v in group])

    for v in variants:
        matched = variant_pattern.match(v)
        try:
            position = int(matched.group(2))
        except Exception as e:
            logger.error(f"Error for {v} from {variants}")
            raise e

        if matched.group(3) == "del":
            if len(grouped) == 0 or grouped[-1][0] == position - 1:
                grouped.append((position, v))
            else:
                result.append(_fix_group(grouped))
                grouped = [(position, v)]
        else:
            if len(grouped) > 0:
                result.append(_fix_group(grouped))
                grouped = []

            result.append(v)

    if len(grouped) > 0:
        result.append(_fix_group(grouped))

    return result


def mut_sequence(seq: str) -> list:
    return group_consecutive_deletions([x for x in seq.split(",") if x != ""])


def match_score(m0: str, m1: str) -> float:
    """
    Match if two mutations are one and the same, otherwise,
    delete one mutation and insert another mutation. To enable
    that, match_score is infinite for mismatching mutations
    """
    return 0 if m0 == m1 else MISMATCH_PENALTY


def edit_distance_(seq0: str, seq1: str) -> float:
    mut0 = mut_sequence(seq0)
    mut1 = mut_sequence(seq1)
    edit_matrix = np.zeros((len(mut0) + 1, len(mut1) + 1))
    """
    Initialization: edit_matrix[i, j] represents the minimum number of
    edits required between mut0[:i], mut1[:j].

    Hence edit_matrix[0, j] = j, and edit_matrix[i, 0] = i representing
    respectively, j insertions into mut0 at the beginning, and i deletions
    from mut0 at the beginning.

    Back to the basics: Why do we need 0 to L + 1 indexing? If we assume
    (0, 0) indicates match of (mut0[0], mut1[0]), then to represent the score
    for insertion at the start, say (-, mut1[0]), we would need to use the
    indexing (-1, 0), which of course doesn't work as expected here.
    """
    edit_matrix[0, 1:] = np.arange(1, len(mut1) + 1) * INSERTION_PENALTY
    edit_matrix[1:, 0] = np.arange(1, len(mut0) + 1) * DELETION_PENALTY

    for i in range(1, len(mut0) + 1):
        for j in range(1, len(mut1) + 1):
            edit_matrix[i, j] = min(
                edit_matrix[i - 1, j - 1] + match_score(mut0[i - 1], mut1[j - 1]),
                edit_matrix[i - 1, j] + DELETION_PENALTY,
                edit_matrix[i, j - 1] + INSERTION_PENALTY,
            )

    return edit_matrix[-1, -1]


def is_only_insert_edit_possible(seq0: list, seq1: list) -> bool:
    """
    Is it possible to align seq0 to seq1 with only insertions into seq0
    """
    i = 0
    j = 0

    if len(seq0) > len(seq1):
        return False

    while i < len(seq0) and j < len(seq1):
        if seq0[i] == seq1[j]:
            i += 1
            j += 1
        else:
            j += 1

    return i == len(seq0)


def insert_only_edit_distance(seq0: str, seq1: str) -> float:
    seq0 = mut_sequence(seq0)
    seq1 = mut_sequence(seq1)

    if is_only_insert_edit_possible(seq0, seq1):
        return abs(len(seq1) - len(seq0))
    else:
        return float("inf")


def edit_distance(*args, **kwargs):
    if math.isinf(MISMATCH_PENALTY) and math.isinf(DELETION_PENALTY):
        return insert_only_edit_distance(*args, **kwargs)
    else:
        return edit_distance_(*args, **kwargs)

