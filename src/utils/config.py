#!/usr/bin/env python3

EMOTION_MAP = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

LABEL2ID = {v: i for i, v in enumerate(EMOTION_MAP.values())}
ID2LABEL = {i: v for v, i in LABEL2ID.items()}

TRAIN_ACTORS = set(range(1, 21))
VAL_ACTORS = set(range(21, 23))
TEST_ACTORS = set(range(23, 25))