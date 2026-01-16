#!/usr/bin/env python3

from pathlib import Path
from abc import ABC, abstractmethod


class AudioFilePreprocessor(ABC):
    """
    Base class for audio file processors.
    """
    @abstractmethod
    def process_file(self, input_path: Path, output_path: Path) -> None:
        """
        Processes a single audio file and saves it to output_path.
        """
        ...