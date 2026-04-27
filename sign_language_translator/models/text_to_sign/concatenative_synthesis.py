"""This module defines the ConcatenativeSynthesis class, which represents a rule based model for translating text to sign language."""

from __future__ import annotations

import random
from enum import Enum
from typing import List, Type, Union

from sign_language_translator.config.enums import (
    SignEmbeddingModels,
    SignFormats,
    normalize_short_code,
)
from sign_language_translator.languages import get_sign_language, get_text_language
from sign_language_translator.languages.sign import SignLanguage
from sign_language_translator.languages.text import TextLanguage
from sign_language_translator.models.text_to_sign.t2s_model import TextToSignModel
from sign_language_translator.vision._utils import get_sign_wrapper_class
from sign_language_translator.vision.sign.sign import Sign


class ConcatenativeSynthesis(TextToSignModel):
    """A class representing a Rule-Based model for translating text to sign language
    by concatenating sign language videos.
    """

    def __init__(
        self,
        text_language: Union[str, TextLanguage, Enum],
        sign_language: Union[str, SignLanguage, Enum],
        sign_format: Union[str, Type[Sign]],
        sign_embedding_model: Union[str, Enum, None] = None,
    ) -> None:
        """
        Args:
            text_language (str | TextLanguage | Enum): (source) The text language processor object or its identifier. (e.g. "urdu" or `slt.languages.text.Urdu()`. See `slt.TextLanguageCodes` for all options.)
            sign_language (str | SignLanguage | Enum): (target) The sign language processor object or its identifier. (e.g. "pk-sl" or `slt.languages.sign.PakistanSignLanguage()`. See `slt.SignLanguageCodes` for all options.)
            sign_format (str | Type[Sign]): (format) The sign features used for mapping labels to sign features. (e.g. "video" or `slt.vision.Video` or "landmarks" or `slt.vision.Landmarks`. See `slt.SignFormatCodes` for all options.)
            sign_embedding_model (str | Enum | None, optional): The name of the model used for extracting features from the signs in available datasets. Not required for Video sign_format. (e.g. "mediapipe-world". See `slt.enums.SignEmbeddingModels` for all options.)
        """
        self._text_language = None
        self._sign_language = None
        self._sign_format = None
        self._sign_embedding_model = None

        self.text_language = text_language
        self.sign_language = sign_language
        self.sign_format = sign_format
        self.sign_embedding_model = sign_embedding_model

    @property
    def text_language(self) -> TextLanguage:
        """An object of `slt.languages.text.TextLanguage` class or its child that defines preprocessing, tokenization & other NLP functions."""
        pass

    @text_language.setter
    def text_language(self, text_language: Union[str, TextLanguage, Enum]):
        pass

    @property
    def sign_language(self) -> SignLanguage:
        """An object of `slt.languages.sign.SignLanguage` class or its child that defines the mapping rules & grammar of a sign language."""
        pass

    @sign_language.setter
    def sign_language(self, sign_language: Union[str, SignLanguage, Enum]):
        pass

    @property
    def sign_format(self) -> Type[Sign]:
        """
        The format of the sign language (e.g. `slt.Vision.sign.sign.Sign` or subclass).

        Class that wraps the sign language features e.g. raw videos or landmarks.
        This class can load the signs from available datasets and concatenate its objects.
        e.g. `slt.Video` or `slt.Landmarks` class.
        """
        pass

    @sign_format.setter
    def sign_format(self, sign_format: Union[str, Type[Sign], Enum]) -> None:
        pass

    @property
    def sign_embedding_model(self) -> Union[str, None]:
        """The name of the model which was used for extracting features from the signs.
        This name is used in the filenames of the preprocessed signs dataset."""
        pass

    @sign_embedding_model.setter
    def sign_embedding_model(self, model: Union[str, Enum, None]) -> None:
        pass

    def translate(self, text: str, *args, **kwargs) -> Sign:
        """
        Translate text to sign language.

        Args:
            text: The input text to be translated.

        Returns:
            The translated sign language sentence.

        """
        pass

    def _map_labels_to_sign(self, labels: List[str]) -> List[Sign]:
        pass

    def _prepare_resource_name(self, label, person=None, camera=None, sep="_"):
        pass

    def __get_text_language_object(
        self, text_language: Union[str, TextLanguage, Enum]
    ) -> TextLanguage:
        pass

    def __get_sign_language_object(
        self, sign_language: Union[str, SignLanguage, Enum]
    ) -> SignLanguage:
        pass

    def __get_sign_format_class(
        self, sign_format: Union[str, Type[Sign], Enum]
    ) -> Type[Sign]:
        pass
