"""Defines a class for constructing Pakistan Sign Language from text using rules."""

import random
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from sign_language_translator.config.assets import Assets
from sign_language_translator.config.enums import SignLanguages
from sign_language_translator.languages.sign.mapping_rules import (
    CharacterByCharacterMappingRule,
    DirectMappingRule,
    LambdaMappingRule,
    MappingRule,
)
from sign_language_translator.languages.sign.sign_language import SignLanguage
from sign_language_translator.languages.vocab import Vocab
from sign_language_translator.text import Tags


class PakistanSignLanguage(SignLanguage):
    """A class representing the Pakistan Sign Language.

    It provides methods for converting tokens to sign dictionaries and restructuring sentences.

    Attributes:
        STOPWORDS (set): A set of stopwords in Pakistan Sign Language.
    """

    STOPWORDS = {"the", "so"}

    @staticmethod
    def name() -> str:
        return SignLanguages.PAKISTAN_SIGN_LANGUAGE.value

    def __init__(self) -> None:
        # load word maps and info from dataset files
        self.vocab = Vocab(
            language=r".+",
            country=r"^pk$",
            organization=r".+",
            part_number=r"[0-9]+",
            data_root_dir=Assets.ROOT_DIR,
            arg_is_regex=True,
        )
        # restructure dict values
        self.word_to_sign_dict = {
            word: self._make_equal_weight_sign_dict(labels)
            for word, labels in self.vocab.word_to_labels.items()
        }

        # define mapping rules
        self._direct_rule = self.__get_direct_mapping_rule(4)
        self._double_handed_spelling_rule = self.__get_spelling_rule(5, "double-handed")
        self._single_handed_spelling_rule = self.__get_spelling_rule(5, "single-handed")
        self._urdu_character_rule = self.__get_urdu_spelling_rule(5)
        self._number_rule = self.__get_number_rule(5)

        self.mapping_rules: List[MappingRule] = [
            self._direct_rule,
            self._double_handed_spelling_rule,
            self._single_handed_spelling_rule,
            self._urdu_character_rule,
            self._number_rule,
            # self._time_rule, # x:y:z -> x hours y minutes z seconds
            # self._date_rule,
        ]

    def tokens_to_sign_dicts(
        self,
        tokens: Iterable[str],
        tags: Optional[Iterable[Any]] = None,
        contexts: Optional[Iterable[Any]] = None,
    ) -> List[Dict[str, Union[List[List[str]], List[float]]]]:
        # fix args
        pass

    def _apply_rules(
        self, token: str, tag=None, context=None
    ) -> List[Dict[str, Union[List[List[str]], List[float]]]]:
        """Applies all the mapping rules to a token.
        Rules with lower value of priority overwrite the result.
        If multiple rules of same priority are applicable, one is selected at random.

        Args:
            token (str): The token to apply the rules to.
            tag (Any, optional): The tag associated with the token. Defaults to None.
            context (Any, optional): The context associated with the token. Defaults to None.

        Returns:
            List[Dict[str, List[List[str]] | List[float]]]:
                A list of sign dictionaries for the token.
        """
        pass

    def restructure_sentence(
        self,
        sentence: Iterable[str],
        tags: Optional[Iterable[Any]] = None,
        contexts: Optional[Iterable[Any]] = None,
    ) -> Tuple[Iterable[str], Iterable[Any], Iterable[Any]]:
        # Fix the args
        pass

    def __call__(
        self,
        tokens: Iterable[str],
        tags: Optional[Iterable[Any]] = None,
        contexts: Optional[Iterable[Any]] = None,
    ) -> List[Dict[str, Union[List[List[str]], List[float]]]]:
        tokens, tags, contexts = self.restructure_sentence(
            tokens, tags=tags, contexts=contexts
        )
        signs = self.tokens_to_sign_dicts(tokens, tags=tags, contexts=contexts)

        return signs

    def __get_direct_mapping_rule(self, priority=5):
        return DirectMappingRule(
            {w: [sd] for w, sd in self.word_to_sign_dict.items()}, priority
        )

    def __get_spelling_rule(
        self,
        priority: int,
        word_sense_filter="handed-letter",
    ):
        return CharacterByCharacterMappingRule(
            {
                k__: v
                for k, v in self.word_to_sign_dict.items()
                for k_ in [self.vocab.remove_word_sense(k)]
                for k__ in [k_.lower(), k_.upper()]
                if word_sense_filter in k
            },
            {Tags.NAME, Tags.ACRONYM},
            priority,
        )

    def __get_urdu_spelling_rule(self, priority=5):
        return CharacterByCharacterMappingRule(
            {
                k: v
                for k, v in self.word_to_sign_dict.items()
                if len(k) == 1 and not k.isnumeric()
            },
            {Tags.NAME},
            priority,
        )

    def __get_number_rule(self, priority=5):
        return LambdaMappingRule(
            is_applicable_function=lambda token, tag, context: (
                tag == Tags.NUMBER
                and all(
                    digit in self.word_to_sign_dict
                    for digit in self.__chunk_a_number(token)
                )
            ),
            apply_function=lambda x: [
                self.word_to_sign_dict[digit] for digit in self.__chunk_a_number(x)
            ],
            priority=priority,
        )

    def __chunk_a_number(self, num):
        return re.findall(
            r"("
            + r"|".join(sorted(self.vocab.numeric_keys, key=len, reverse=True))
            + r"|\d|\.|.)",
            str(self.vocab.words_to_numbers.get(num, num)),
        )
