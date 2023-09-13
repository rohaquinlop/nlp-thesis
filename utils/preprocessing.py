"""
Preprocessing module will contain all the functions and classes to preprocess the data.
"""
import re
import nltk.data
from nltk.stem import PorterStemmer
import nltk.corpus
from nltk import word_tokenize, ngrams, sent_tokenize


class Data:
    """
    Data class will represent the data to be summarized.
    """

    def __init__(self, text_raw: str) -> None:
        self.text_raw = text_raw
        self.text_sentences = []
        self.uncleaned_sentences = []
        self.sentence_tokenized = []
        self.stemmed_sentences = []
        self.ngrams = []
        self.char_ngrams = []

    def tokenize(self) -> None:
        """
        Tokenize the text into sentences.
        """
        text_sentences = []

        for line in self.text_raw.splitlines():
            text_sentences.extend(sent_tokenize(line, language="english"))

        self.text_sentences = text_sentences

    def clean(self, stem=True, n_grams_length=2) -> None:
        """
        Clean the text.
        """
        uncleaned_sentences = []
        cleaned_ngrams = []

        cleaned_tokenized_sentences = []
        stopwords = set(nltk.corpus.stopwords.words("english"))

        for sentence in self.text_sentences:
            sentence = re.sub(r"[^a-z\d ]", "", sentence.lower())
            uncleaned_sentences.append(sentence)
            tokens = word_tokenize(sentence)
            cleaned_vec = [w for w in tokens if w.lower() not in stopwords]
            cleaned_tokenized_sentences.append(cleaned_vec)
            cleaned_ngrams.append(list(ngrams(cleaned_vec, n_grams_length)))

        self.sentence_tokenized = list(
            filter(lambda x: len(x) >= 1, cleaned_tokenized_sentences)
        )

        self.uncleaned_sentences = uncleaned_sentences
        self.ngrams = cleaned_ngrams
        self.char_ngrams = list(
            ngrams("".join(self.text_sentences).split(), n_grams_length)
        )

        if stem:
            # PorterStemmer: is a process for removing suffixes from words
            # ps.stem('code') -> code
            # ps.stem('coding') -> code
            # ps.stem('coded') -> code
            stemmed = []
            porter_stemmer = PorterStemmer()

            for sentence in self.sentence_tokenized:
                stem_sentence = [porter_stemmer.stem(word) for word in sentence]

                stemmed.append(stem_sentence)

            self.stemmed_sentences = stemmed

    def generate_ngrams(self, n_grams_length=3) -> list:
        """
        Get the ngrams from the text.
        """
        n_grams = [list(ngrams(sentence, n_grams_length)) for sentence in self.text_raw]

        return n_grams
