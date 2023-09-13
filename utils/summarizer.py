"""
Implementation of Levenshtein distance based summarizer
"""
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)
from collections import (
    Counter,
)
from math import (
    log as math_log,
)
import numpy as np
from sumy.summarizers._summarizer import (
    AbstractSummarizer,
)
from sumy.models.dom._document import (
    ObjectDocumentModel,
)


class TestSummarizer(AbstractSummarizer):
    """
    Test class for summarizer
    """

    threshold = 0.1
    epsilon = 0.1
    _stop_words = frozenset()

    @property
    def stopwords(self):
        return self._stop_words

    @stopwords.setter
    def stopwords(self, words):
        self._stop_words = frozenset(map(self.normalize_word, words))

    def get_sentences_words(self, document: ObjectDocumentModel):
        return [self._to_words_set(s) for s in document.sentences]

    def __call__(self, document: ObjectDocumentModel, sentence_count: int):
        sentences_words = [self._to_words_set(s) for s in document.sentences]

        if not sentences_words:
            return tuple()

        tf_metrics = self._compute_tf(sentences_words)
        idf_metrics = self._compute_idf(sentences_words)

        matrix = self._create_matrix(
            sentences_words, self.threshold, tf_metrics, idf_metrics
        )
        scores = self.power_method(matrix, self.epsilon)
        ratings = dict(zip(document.sentences, scores))

        return self._get_best_sentences(document.sentences, sentence_count, ratings)

    def _to_words_set(self, sentence):
        words = map(self.normalize_word, sentence.words)
        return [self.stem_word(w) for w in words if w not in self.stopwords]

    def _compute_tf(self, sentences):
        tf_values = map(Counter, sentences)

        tf_metrics = []
        for sentence in tf_values:
            metrics = {}
            max_tf = self._find_tf_max(sentence)

            for term, tf in sentence.items():
                metrics[term] = tf / max_tf

            tf_metrics.append(metrics)

        return tf_metrics

    @staticmethod
    def _find_tf_max(terms):
        return max(terms.values()) if terms else 1

    @staticmethod
    def _compute_idf(sentences) -> dict[str, float]:
        idf_metrics: dict[str, float] = {}
        sentences_count = len(sentences)

        for sentence in sentences:
            for term in sentence:
                if term not in idf_metrics:
                    n_j = sum(1 for s in sentences if term in s)
                    idf_metrics[term] = math_log(sentences_count / (1 + n_j))

        return idf_metrics

    def _create_matrix(self, sentences, threshold, tf_metrics, idf_metrics):
        """
        Create matrix of shape |sentences|x|sentences|
        """
        sentences_count = len(sentences)
        matrix = np.zeros((sentences_count, sentences_count))
        degrees = np.zeros((sentences_count,))

        for row, (sentence1, tf1) in enumerate(zip(sentences, tf_metrics)):
            for col, (sentence2, tf2) in enumerate(zip(sentences, tf_metrics)):
                matrix[row, col] = self.cosine_similarity(
                    sentence1, sentence2, tf1, tf2, idf_metrics
                )

                if matrix[row][col] > threshold:
                    matrix[row][col] = 1.0
                    degrees[row] += 1
                else:
                    matrix[row][col] = 0.0

        for row in range(sentences_count):
            for col in range(sentences_count):
                if degrees[row] == 0:
                    degrees[row] = 1

                matrix[row][col] = matrix[row][col] / degrees[row]

        return matrix

    def create_matrix(self, document: ObjectDocumentModel, sentence_count):
        sentences_words = [self._to_words_set(s) for s in document.sentences]

        if not sentences_words:
            return None

        tf_metrics = self._compute_tf(sentences_words)
        idf_metrics = self._compute_idf(sentences_words)

        return self._create_matrix(
            sentences_words, self.threshold, tf_metrics, idf_metrics
        )

    @staticmethod
    def cosine_similarity(
        sentence1, sentence2, tf1, tf2, idf_metrics: dict[str, float]
    ) -> float:
        unique_words1 = frozenset(sentence1)
        unique_words2 = frozenset(sentence2)
        common_words = unique_words1 & unique_words2

        numerator = 0.0
        for term in common_words:
            numerator += tf1[term] * tf2[term] * idf_metrics[term] ** 2

        denominator1 = sum((tf1[t] * idf_metrics[t]) ** 2 for t in unique_words1)
        denominator2 = sum((tf2[t] * idf_metrics[t]) ** 2 for t in unique_words2)

        if denominator1 > 0 and denominator2 > 0:
            return numerator / (denominator1 * denominator2) ** 0.5
        else:
            return 0.0

    @staticmethod
    def power_method(matrix, epsilon):
        transposed_matrix = matrix.T
        sentences_count = len(matrix)
        p_vector = np.array([1.0 / sentences_count] * sentences_count)
        lambda_val = 1.0

        while lambda_val > epsilon:
            next_p = np.dot(transposed_matrix, p_vector)
            lambda_val = np.linalg.norm(np.subtract(next_p, p_vector))
            p_vector = next_p

        return p_vector


class LevenshteinSummarizer(AbstractSummarizer):
    """
    Levenshtein distance based summarizer
    """

    epsilon = 0.1
    _stop_words = frozenset()

    @property
    def stopwords(self):
        """
        Returns the stop words
        """
        return self._stop_words

    @stopwords.setter
    def stopwords(self, words):
        self._stop_words = frozenset(map(self.normalize_word, words))

    def get_sentences_words(self, document: ObjectDocumentModel):
        """
        Returns the list of sentences words
        """
        return [self._to_words_set(s) for s in document.sentences]

    def _to_words_set(self, sentence):
        words = map(self.normalize_word, sentence.words)
        return [self.stem_word(w) for w in words if w not in self.stopwords]

    def __call__(
        self,
        document: ObjectDocumentModel,
        sentence_count: int,
    ):
        sentences_words = self.get_sentences_words(document)

        if not sentences_words:
            return tuple()

        matrix = self._create_matrix(sentences_words)
        scores = self.power_method(matrix, self.epsilon)
        ratings = dict(zip(document.sentences, scores))

        return self._get_best_sentences(document.sentences, sentence_count, ratings)

    def create_matrix(self, document: ObjectDocumentModel):
        """
        Returns the matrix of shape |sentences|x|sentences|
        """
        sentences_words = [self._to_words_set(s) for s in document.sentences]

        if not sentences_words:
            return None

        return self._create_matrix(sentences_words)

    def _create_matrix(self, sentences):
        """
        Create matrix of shape |sentences|x|sentences|
        """
        sentences_count = len(sentences)
        matrix = np.zeros((sentences_count, sentences_count))

        for row, sentence1 in enumerate(sentences):
            for col, sentence2 in enumerate(sentences):
                matrix[row, col] = self._compute_levenshtein_distance(
                    sentence1,
                    sentence2,
                )

        return matrix

    def _compute_levenshtein_distance(self, sentence1, sentence2):
        """
        Compute the Levenshtein distance between two sentences
        """
        total_distance = 0
        min_len = min(len(sentence1), len(sentence2))

        for i in range(min_len):
            total_distance += self._compute_sentence_distance(
                sentence1[i],
                sentence2[i],
            )

        # min aumenta el precision pero disminuye el recall
        # max aumenta el recall pero disminuye la precision, hay un runtime error en la funcion power_method
        if min_len == 0:
            return 0
        else:
            return total_distance / (min_len)  # min or max? -> #TODO:

    def _compute_sentence_distance(self, word1, word2):
        """
        Compute the Levenshtein distance between two words
        """
        n_word_1 = len(word1)
        m_word_2 = len(word2)
        levenshtein_distance = [
            [0 for _ in range(m_word_2 + 1)] for _ in range(n_word_1 + 1)
        ]

        for i in range(n_word_1 + 1):
            levenshtein_distance[i][0] = i

        for i in range(m_word_2 + 1):
            levenshtein_distance[0][i] = i

        for i in range(1, n_word_1 + 1):
            for j in range(1, m_word_2 + 1):
                if word1[i - 1] == word2[j - 1]:
                    levenshtein_distance[i][j] = levenshtein_distance[i - 1][j - 1]
                else:
                    insertion = 1 + levenshtein_distance[i][j - 1]
                    deletion = 1 + levenshtein_distance[i - 1][j]
                    replacement = 1 + levenshtein_distance[i - 1][j - 1]

                    levenshtein_distance[i][j] = min(insertion, deletion, replacement)

        l_distance = levenshtein_distance[n_word_1][m_word_2]
        l_distance = 1 - (
            l_distance / min(n_word_1, m_word_2)
        )  # min o max? -MOD normalizacion
        return l_distance

    @staticmethod
    def power_method(matrix, epsilon):
        """
        Power method implementation
        """
        transposed_matrix = matrix.T
        sentences_count = len(matrix)
        p_vector = np.array([1.0 / sentences_count] * sentences_count)
        lambda_val = 1.0

        while lambda_val > epsilon:
            next_p = np.dot(transposed_matrix, p_vector)
            lambda_val = np.linalg.norm(np.subtract(next_p, p_vector))
            p_vector = next_p

        return p_vector
