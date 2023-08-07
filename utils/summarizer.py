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
    
    matrix = self._create_matrix(sentences_words, self.threshold, tf_metrics, idf_metrics)
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
    degrees = np.zeros((sentences_count, ))
    
    for row, (sentence1, tf1) in enumerate(zip(sentences, tf_metrics)):
      for col, (sentence2, tf2) in enumerate(zip(sentences, tf_metrics)):
        matrix[row, col] = self.cosine_similarity(sentence1, sentence2, tf1, tf2, idf_metrics)
        
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
    
    return self._create_matrix(sentences_words, self.threshold, tf_metrics, idf_metrics)
  
  @staticmethod
  def cosine_similarity(
    sentence1,
    sentence2,
    tf1,
    tf2,
    idf_metrics: dict[str, float]
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


class SimRankSummarizer(AbstractSummarizer):
  """
  SimRank Text Summarizer
  """
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
  
  def _to_words_set(self, sentence):
    words = map(self.normalize_word, sentence.words)
    return [self.stem_word(w) for w in words if w not in self.stopwords]
  
  def __call__(self, document: ObjectDocumentModel, sentence_count: int, decay_factor: float = 0.85):
    sentences_words = self.get_sentences_words(document)
    
    if not sentences_words:
      return tuple()
    
    matrix = self._create_matrix(sentences_words, decay_factor)
    scores = self.power_method(matrix, self.epsilon)
    ratings = dict(zip(document.sentences, scores))
    
    return self._get_best_sentences(document.sentences, sentence_count, ratings)
  
  def _create_matrix(self, sentences, decay_factor: float):
    """
    Create matrix of shape |sentences|x|sentences|
    """
    sentences_count = len(sentences)
    matrix = np.zeros((sentences_count, sentences_count))
    
    for row, sentence1 in enumerate(sentences):
      for col, sentence2 in enumerate(sentences):
        if row == col:
          matrix[row][col] = 1.0
        else:
          matrix[row][col] = self.similarity_rank(sentence1, sentence2, decay_factor)
    
    return matrix
  
  @staticmethod
  def similarity_init(sentence1, sentence2):
    n = len(sentence1)
    m = len(sentence2)
    
    matrix = np.zeros((n, m))
    
    # Initialize matrix
    for i in range(n):
      for j in range(m):
        matrix[i][j] = 1.0 if sentence1[i] == sentence2[j] else 0.0
    
    return matrix
  
  @staticmethod
  def calculate_similarity_rank(matrix, sentence1, sentence2, row: int, col: int, decay_factor: float) -> float:
    if sentence1[row] == sentence2[col]:
      return 1.0
    
    simrank_sum = 0.0
    n = len(sentence1)
    m = len(sentence2)
    
    if (n-1)*(m-1) == 0:
      return 0.0
    
    for i in range(n):
      for j in range(m):
        if i != row and j != col:
          simrank_sum += matrix[i][j]
    
    scale = decay_factor / ( (n-1)*(m-1) )
    new_similarity_rank = scale * simrank_sum
    
    return new_similarity_rank

  @staticmethod
  def similarity_rank(sentence1, sentence2, decay_factor: float):
    n = len(sentence1)
    m = len(sentence2)
    
    matrix = np.zeros((n, m))
  
  def create_matrix(self, document: ObjectDocumentModel, sentence_count):
    sentences_words = [self._to_words_set(s) for s in document.sentences]
    
    if not sentences_words:
      return None
    
    tf_metrics = self._compute_tf(sentences_words)
    idf_metrics = self._compute_idf(sentences_words)
    
    return self._create_matrix(sentences_words, self.threshold, tf_metrics, idf_metrics)
  
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
