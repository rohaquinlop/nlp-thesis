from automathon import (
  NFA,
)
import os


def draw_matrix(matrix, weights=None, f_set=None, image_name="graph_connections") -> None:
  Q :set[str] = set()
  sigma = set()
  delta: dict[str, dict[str, set[str]]] = {}
  
  n = len(matrix)
  initialState = None
  
  for row in range(n):
    for col in range(n):
      if matrix[row][col] > 0:
        Q.add(str(row))
        Q.add(str(col))
        
        similarity = f"{matrix[row][col]:.2f}"
        
        if similarity == '1.00':
          initialState = str(row)
        
        if str(row) not in delta:
          delta[str(row)] = {similarity: {str(col)}}
        else:
          if similarity not in delta[str(row)]:
            delta[str(row)][similarity] = {str(col)}
          else:
            delta[str(row)][similarity].add(str(col))
        
        sigma.add(similarity)

  # for node in graph:
  #   adj_set = set(graph[node])
  #   Q.add(node)
  #   Q = Q.union(adj_set)
    
  #   delta[node] = {'-': adj_set}
  #   # for i in range(len(sentence)-1):
  #   #   word1 = sentence[i]
  #   #   word2 = sentence[i+1]
      
  #   #   Q.add(word1)
  #   #   Q.add(word2)
      
  #   #   if word1 not in delta:
  #   #     delta[word1] = {'-': {word2}}
  #   #   else:
  #   #     delta[word1]['-'].add(word2)

  if initialState is None:
    initialState = '0'
  if f_set is None:
    F = {}
  else:
    F = f_set

  automata = NFA(Q, sigma, delta, initialState, F)
  automata.view(image_name)
  
  ## Delete .gv file
  if os.path.isfile(f"{image_name}.gv"):
    os.remove(f"{image_name}.gv")
  
  ## Rename .gv.png to .png
  if os.path.isfile(f"{image_name}.gv.png"):
    os.rename(f"{image_name}.gv.png", f"{image_name}.png")