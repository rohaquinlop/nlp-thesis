import os
from tqdm import tqdm
import pandas as pd

root_dir = '/Users/rhafid/Documents/thesis/BBC News Summary'

sub_dir = 'business'

articles = []
summaries = []

with tqdm(total=len(os.listdir(os.path.join(root_dir, 'News Articles', sub_dir)))) as pbar:
  for file in os.listdir(os.path.join(root_dir, 'News Articles', sub_dir)):
    with open(os.path.join(root_dir, 'News Articles', sub_dir, file), 'r') as f:
      articles.append(f.read())

with tqdm(total=len(os.listdir(os.path.join(root_dir, 'Summaries', sub_dir)))) as pbar:
  for file in os.listdir(os.path.join(root_dir, 'Summaries', sub_dir)):
    with open(os.path.join(root_dir, 'Summaries', sub_dir, file), 'r') as f:
      summaries.append(f.read())

df = pd.DataFrame({'Article': articles, 'Summary': summaries})
df.to_excel('business_articles.xlsx', index=False)
