"""
Module to consolidate the BBC News Summary dataset into a single Excel file.
"""
import os
from tqdm import tqdm
import pandas as pd

ROOT_DIR = "/Users/rhafid/Documents/thesis/BBC News Summary"

SUB_DIR = "business"

articles = []
summaries = []

with tqdm(
    total=len(os.listdir(os.path.join(ROOT_DIR, "News Articles", SUB_DIR)))
) as pbar:
    for file in os.listdir(os.path.join(ROOT_DIR, "News Articles", SUB_DIR)):
        with open(
            os.path.join(ROOT_DIR, "News Articles", SUB_DIR, file),
            "r",
            encoding="utf-8",
        ) as f:
            articles.append(f.read())

with tqdm(total=len(os.listdir(os.path.join(ROOT_DIR, "Summaries", SUB_DIR)))) as pbar:
    for file in os.listdir(os.path.join(ROOT_DIR, "Summaries", SUB_DIR)):
        with open(
            os.path.join(ROOT_DIR, "Summaries", SUB_DIR, file), "r", encoding="utf-8"
        ) as f:
            summaries.append(f.read())

df = pd.DataFrame({"Article": articles, "Summary": summaries})
df.to_excel("business_articles.xlsx", index=False)
