#! /usr/bin/env python
# coding=utf-8
"""
Cleaning of the text data parsed from html-s.
"""
import re

data_path = "data/total6.txt"
out_path = "data/total6_edited.txt"

with open(data_path) as f:
    text = f.read()

# Tab
text = re.sub(re.compile(r"\t"), " ", text)

# Apostrophe
text = re.sub(re.compile(r"‘"), "'", text)
text = re.sub(re.compile(r"\\'"), "'", text)
text = re.sub(re.compile(r" 's(?=\W)"), "'s", text)  # fix such case as "<word> 's"

with open(out_path, "w") as f:
    f.write(text)
