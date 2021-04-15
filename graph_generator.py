import json
from typing import List, Dict

Paper = Dict[str,str]

def load_arxiv(dir : str, limit : int = 0) -> List[Paper]:
    inputs = open(dir).readlines()
    papers = []
    
    if limit == 0:
        for p in inputs:
            papers.append(json.loads(p))
    else:
        for (_,p) in zip(range(limit), inputs):
            papers.append(json.loads(p))
    
    return papers

def main():
    papers = load_arxiv("./Data/arxiv-metadata-oai-snapshot.json", 100)

    for p in papers:
        print(p['title'])

main()