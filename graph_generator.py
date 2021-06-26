import json
import datetime
import networkx as nx
import itertools as it
import matplotlib.pyplot as plt
from typing import List, Dict

Paper = Dict[str,str]

def filter_by_category(papers : List[Paper], category : str) -> List[Dict[str,str]]:
    filtered = []
    for p in papers:
        categories = p['categories'].split(' ')
        if category in categories:
            filtered.append(p)

    return filtered

def load_arxiv(dir : str, limit : int = 0, category : str = None) -> List[Paper]:
    inputs = open(dir).readlines()
    papers = []

    for p in inputs:
        papers.append(json.loads(p))

    if category is not None:
        papers = filter_by_category(papers, category)
    
    if limit == 0:
        return papers

    return papers[:limit]

def main():
    start = datetime.datetime.now()
    arxiv_graph = nx.DiGraph()
    papers = load_arxiv("./Data/arxiv-metadata-oai-snapshot.json", 10750, "hep-th")

    for p in papers:
        #parse author names, format as singular strings
        authors_parsed = p['authors_parsed']
        authors = list(map(lambda ns: '-'.join(ns), authors_parsed))
        #add to graph
        if len(authors) > 1:
            arxiv_graph.add_nodes_from(authors)
            for (a,b) in it.permutations(authors,2):
                if arxiv_graph.has_edge(a,b):
                    arxiv_graph[a][b]['weight'] += 1
                else:
                    arxiv_graph.add_edge(a,b,weight=1)
    
    end = datetime.datetime.now()

    print("Generated graph in {} seconds".format((end-start).seconds))
    print("Printing graph with {} edges between {} nodes.".format(arxiv_graph.size(), len(arxiv_graph.nodes())))
    nx.draw(arxiv_graph)
    plt.show()

main()