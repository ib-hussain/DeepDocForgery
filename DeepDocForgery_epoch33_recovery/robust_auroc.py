"""Candidate pure-Python AUROC implementation for integration into metrics.py.
Pairwise remove invalid scores; track their count separately in the caller.
Not installed automatically. Review metrics.py's existing metric contract first.
"""
import math


def binary_auroc(scores, labels):
    if len(scores)!=len(labels):
        raise ValueError('Scores and labels length mismatch')
    paired=[]
    for score,label in zip(scores,labels):
        score=float(score)
        if not math.isfinite(score):
            continue
        if int(label) not in (0,1):
            raise ValueError('Labels must be binary')
        paired.append((score,int(label)))
    positives=sum(label for _,label in paired)
    negatives=len(paired)-positives
    if not positives or not negatives:
        return float('nan')  # undefined; caller must handle before JSON/ranking
    paired.sort(key=lambda x:x[0])
    rank=1
    positive_rank_sum=0.0
    i=0
    while i<len(paired):
        j=i+1
        while j<len(paired) and paired[j][0]==paired[i][0]:
            j+=1
        average_rank=(rank+(rank+j-i-1))/2.0
        positive_rank_sum+=average_rank*sum(label for _,label in paired[i:j])
        rank+=j-i
        i=j
    return (positive_rank_sum-positives*(positives+1)/2.0)/(positives*negatives)
