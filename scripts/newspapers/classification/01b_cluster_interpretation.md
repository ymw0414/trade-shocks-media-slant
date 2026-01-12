# 01b Cluster Interpretation Log
Purpose: Conservative identification of economic-related clusters
Data: Pre-1990 newspaper articles (1986–1989)
Method: TF-IDF (1–2 grams) + KMeans (K=10)

----------------------------------------------------------------
GENERAL PRINCIPLE
----------------------------------------------------------------
- We label clusters, not individual articles.
- Only clusters that are unambiguously economic are retained.
- Ambiguous clusters are excluded from supervised training.
- This document records all human judgments for transparency.

----------------------------------------------------------------
CLUSTER-BY-CLUSTER INTERPRETATION
----------------------------------------------------------------

Cluster 0
Top terms:
- term1, term2, term3, ...

Interpretation:
- [ ] ECON
- [ ] NON-ECON
- [ ] AMBIGUOUS (DROP)

Justification:
- One or two sentences explaining why.
- Focus on dominant content, not edge cases.

-------------------------------------------------------------

Cluster 1
Top terms:
- ...

Interpretation:
- [ ] ECON
- [ ] NON-ECON
- [ ] AMBIGUOUS (DROP)

Justification:
- ...

-------------------------------------------------------------

Cluster 2
Top terms:
- ...

Interpretation:
- [ ] ECON
- [ ] NON-ECON
- [ ] AMBIGUOUS (DROP)

Justification:
- ...

-------------------------------------------------------------

(Repeat for all clusters)

----------------------------------------------------------------
SUMMARY
----------------------------------------------------------------
Clusters labeled ECON:
- Cluster X
- Cluster Y

Clusters labeled NON-ECON:
- Cluster A
- Cluster B

Clusters dropped as AMBIGUOUS:
- Cluster C
- Cluster D

Rationale for exclusion:
- Mixed political/economic framing
- Event-driven coverage
- Editorial/opinion content
