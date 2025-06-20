import numpy as np
def sample_negatives(pos_src, nodes, existing):
    neg = []
    for s in pos_src:
        neg_i = np.random.choice(nodes)
        while (s.item(), neg_i) in existing:
            neg_i = np.random.choice(nodes)
        neg.append(neg_i)
    return neg