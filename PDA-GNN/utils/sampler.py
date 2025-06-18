import numpy as np
def sample_negatives(pos_src, nodes, existing, max_tries=10):
    neg = []
    node_set = set(nodes)
    for s in pos_src:
        for _ in range(max_tries):
            neg_i = np.random.choice(list(node_set))
            if (s.item(), neg_i) not in existing and neg_i != s.item():
                break
        else:
            neg_i = (s.item() + 1) % len(node_set)
            while neg_i == s.item():
                neg_i = (neg_i + 1) % len(node_set)
        neg.append(neg_i)
    return neg