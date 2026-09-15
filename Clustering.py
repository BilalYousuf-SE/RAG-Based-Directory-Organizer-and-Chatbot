from sklearn.cluster import AgglomerativeClustering
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from difflib import SequenceMatcher
import re

# A filename needs at least this many alphabetic characters (ignoring the
# extension) before we trust string-similarity on it at all. Filenames
# like "3337.pdf", "4.pdf", "INV-100.pdf" are effectively random/ID-like
# strings -- SequenceMatcher will still produce a similarity number for
# them, but it reflects coincidental character overlap, not real topical
# similarity, and title_similarity carries real weight in the final score.
MIN_ALPHA_CHARS_FOR_TITLE_SIM = 4


def _has_meaningful_title(file_name):
    stem = file_name.rsplit(".", 1)[0]
    alpha_chars = re.sub(r"[^a-zA-Z]", "", stem)
    return len(alpha_chars) >= MIN_ALPHA_CHARS_FOR_TITLE_SIM

def calculate_document_relationship(
    doc1,
    doc2,
    embedding_weight=1,
    keyword_weight=0,
    title_weight=0
):
    """
    Returns a score between 0 and 1 indicating
    how related two documents are.
    """

    # -------------------------
    # Embedding Similarity
    # -------------------------
    embedding_similarity = cosine_similarity(
        [doc1["Document_Embedding"]],
        [doc2["Document_Embedding"]]
    )[0][0]

    # -------------------------
    # Keyword Similarity
    # -------------------------
    keywords1 = set(
        keyword.lower()
        for keyword in doc1.get("keywords", [])
    )

    keywords2 = set(
        keyword.lower()
        for keyword in doc2.get("keywords", [])
    )

    if keywords1 or keywords2:
        keyword_similarity = (
            len(keywords1.intersection(keywords2))
            /
            len(keywords1.union(keywords2))
        )
    else:
        keyword_similarity = 0

    # -------------------------
    # Title Similarity
    # -------------------------
    # Only trust this signal if BOTH filenames look like real words
    # (not IDs/serials/invoice numbers). Otherwise drop it and redistribute
    # its weight to the two signals that are actually about content.
    title_is_meaningful = (
        _has_meaningful_title(doc1["file_name"])
        and _has_meaningful_title(doc2["file_name"])
    )

    if title_is_meaningful:
        title_similarity = SequenceMatcher(
            None,
            doc1["file_name"].lower(),
            doc2["file_name"].lower()
        ).ratio()
        effective_title_weight = title_weight
    else:
        title_similarity = 0
        effective_title_weight = 0

    # Redistribute any weight not spent on title similarity, proportionally,
    # across embedding and keyword weight so the score still sums to 1.0.
    dropped_weight = title_weight - effective_title_weight
    total_remaining = embedding_weight + keyword_weight
    if dropped_weight > 0 and total_remaining > 0:
        embedding_weight += dropped_weight * (embedding_weight / total_remaining)
        keyword_weight += dropped_weight * (keyword_weight / total_remaining)

    # -------------------------
    # Final Weighted Score
    # -------------------------
    score = (
        embedding_similarity * embedding_weight
        + keyword_similarity * keyword_weight
        + title_similarity * effective_title_weight
    )

    return round(float(score), 4)



def _document_embedding_matrix(documents):

    return np.array(
        [document["Document_Embedding"] for document in documents]
    )


def find_duplicate_groups(documents, threshold=0.999):
    """
    Groups documents whose Document_Embedding cosine similarity is
    effectively 1 (identical content) using union-find over every pair
    scoring >= threshold. 0.999 rather than a strict 1.0 is used because
    floating point cosine similarity on two independently-averaged
    embeddings almost never lands on exactly 1.0 even for genuinely
    identical text.

    Returns a list of groups, each a sorted list of document ids
    (size >= 2). Documents with no near-1.0 match to anything are not
    included in any group.
    """

    if len(documents) < 2:
        return []

    similarity = cosine_similarity(
        _document_embedding_matrix(documents)
    )

    parent = list(range(len(documents)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(len(documents)):
        for j in range(i + 1, len(documents)):
            if similarity[i][j] >= threshold:
                union(i, j)

    groups = {}

    for i, document in enumerate(documents):
        root = find(i)
        groups.setdefault(root, []).append(document["id"])

    return [
        sorted(group)
        for group in groups.values()
        if len(group) > 1
    ]


def find_version_pairs(documents, low=0.90, high=0.999):
    """
    Returns (doc_id_a, doc_id_b, score) tuples for every pair of documents
    whose embedding cosine similarity falls in [low, high) -- similar
    enough to likely be different drafts/versions of the same document,
    but not near-identical (those are caught by find_duplicate_groups
    instead, using the same `high` cutoff as its default threshold).
    """

    if len(documents) < 2:
        return []

    similarity = cosine_similarity(
        _document_embedding_matrix(documents)
    )

    pairs = []

    for i in range(len(documents)):
        for j in range(i + 1, len(documents)):

            score = similarity[i][j]

            if low <= score < high:
                pairs.append((
                    documents[i]["id"],
                    documents[j]["id"],
                    round(float(score), 4)
                ))

    return pairs


def group_version_pairs(version_pairs):
    """
    Groups the (doc_id_a, doc_id_b, score) tuples from find_version_pairs
    into connected chains, using the same union-find approach
    find_duplicate_groups uses for exact duplicates.

    A document can appear in more than one version pair (e.g. v1-v2 and
    v2-v3 both clear the threshold even though v1-v3 doesn't), so grouping
    by connectivity -- rather than treating each pair independently --
    keeps every version of the same document together as one chain, ready
    to be ordered by creation date and labeled v1, v2, v3...

    Returns a list of groups, each a sorted list of document ids
    (size >= 2).
    """

    if not version_pairs:
        return []

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for doc_id_a, doc_id_b, _ in version_pairs:
        union(doc_id_a, doc_id_b)

    groups = {}

    for doc_id_a, doc_id_b, _ in version_pairs:
        root = find(doc_id_a)
        groups.setdefault(root, set()).update((doc_id_a, doc_id_b))

    return [
        sorted(group)
        for group in groups.values()
    ]


def build_document_graph(relationships, min_score=0.30):

    G = nx.Graph()

    for document in relationships:

        source_id = document["id"]

        G.add_node(source_id)

        for relationship in document["Relationships"]:

            target_id = relationship["document_id"]
            score = relationship["score"]

            if score >= min_score:

                G.add_edge(
                    source_id,
                    target_id,
                    weight=score
                )

    return G


def find_communities_connected(graph):
    """
    Original method: connected components.

    WARNING -- this is mathematically equivalent to single-linkage
    clustering, and it chains: if A-B and B-C both clear min_score, A and
    C land in the same community even if A-C would score near zero. One
    weak "bridge" document merges two otherwise-unrelated topic groups
    into one giant blob. Kept here for comparison against
    find_communities(), which doesn't have this problem.
    """

    communities = list(
        nx.connected_components(graph)
    )

    return [
        sorted(list(community))
        for community in communities
    ]


def find_communities(graph):
    """
    Community detection via modularity maximization (Clauset-Newman-Moore
    greedy algorithm), used as the default in place of connected components.

    Unlike connected components, this doesn't chain on a single weak edge:
    it looks for groups of nodes that are more densely connected to each
    other than to the rest of the graph, so one bridging document with a
    borderline score to two different topic clusters won't necessarily
    merge those clusters -- it'll only get pulled in if it's genuinely
    more connected to one side.
    """

    if graph.number_of_edges() == 0:
        # No edges at all -- every node is its own community.
        return [[node] for node in sorted(graph.nodes())]

    communities = nx.algorithms.community.greedy_modularity_communities(
        graph,
        weight="weight"
    )

    return [
        sorted(list(community))
        for community in communities
    ]


def split_oversized_communities(
    communities,
    documents,
    max_community_size=8,
    distance_threshold=0.35
):
    """
    Safety net that runs AFTER find_communities(). Modularity maximization
    picks the graph partition that's best *on average*, which usually
    means it won't split a community further even if that community is
    still a loose mix of two document types -- as long as merging them
    scores better overall than not. This only matters once relationship
    scoring is already keyword-aware (see create_relationship_scores in
    main.py); it's a backstop for whatever still slips through, not the
    primary fix for mixed communities.

    Any community with more than max_community_size documents gets
    re-clustered on its own using agglomerative clustering over cosine
    distance between Document_Embeddings. distance_threshold controls how
    tight a subgroup has to be to split off -- a genuinely single-topic
    community (all pairwise distances under the threshold) comes back as
    one cluster unchanged; a mixed one gets split into tighter subgroups.
    Communities at or under max_community_size are left as-is.
    """

    document_lookup = {
        document["id"]: document
        for document in documents
    }

    refined_communities = []

    for community in communities:

        if len(community) <= max_community_size:
            refined_communities.append(community)
            continue

        embeddings = np.array([
            document_lookup[document_id]["Document_Embedding"]
            for document_id in community
        ])

        try:
            sub_clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                metric="cosine",
                linkage="average"
            )
            labels = sub_clustering.fit_predict(embeddings)

        except TypeError:
            # scikit-learn < 1.2 used `affinity` instead of `metric`.
            sub_clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                affinity="cosine",
                linkage="average"
            )
            labels = sub_clustering.fit_predict(embeddings)

        sub_groups = {}

        for document_id, label in zip(community, labels):
            sub_groups.setdefault(label, []).append(document_id)

        refined_communities.extend(
            sorted(group)
            for group in sub_groups.values()
        )

    return refined_communities


def print_communities(communities, documents):

    document_lookup = {
        document["id"]: document["file_name"]
        for document in documents
    }

    for index, community in enumerate(communities, start=1):

        print(f"\n{'=' * 50}")
        print(f"COMMUNITY {index}")
        print(f"{'=' * 50}")

        for document_id in community:

            print(
                f"[{document_id}] "
                f"{document_lookup[document_id]}"
            )