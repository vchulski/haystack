from laserembeddings import Laser
from scipy.spatial import distance as sc_distance

from haystack.database.base import Document


class LaserRanker:
    """
    Laser ranker for reranking documents using the laser embeddings:
    (https://github.com/yannvgn/laserembeddings)

    With the ranker, you can:
     - directly get new scores of the documents via rerank()
    """

    def __init__(self,):
        """ Initialize laser embedder. """

        self.model = Laser()

    def rerank(self, query: str, documents: [Document], top_k: int = 5):
        """
        Use Laser embedder to find top k most relevant passages to return in the supplied list of Document.

        Returns dictionaty containing initial query,

        :param query: query string
        :param documents: list of Document in which to search for the answer
        :param top_k: the maximum number of answers to return
        :return: dict containing query and top k documents

        """

        ranked_documents = []
        corpus = [doc.text for doc in documents]
        # here in case we have already encoded vectors there is no need to encode corpus once again
        corpus_embeddings = self.model.embed_sentences(corpus, lang="en")
        query_embedding = self.model.embed_sentences([query], lang="en")
        distances = sc_distance.cdist(query_embedding, corpus_embeddings, "cosine")[0]
        closests = zip(range(len(distances)), distances)
        closests = sorted(closests, key=lambda x: x[1])
        for (
            idx,
            distance,
        ) in closests:  # [:top_k] - move it down, but maybe better here?
            passage = corpus[idx]  # .strip()
            score = 1 - distance

            # finding doc by it's text - maybe there is a better way?
            for doc in list(filter(lambda doc: doc.text == passage, documents)):
                if doc:
                    assert doc.text == passage
                    doc["score"] = score
                    ranked_documents.append(doc)

        # ranked_documents = sorted(
        #     ranked_documents, key=lambda k: k["score"], reverse=True
        # )
        ranked_documents = ranked_documents[:top_k]

        results = {"query": query, "top_k": top_k, "ranked_documents": ranked_documents}

        return results
