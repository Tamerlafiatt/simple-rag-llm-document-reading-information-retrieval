from pathlib import Path

import faiss
import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import chat
from sentence_transformers import SentenceTransformer


class SimpleRAG:
    def __init__(
        self,
        pdf_path,
        embedding_model_name="BAAI/bge-small-en-v1.5",
        llm_model="qwen3:4b",
        chunk_size=1000,
        chunk_overlap=200,
    ):
        self.pdf_path = Path(pdf_path)
        self.source_name = self.pdf_path.name
        self.llm_model = llm_model

        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.chunks = self._load_and_chunk()
        self.embeddings = self.embedding_model.encode(
            [chunk.page_content for chunk in self.chunks],
            normalize_embeddings=True,
        ).astype("float32")

        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def _load_and_chunk(self):
        pdf = pymupdf.open(self.pdf_path)

        documents = [
            Document(
                page_content=page.get_text(),
                metadata={
                    "page": page_number + 1,
                    "source": self.source_name,
                },
            )
            for page_number, page in enumerate(pdf)
        ]

        return self.splitter.split_documents(documents)

    def retrieve(self, query, k=5):
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, chunk_index in zip(scores[0], indices[0]):
            chunk = self.chunks[chunk_index]
            results.append(
                {
                    "text": chunk.page_content,
                    "page": chunk.metadata["page"],
                    "source": chunk.metadata["source"],
                    "score": float(score),
                }
            )

        return results

    def answer(self, question, k=3):
        results = self.retrieve(question, k=k)

        context = "\n\n".join(
            f"[Source {i}: {result['source']}, page {result['page']}]\n"
            f"{result['text']}"
            for i, result in enumerate(results, start=1)
        )

        prompt = f"""
You are a question-answering assistant.

Use only the provided context to answer the question.
If the answer is not contained in the context, say:
"I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

        response = chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
        )

        return {
            "answer": response.message.content,
            "sources": [
                {
                    "source": result["source"],
                    "page": result["page"],
                    "score": result["score"],
                }
                for result in results
            ],
        }


if __name__ == "__main__":
    rag = SimpleRAG("data/documents/document.pdf")
    result = rag.answer("What is this document about?")

    print(result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(
            f"- {source['source']}, page {source['page']} "
            f"(similarity={source['score']:.3f})"
        )
