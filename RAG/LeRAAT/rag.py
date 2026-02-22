# ======================
# 인덱스 로드
# ======================
import os
import faiss
import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain.docstore.document import Document

# 임베딩 모델 (⚠️ 경고는 나오지만 우선 사용)
embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")

# 경로
db_dir = "data_new/new_vector_DB/faiss_e5_large_v2"

# FAISS 인덱스 로드
faiss_index = faiss.read_index(os.path.join(db_dir, "index.faiss"))

# 메타데이터 로드 (.pkl)
with open(os.path.join(db_dir, "index.pkl"), "rb") as f:
    meta_list = pickle.load(f)

# docstore + id 매핑 구성
docstore_dict = {}
index_to_docstore_id = {}

# for i, meta in enumerate(meta_list):
#     doc = Document(page_content=meta.get("text", ""), metadata=meta)
#     doc_id = str(i)
#     docstore_dict[doc_id] = doc
#     index_to_docstore_id[i] = doc_id

for i, meta in enumerate(meta_list):
    doc = Document(page_content=meta.get("text", ""), metadata=meta)
    doc_id = str(i)
    docstore_dict[doc_id] = doc
    index_to_docstore_id[int(i)] = doc_id   # np.int64 → int 로 맞춤

docstore = InMemoryDocstore(docstore_dict)

# 최종 벡터스토어
db = FAISS(
    embedding_function=embeddings,
    index=faiss_index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id
)

retriever = db.as_retriever(search_kwargs={"k": 3})

# ======================
# RAG 체인 정의
# ======================
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

llm = Ollama(
    model="llama3:8b",
    base_url="http://localhost:11434",
    temperature=0.2
)

prompt_template = """You are an expert aviation assistant.
Use the context to answer the question step by step.
Answer ONLY using the given context.
If not found, say you don't know.

Context:
{context}

Question:
{question}

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# ======================
# 질의 실행 함수
# ======================
def ask_question(query: str):
    result = qa_chain.invoke({"query": query})
    answer = result["result"]
    sources = result["source_documents"]

    print("\n=== Answer ===\n")
    print(answer)

    print("\n=== Sources ===\n")
    if not sources:
        print("⚠️ No source documents available.")
    else:
        for s in sources:
            file_name = s.metadata.get("source_file", "Unknown_File")
            title = s.metadata.get("title", "Unknown_Title")
            snippet = s.page_content[:100]  # 필요하면 자르기
            print(f"[FILE] {file_name}\n[TITLE] {title}\nSnippet: {snippet}\n")

    return answer

# ======================
# 예시 실행
# ======================
if __name__ == "__main__":
    query = input("질문을 입력하세요: ")
    ask_question(query)

