import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from pinecone import Pinecone

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# 1. Cargar los dos PDFs
loader_issues = PyPDFLoader("data/operations issues logiscore.pdf")
loader_root_cause = PyPDFLoader("data/root cause logiccore.pdf")

documentos_issues = loader_issues.load()
documentos_root_cause = loader_root_cause.load()

# 2. Etiquetar cada documento según su tipo (esto va aquí 👇)
for doc in documentos_issues:
    doc.metadata["tipo"] = "incidente"

for doc in documentos_root_cause:
    doc.metadata["tipo"] = "causa_raiz"

# 3. Combinar ambos documentos en una sola lista
todos_los_documentos = documentos_issues + documentos_root_cause

print(f"✅ Total de documentos combinados: {len(todos_los_documentos)}")

# 4. Dividir en fragmentos (chunks) para mejor búsqueda semántica
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
chunks = splitter.split_documents(todos_los_documentos)

print(f"✅ Se generaron {len(chunks)} fragmentos de los 2 PDFs")

# 5. Crear embeddings con Cohere
embeddings = CohereEmbeddings(
    model="embed-multilingual-v3.0",
    cohere_api_key=COHERE_API_KEY
)

# 6. Conectar a Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# 5. Crear embeddings con Cohere
embeddings = CohereEmbeddings(
    model="embed-multilingual-v3.0",
    cohere_api_key=COHERE_API_KEY
)
print("✅ Embeddings de Cohere creados correctamente")

# 6. Conectar a Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
print("✅ Conexión a Pinecone establecida")
