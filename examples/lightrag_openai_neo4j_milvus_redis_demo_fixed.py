import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_embed
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

# 导入补丁版本的LightRAG
import sys
sys.path.append('/home/shao/SR_Assistance/KAG')
from LightRAG_patch import LightRAGPatch

# WorkingDir
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR = os.path.join(ROOT_DIR, "myKG")
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
print(f"WorkingDir: {WORKING_DIR}")

# 其余初始化代码不变
llm_model_func = openai_complete_if_cache
embedding_func = EmbeddingFunc(ollama_embed)

async def initialize_rag():
    # 使用补丁版本的LightRAG
    rag = LightRAGPatch(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        llm_model_max_token_size=32768,
        embedding_func=embedding_func,
        chunk_token_size=512,
        chunk_overlap_token_size=256,
        kv_storage="RedisKVStorage",
        graph_storage="Neo4JStorage",
        vector_storage="MilvusVectorDBStorage",
        doc_status_storage="JsonDocStatusStorage",
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def main():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())

    with open("./book.txt", "r", encoding="utf-8") as f:
        # 使用带有folder_id参数的insert方法
        rag.insert(f.read(), folder_id=0)

    # 原始查询代码不变
    query_param = QueryParam(
        query="What is the capital of France?",
        top_k=5,
        max_token_size=512,
    )
    result = rag.query(query_param)
    print(result)

if __name__ == "__main__":
    main()