# tenant_aware/lightrag_aware.py
import asyncio
import os
from dataclasses import asdict, field
from typing import Any, AsyncIterator, Callable, Iterator, final, List, Union

# 导入原始 LightRAG 和所需类型
from lightrag.lightrag import LightRAG
from lightrag.base import QueryParam, StoragesStatus, BaseKVStorage, BaseVectorStorage, BaseGraphStorage, DocStatusStorage, DocStatus, DocProcessingStatus
from lightrag.utils import logger, always_get_an_event_loop, compute_mdhash_id ,split_string_by_multi_markers # Import compute_mdhash_id
from datetime import datetime # Import datetime
from lightrag.types import KnowledgeGraph
from lightrag.operate import extract_entities, kg_query, naive_query, mix_kg_vector_query # Assuming these are modified
from .redis_impl_aware import TenantAwareRedisKVStorage
from .milvus_impl_aware import TenantAwareMilvusVectorDBStorage
from .neo4j_impl_aware import TenantAwareNeo4JStorage
try:
    from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage
except ImportError:
    class JsonDocStatusStorage: pass # Dummy
from lightrag.namespace import make_namespace, NameSpace
from typing import Any, AsyncIterator, Callable, Iterator, final, List, Union, Dict, Tuple, Optional # Add Dict, Tuple, Optional
import pandas as pd # For export
import csv # For export
import io # For export
from lightrag.prompt import GRAPH_FIELD_SEP # For merging attributes
from lightrag.kg.shared_storage import get_namespace_data, get_pipeline_status_lock, initialize_share_data


@final
# @dataclass # LightRAG 已经是 dataclass, 子类不需要重复
class MyTenantAwareLightRAG(LightRAG):
    """
    Tenant-aware version of LightRAG that uses folder_id for data isolation.
    """

    # .重写初始化存储逻辑
    async def initialize_storages(self):
        """Asynchronously initialize the tenant-aware storages."""
        # 防止重复初始化
        if self._storages_status != StoragesStatus.CREATED:
             logger.warning(f"Storages already initialized or in incorrect state: {self._storages_status}")
             return

        logger.info("Initializing Tenant-Aware Storages...")
        global_config = asdict(self) # 获取当前 RAG 实例的配置

        # --- Instantiate Tenant-Aware Adapters ---
        # 注意: 确保将正确的 global_config 和 embedding_func 传递给适配器
        # KV Stores (Redis for cache, full_docs, text_chunks)
        self.llm_response_cache = TenantAwareRedisKVStorage(
            namespace=make_namespace(self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE),
            global_config=global_config,
            embedding_func=self.embedding_func
        )
        self.full_docs = TenantAwareRedisKVStorage( # 假设文档和块也用 Redis
            namespace=make_namespace(self.namespace_prefix, NameSpace.KV_STORE_FULL_DOCS),
            global_config=global_config,
            embedding_func=self.embedding_func
        )
        self.text_chunks = TenantAwareRedisKVStorage(
             namespace=make_namespace(self.namespace_prefix, NameSpace.KV_STORE_TEXT_CHUNKS),
             global_config=global_config,
             embedding_func=self.embedding_func
        )

        # Vector Stores (Milvus)
        self.entities_vdb = TenantAwareMilvusVectorDBStorage(
             namespace=make_namespace(self.namespace_prefix, NameSpace.VECTOR_STORE_ENTITIES),
             global_config=global_config,
             embedding_func=self.embedding_func,
             meta_fields={"entity_name", "source_id", "content", "file_path"}, # 保留元字段
             **self.vector_db_storage_cls_kwargs # 传递额外参数
        )
        self.relationships_vdb = TenantAwareMilvusVectorDBStorage(
             namespace=make_namespace(self.namespace_prefix, NameSpace.VECTOR_STORE_RELATIONSHIPS),
             global_config=global_config,
             embedding_func=self.embedding_func,
             meta_fields={"src_id", "tgt_id", "source_id", "content", "file_path"},
             **self.vector_db_storage_cls_kwargs
        )
        self.chunks_vdb = TenantAwareMilvusVectorDBStorage(
             namespace=make_namespace(self.namespace_prefix, NameSpace.VECTOR_STORE_CHUNKS),
             global_config=global_config,
             embedding_func=self.embedding_func,
             meta_fields={"full_doc_id", "content", "file_path"},
             **self.vector_db_storage_cls_kwargs
        )

        # Graph Store (Neo4j)
        self.chunk_entity_relation_graph = TenantAwareNeo4JStorage(
             namespace=make_namespace(self.namespace_prefix, NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION),
             global_config=global_config,
             embedding_func=self.embedding_func
        )

        # Doc Status Store (Using original Json implementation as decided)
        # Ensure the class exists at the expected path or adjust import
        try:
            from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage
            self.doc_status = JsonDocStatusStorage(
                namespace=make_namespace(self.namespace_prefix, NameSpace.DOC_STATUS),
                global_config=global_config,
                embedding_func=None, # Doc status doesn't need embeddings
            )
        except ImportError:
             logger.error("Could not import JsonDocStatusStorage. Check path.")
             raise

        # --- Initialize all storage instances ---
        storages_to_init: List[Union[BaseKVStorage, BaseVectorStorage, BaseGraphStorage, DocStatusStorage]] = [
             self.llm_response_cache, self.full_docs, self.text_chunks,
             self.entities_vdb, self.relationships_vdb, self.chunks_vdb,
             self.chunk_entity_relation_graph, self.doc_status,
        ]

        tasks = [storage.initialize() for storage in storages_to_init if storage]
        await asyncio.gather(*tasks)

        self._storages_status = StoragesStatus.INITIALIZED
        logger.info("Tenant-Aware Storages Initialized.")


    # --- 重写需要 folder_id 的公共方法 ---

    def insert(
        self,
        input: str | list[str],
        folder_id: int, # <--- 新增参数
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        ids: str | list[str] | None = None,
        file_paths: str | list[str] | None = None,
    ) -> None:
        """Sync Insert documents into a specific folder."""
        if folder_id is None:
            raise ValueError("folder_id must be provided for insert operation")
        loop = always_get_an_event_loop()
        loop.run_until_complete(
            self.ainsert(
                input, folder_id, split_by_character, split_by_character_only, ids, file_paths
            )
        )

    async def ainsert(
        self,
        input: str | list[str],
        folder_id: int, # <--- 新增参数
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        ids: str | list[str] | None = None,
        file_paths: str | list[str] | None = None,
    ) -> None:
        """Async Insert documents into a specific folder."""
        if folder_id is None:
            raise ValueError("folder_id must be provided for ainsert operation")
        # enqueue 不需要 folder_id，因为它操作的是全局 DocStatus
        await self.apipeline_enqueue_documents(input, ids, file_paths)
        # process 需要 folder_id 传递给内部的 extract_entities
        await self.apipeline_process_enqueue_documents(
            folder_id, split_by_character, split_by_character_only # <--- 传递 folder_id
        )

    # 修改处理队列的方法以接受并传递 folder_id
    async def apipeline_process_enqueue_documents(
        self,
        folder_id: int, # <--- folder_id 参数
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
    ) -> None:
        """
        Processes enqueued documents, passing folder_id to the extraction step.
        """
        if folder_id is None:
             raise ValueError("folder_id is required for apipeline_process_enqueue_documents")

        # 从共享存储获取 pipeline_status 和 lock
        pipeline_status = await get_namespace_data("pipeline_status")
        pipeline_status_lock = get_pipeline_status_lock()

        async with pipeline_status_lock:
             # 检查是否已有进程在运行
             if pipeline_status.get("busy", False):
                  pipeline_status["request_pending"] = True
                  logger.info("Another process is busy. Request queued.")
                  return

             # 获取待处理文档 (逻辑不变)
             processing_docs, failed_docs, pending_docs = await asyncio.gather(
                  self.doc_status.get_docs_by_status(DocStatus.PROCESSING),
                  self.doc_status.get_docs_by_status(DocStatus.FAILED),
                  self.doc_status.get_docs_by_status(DocStatus.PENDING),
             )
             to_process_docs: dict[str, DocProcessingStatus] = {
                  **processing_docs, **failed_docs, **pending_docs
             }
             if not to_process_docs:
                  logger.info("No documents to process")
                  return # 没有文档需要处理，直接返回

             # *** FIX 1: 更新 pipeline_status 字典，使用正确的 Python 语法 ***
             # 获取第一个文档的文件路径用于 job_name (如果存在)
             first_doc_tuple = next(iter(to_process_docs.items()), None)
             first_doc_path_short = "Unknown Job"
             if first_doc_tuple:
                 first_doc_path = getattr(first_doc_tuple[1], "file_path", "Unknown Path")
                 path_prefix = first_doc_path[:20] + ("..." if len(first_doc_path) > 20 else "")
                 first_doc_path_short = f"{path_prefix}[{len(to_process_docs)} files]"

             pipeline_status.update({
                 "busy": True,
                 "job_name": first_doc_path_short, # 使用简短路径作为任务名
                 "job_start": datetime.now().isoformat(),
                 "docs": len(to_process_docs),
                 "batchs": (len(to_process_docs) + self.max_parallel_insert - 1) // self.max_parallel_insert, # 计算批次数
                 "cur_batch": 0, # 初始化当前批次
                 "request_pending": False, # 清除挂起请求标志
                 "latest_message": f"Starting processing for {len(to_process_docs)} documents.",
                 # 清空历史消息列表 (确保 history_messages 是列表)
                 "history_messages": pipeline_status.get("history_messages", [])[:0] # 清空列表内容
             })
             # 确保 history_messages 字段存在且是列表
             if "history_messages" not in pipeline_status or not isinstance(pipeline_status["history_messages"], list):
                  pipeline_status["history_messages"] = []
             # 记录开始消息
             pipeline_status["history_messages"].append(pipeline_status["latest_message"])


        # --- 处理循环 ---
        try:
             while True: # 循环直到没有文档或没有挂起请求
                  # 检查是否还有文档 (可能在上一轮处理后变为空)
                  if not to_process_docs:
                       logger.info("No more documents in the current processing cycle.")
                       break

                  # 分批处理文档
                  docs_batches = [
                       list(to_process_docs.items())[i : i + self.max_parallel_insert]
                       for i in range(0, len(to_process_docs), self.max_parallel_insert)
                  ]
                  total_batches = len(docs_batches)
                  logger.info(f"Processing {len(to_process_docs)} document(s) in {total_batches} batches for folder {folder_id}")


                  # --- 内部函数: 处理单个文档 ---
                  async def process_document(
                       doc_id: str,
                       status_doc: DocProcessingStatus,
                       target_folder_id: int,
                       split_by_char: str | None,
                       split_by_char_only: bool,
                       p_status: dict,
                       p_lock: asyncio.Lock,
                  ) -> None:
                       current_time_iso = datetime.now().isoformat() # 获取当前时间一次
                       try:
                            file_path = getattr(status_doc, "file_path", "unknown_source")

                            # *** FIX 3: 调用 self.chunking_func 并传入正确的参数 ***
                            chunk_list = self.chunking_func(
                                status_doc.content,
                                split_by_char,            # 使用传入的参数
                                split_by_char_only,       # 使用传入的参数
                                self.chunk_overlap_token_size, # 从 self 获取配置
                                self.chunk_token_size,         # 从 self 获取配置
                                self.tiktoken_model_name,    # 从 self 获取配置
                            )
                            chunks: dict[str, Any] = {
                                 compute_mdhash_id(dp["content"], prefix="chunk-"): {
                                      **dp, # 包含 chunking_func 返回的所有字段 (content, tokens, chunk_order_index)
                                      "full_doc_id": doc_id,
                                      "file_path": file_path,
                                 } for dp in chunk_list # 使用正确的变量名
                            }

                            # 更新文档状态为 PROCESSING
                            # 保留原始内容、摘要、长度、创建时间，更新时间和状态
                            doc_status_update = {
                                 doc_id: {
                                      "status": DocStatus.PROCESSING,
                                      "chunks_count": len(chunks),
                                      "content": status_doc.content,
                                      "content_summary": status_doc.content_summary,
                                      "content_length": status_doc.content_length,
                                      "created_at": status_doc.created_at,
                                      "updated_at": current_time_iso, # 更新时间
                                      "file_path": file_path,
                                      "error": None # 清除之前的错误（如果有）
                                 }
                            }
                            await self.doc_status.upsert(doc_status_update) # 更新全局状态

                            # 并行处理存储和图构建 (传递 folder_id)
                            await asyncio.gather(
                                 self.chunks_vdb.upsert(chunks, folder_id=target_folder_id),
                                 self._process_entity_relation_graph(
                                      chunks, folder_id=target_folder_id,
                                      pipeline_status=p_status, pipeline_status_lock=p_lock
                                 ),
                                 self.full_docs.upsert({doc_id: {"content": status_doc.content, "file_path": file_path}}, folder_id=target_folder_id), # 在 full_docs 也存 file_path
                                 self.text_chunks.upsert(chunks, folder_id=target_folder_id),
                            )

                            # 更新文档状态为 PROCESSED
                            doc_status_update[doc_id]["status"] = DocStatus.PROCESSED
                            doc_status_update[doc_id]["updated_at"] = datetime.now().isoformat() # 再次更新时间
                            await self.doc_status.upsert(doc_status_update)

                       except Exception as e:
                            # *** FIX 2: 更新 doc_status 字典，使用正确的失败状态字段 ***
                            error_msg = f"Failed to process document {doc_id} for folder {target_folder_id}: {str(e)}"
                            logger.error(error_msg, exc_info=True) # 添加 exc_info=True 获取 traceback
                            # 更新 pipeline 状态
                            async with p_lock:
                                 p_status["latest_message"] = error_msg
                                 # 避免历史消息过多，可以限制长度
                                 if len(p_status.get("history_messages", [])) < 100:
                                     p_status.setdefault("history_messages", []).append(error_msg)

                            # 更新文档状态为 FAILED
                            # 保留原有信息，添加错误信息
                            failed_status_update = {
                                 doc_id: {
                                      "status": DocStatus.FAILED,
                                      "error": str(e), # 存储错误信息
                                      "chunks_count": status_doc.chunks_count, # 保留之前的块数（如果有）
                                      "content": status_doc.content,
                                      "content_summary": status_doc.content_summary,
                                      "content_length": status_doc.content_length,
                                      "created_at": status_doc.created_at,
                                      "updated_at": datetime.now().isoformat(), # 更新时间
                                      "file_path": getattr(status_doc, "file_path", "unknown_source"), # 保留文件路径
                                 }
                            }
                            await self.doc_status.upsert(failed_status_update) # 更新全局状态

                  # --- 在循环中处理批次时传递 folder_id ---
                  for batch_idx, docs_batch in enumerate(docs_batches):
                       current_batch_num = batch_idx + 1
                       # 更新 pipeline 状态
                       log_message = f"Starting batch {current_batch_num}/{total_batches} for folder {folder_id}..."
                       logger.info(log_message)
                       async with pipeline_status_lock:
                            pipeline_status["cur_batch"] = current_batch_num
                            pipeline_status["latest_message"] = log_message
                            if len(pipeline_status.get("history_messages", [])) < 100:
                                 pipeline_status.setdefault("history_messages", []).append(log_message)

                       # 创建并执行任务
                       doc_tasks = [
                            process_document(
                                 doc_id, status_doc,
                                 folder_id, # <--- 传递正确的 folder_id
                                 split_by_character, split_by_character_only,
                                 pipeline_status, pipeline_status_lock
                            ) for doc_id, status_doc in docs_batch
                       ]
                       await asyncio.gather(*doc_tasks)

                       # 每批处理完后调用 _insert_done
                       await self._insert_done(folder_id=folder_id)
                       # 更新 pipeline 状态
                       log_message = f"Completed batch {current_batch_num}/{total_batches} for folder {folder_id}."
                       logger.info(log_message)
                       async with pipeline_status_lock:
                           pipeline_status["latest_message"] = log_message
                           if len(pipeline_status.get("history_messages", [])) < 100:
                                pipeline_status.setdefault("history_messages", []).append(log_message)


                  # --- 检查挂起请求并获取新文档 ---
                  has_pending_request = False
                  async with pipeline_status_lock:
                       has_pending_request = pipeline_status.get("request_pending", False)
                       if has_pending_request:
                            pipeline_status["request_pending"] = False # 清除标志

                  if not has_pending_request:
                       logger.info(f"No pending request for folder {folder_id}. Finishing cycle.")
                       break # 没有挂起请求，结束当前处理周期

                  # 如果有挂起请求，重新获取待处理文档列表
                  logger.info(f"Processing additional documents due to pending request for folder {folder_id}")
                  processing_docs, failed_docs, pending_docs = await asyncio.gather(
                       self.doc_status.get_docs_by_status(DocStatus.PROCESSING),
                       self.doc_status.get_docs_by_status(DocStatus.FAILED),
                       self.doc_status.get_docs_by_status(DocStatus.PENDING),
                  )
                  to_process_docs = {**processing_docs, **failed_docs, **pending_docs}
                  # 更新 pipeline 状态中的文档总数和批次数
                  async with pipeline_status_lock:
                       pipeline_status["docs"] = len(to_process_docs)
                       pipeline_status["batchs"] = (len(to_process_docs) + self.max_parallel_insert - 1) // self.max_parallel_insert
                       pipeline_status["cur_batch"] = 0 # 重置当前批次

                  if not to_process_docs:
                       logger.info(f"No new documents found despite pending request for folder {folder_id}.")
                       # 可以在这里更新 pipeline_status 的消息
                       break # 没有新文档了，退出循环


        finally:
             # 最终重置 busy 状态
             final_log_message = f"Document processing pipeline cycle completed for folder {folder_id}"
             logger.info(final_log_message)
             async with pipeline_status_lock:
                  pipeline_status["busy"] = False
                  pipeline_status["latest_message"] = final_log_message
                  if len(pipeline_status.get("history_messages", [])) < 100:
                       pipeline_status.setdefault("history_messages", []).append(final_log_message)

    # 修改 _process_entity_relation_graph 以接受并传递 folder_id
    async def _process_entity_relation_graph(
        self, chunk: dict[str, Any], folder_id: int, # <--- 新增参数
        pipeline_status=None, pipeline_status_lock=None
    ) -> None:
        if folder_id is None:
             raise ValueError("_process_entity_relation_graph requires folder_id")
        try:
            # 调用 operate.py 中被修改过的 extract_entities
            await extract_entities(
                chunk,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                relationships_vdb=self.relationships_vdb,
                global_config=asdict(self),
                folder_id=folder_id, # <--- 传递 folder_id
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
                llm_response_cache=self.llm_response_cache,
            )
        except Exception as e:
            logger.error(f"Failed to extract entities/relationships for folder {folder_id}")
            raise e

    # 修改 _insert_done 以接受 folder_id (如果需要，例如用于 tenant-specific 刷新)
    async def _insert_done(self, folder_id: int = None) -> None: # <--- folder_id 可选
         # 原来的逻辑是调用所有存储的 index_done_callback
         # 对于 Neo4j, Milvus, Redis，这通常是无操作或全局刷新
         # 如果某个存储需要特定于租户的刷新，需要在这里处理
         tasks = [
              storage.index_done_callback() # 假设这个回调是全局的
              for storage in [
                   self.full_docs, self.text_chunks, self.llm_response_cache,
                   self.entities_vdb, self.relationships_vdb, self.chunks_vdb,
                   self.chunk_entity_relation_graph, self.doc_status # DocStatus 也要调用
              ] if storage is not None
         ]
         await asyncio.gather(*tasks)
         logger.info(f"Index done callback completed (folder context: {folder_id if folder_id else 'Global'})")


    def query(
        self,
        query_text: str, # 重命名以避免与内部变量冲突
        folder_id: int, # <--- 新增参数
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> str | Iterator[str]:
        """Perform a sync query within a specific folder."""
        if folder_id is None:
            raise ValueError("folder_id must be provided for query operation")
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query_text, folder_id, param, system_prompt)) # type: ignore

    async def aquery(
        self,
        query_text: str, # 重命名
        folder_id: int, # <--- 新增参数
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> str | AsyncIterator[str]:
        """Perform an async query within a specific folder."""
        if folder_id is None:
            raise ValueError("folder_id must be provided for aquery operation")

        global_config = asdict(self)
        response = None

        # 根据模式调用修改后的 operate.py 函数，并传递 folder_id
        if param.mode in ["local", "global", "hybrid"]:
            response = await kg_query( # 假设 kg_query 已被修改
                query_text.strip(),
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                global_config,
                folder_id=folder_id, # <--- 传递 folder_id
                hashing_kv=self.llm_response_cache,
                system_prompt=system_prompt,
            )
        elif param.mode == "naive":
             response = await naive_query( # 假设 naive_query 已被修改
                  query_text.strip(),
                  self.chunks_vdb,
                  self.text_chunks,
                  param,
                  global_config,
                  folder_id=folder_id, # <--- 传递 folder_id
                  hashing_kv=self.llm_response_cache,
                  system_prompt=system_prompt,
             )
        elif param.mode == "mix":
             response = await mix_kg_vector_query( # 假设 mix_kg_vector_query 已被修改
                  query_text.strip(),
                  self.chunk_entity_relation_graph,
                  self.entities_vdb,
                  self.relationships_vdb,
                  self.chunks_vdb,
                  self.text_chunks,
                  param,
                  global_config,
                  folder_id=folder_id, # <--- 传递 folder_id
                  hashing_kv=self.llm_response_cache,
                  system_prompt=system_prompt,
             )
        else:
            raise ValueError(f"Unknown mode {param.mode}")

        # 查询后的回调，可能需要 folder_id
        await self._query_done(folder_id=folder_id)
        return response

    async def _query_done(self, folder_id: int = None): # <--- folder_id 可选
         # 主要刷新缓存
         await self.llm_response_cache.index_done_callback()
         logger.debug(f"Query done callback completed (folder context: {folder_id if folder_id else 'Global'})")


    # --- 重写 CRUD 方法 ---
    def delete_by_entity(self, entity_name: str, folder_id: int) -> None:
        if folder_id is None: raise ValueError("folder_id required")
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name, folder_id))

    async def adelete_by_entity(self, entity_name: str, folder_id: int) -> None:
        if folder_id is None: raise ValueError("folder_id required")
        try:
            # 调用 tenant-aware 适配器方法
            await asyncio.gather(
                 self.entities_vdb.delete_entity(entity_name, folder_id=folder_id),
                 self.relationships_vdb.delete_entity_relation(entity_name, folder_id=folder_id),
                 self.chunk_entity_relation_graph.delete_node(entity_name, folder_id=folder_id)
            )
            logger.info(f"Entity '{entity_name}' deleted from folder {folder_id}.")
            await self._delete_by_entity_done(folder_id=folder_id) # 可能需要 folder_id
        except Exception as e:
            logger.error(f"Error deleting entity '{entity_name}' in folder {folder_id}: {e}")

    async def _delete_by_entity_done(self, folder_id: int = None) -> None:
         # 刷新相关存储
         await asyncio.gather(
              self.entities_vdb.index_done_callback(),
              self.relationships_vdb.index_done_callback(),
              self.chunk_entity_relation_graph.index_done_callback(),
         )
         logger.debug(f"Delete entity done callback completed (folder context: {folder_id if folder_id else 'Global'})")



    async def _crud_done(self, storages_involved: List[Union[BaseKVStorage, BaseVectorStorage, BaseGraphStorage]], folder_id: int):
        """Calls index_done_callback on relevant storages after a CRUD operation."""
        tasks = [storage.index_done_callback() for storage in storages_involved if storage]
        await asyncio.gather(*tasks)
        logger.debug(f"CRUD done callback completed for folder {folder_id}")     

    # --- Delete Relation ---
    def delete_by_relation(self, source_entity: str, target_entity: str, folder_id: int) -> None:
        """Synchronously delete a relation between two entities within a folder."""
        if folder_id is None: raise ValueError("folder_id required")
        loop = always_get_an_event_loop()
        loop.run_until_complete(self.adelete_by_relation(source_entity, target_entity, folder_id))

    async def adelete_by_relation(self, source_entity: str, target_entity: str, folder_id: int) -> None:
        """Asynchronously delete a relation between two entities within a folder."""
        if folder_id is None: raise ValueError("folder_id required")
        try:
            # Check existence within the folder first using tenant-aware adapter
            edge_exists = await self.chunk_entity_relation_graph.has_edge(source_entity, target_entity, folder_id=folder_id)
            if not edge_exists:
                logger.warning(f"Relation {source_entity}-{target_entity} not found in folder {folder_id}")
                return

            # Delete from graph store (tenant-aware)
            # Assuming adapter's remove_edges takes folder_id
            await self.chunk_entity_relation_graph.remove_edges([(source_entity, target_entity)], folder_id=folder_id)

            # Delete from vector store (tenant-aware)
            relation_id = compute_mdhash_id(source_entity + target_entity, prefix="rel-")
            relation_id_rev = compute_mdhash_id(target_entity + source_entity, prefix="rel-") # If reverse relations might be stored
            await self.relationships_vdb.delete([relation_id, relation_id_rev], folder_id=folder_id)

            logger.info(f"Relation {source_entity}-{target_entity} deleted from folder {folder_id}.")
            # Call CRUD done callback for relevant stores
            await self._crud_done([self.chunk_entity_relation_graph, self.relationships_vdb], folder_id=folder_id)
        except Exception as e:
            logger.error(f"Error deleting relation {source_entity}-{target_entity} folder {folder_id}: {e}")

    # --- Delete By Doc ID (Tenant-Aware) ---
    def delete_by_doc_id(self, doc_id: str, folder_id: int) -> None:
        """Synchronously delete a document and its related data within a folder."""
        if folder_id is None: raise ValueError("folder_id required")
        loop = always_get_an_event_loop()
        loop.run_until_complete(self.adelete_by_doc_id(doc_id, folder_id))

    async def adelete_by_doc_id(self, doc_id: str, folder_id: int) -> None:
        """Asynchronously delete a document and related data within a folder."""
        if folder_id is None: raise ValueError("folder_id required")
        # 1. Check Doc Status (Global - as decided)
        doc_status_info = await self.doc_status.get_by_id(doc_id)
        if not doc_status_info:
            logger.warning(f"Document {doc_id} not found in global status. Cannot delete."); return
        # Optional: Verify doc_id belongs to folder_id via your MySQL DB here.

        logger.info(f"Starting deletion for doc {doc_id} within folder {folder_id}")

        # 2. Find related chunk IDs *within the folder*
        chunk_ids_to_delete = []
        try:
             # Assuming Milvus adapter has query_by_filter or similar
             if hasattr(self.chunks_vdb, 'query_by_filter'):
                  chunk_results = await self.chunks_vdb.query_by_filter(
                       filter_expr=f'full_doc_id == "{doc_id}" and folder_id == {folder_id}',
                       output_fields=["id"]
                  )
                  chunk_ids_to_delete = [r["id"] for r in chunk_results]
             else: # Fallback using KV store (potentially less efficient)
                  logger.warning("chunks_vdb lacks query_by_filter. Using KV fallback for chunk ID retrieval.")
                  # Assumes text_chunks adapter has a way to list/filter by folder
                  # This requires modification to TenantAwareRedisKVStorage or base interface
                  # Example placeholder:
                  all_folder_chunks = await self.text_chunks.get_all_in_folder(folder_id) # Needs implementation
                  chunk_ids_to_delete = [cid for cid, cdata in all_folder_chunks.items() if cdata.get("full_doc_id") == doc_id]

        except Exception as e:
             logger.error(f"Error finding chunks for doc {doc_id} folder {folder_id}: {e}"); return


        if not chunk_ids_to_delete:
            logger.warning(f"No chunks found for doc {doc_id} in folder {folder_id}. Deleting doc entry only.")
            await asyncio.gather(
                self.full_docs.delete([doc_id], folder_id=folder_id), # Delete doc content in folder
                self.doc_status.delete([doc_id]) # Delete global status
            )
            await self._crud_done([self.full_docs, self.doc_status], folder_id=folder_id)
            return

        logger.debug(f"Found {len(chunk_ids_to_delete)} chunks to delete for doc {doc_id} folder {folder_id}")

        # 3. Delete Chunks (VDB and KV, tenant-aware)
        await asyncio.gather(
            self.chunks_vdb.delete(chunk_ids_to_delete, folder_id=folder_id),
            self.text_chunks.delete(chunk_ids_to_delete, folder_id=folder_id)
        )

        # 4. Find and handle related Entities/Relationships (within folder)
        # This part remains complex and requires careful handling of source_id updates.
        # Simple Approach: Delete entities/relationships ONLY if *all* their source_ids
        # point to the chunks being deleted within this folder.
        entities_to_delete = set()
        rels_to_delete_vdb_ids = set()
        # We need to iterate through nodes and edges associated with the deleted chunks.
        # This can be inefficient without proper indexing or graph structure.

        # Simplified logic: Iterate nodes in the folder, check if sources are subset of deleted chunks.
        all_folder_entities = await self.chunk_entity_relation_graph.get_all_labels(folder_id=folder_id)
        update_node_tasks = []
        delete_entity_graph_tasks = []

        for entity_name in all_folder_entities:
             node_data = await self.chunk_entity_relation_graph.get_node(entity_name, folder_id=folder_id)
             if node_data and node_data.get("source_id"):
                  sources = set(split_string_by_multi_markers(node_data["source_id"], [GRAPH_FIELD_SEP]))
                  deleted_chunk_set = set(chunk_ids_to_delete)
                  if sources.issubset(deleted_chunk_set): # All sources are being deleted
                       entities_to_delete.add(entity_name)
                       delete_entity_graph_tasks.append(
                            self.chunk_entity_relation_graph.delete_node(entity_name, folder_id=folder_id)
                       )
                  elif sources & deleted_chunk_set: # Some sources are being deleted
                       remaining_sources = sources - deleted_chunk_set
                       node_data["source_id"] = GRAPH_FIELD_SEP.join(sorted(list(remaining_sources)))
                       update_node_tasks.append(
                            self.chunk_entity_relation_graph.upsert_node(entity_name, node_data, folder_id=folder_id)
                       )
                       # Note: VDB entry for this updated entity might become stale without re-embedding/update.

        # Execute graph updates/deletions
        await asyncio.gather(*update_node_tasks)
        await asyncio.gather(*delete_entity_graph_tasks)

        # Delete orphaned entities from VDB
        if entities_to_delete:
             delete_vdb_tasks = [self.entities_vdb.delete_entity(name, folder_id=folder_id) for name in entities_to_delete]
             await asyncio.gather(*delete_vdb_tasks)
             logger.info(f"Deleted {len(entities_to_delete)} orphaned entities from VDB in folder {folder_id}")

        # Relationship Cleanup (Potentially complex)
        # If graph `delete_node` uses `DETACH DELETE`, relationships are handled in graph.
        # We might still need to clean up relationship entries in VDB if they only referenced deleted entities/chunks.
        # This requires querying relationships based on source_id or entity IDs.
        # For now, we'll assume graph deletion handles the primary cleanup.
        logger.warning(f"Relationship VDB cleanup during doc deletion for folder {folder_id} might be incomplete.")

        # 5. Delete original document content (KV, tenant-aware) and status (Global)
        await asyncio.gather(
             self.full_docs.delete([doc_id], folder_id=folder_id),
             self.doc_status.delete([doc_id]) # Global delete
        )

        # 6. Callback
        await self._crud_done([
             self.chunks_vdb, self.text_chunks, self.entities_vdb,
             self.chunk_entity_relation_graph, self.full_docs, self.doc_status
             ], folder_id=folder_id)

        logger.info(f"Successfully deleted doc {doc_id} and related data from folder {folder_id}.")


    # --- Get Relation Info ---
    # Redefine to match the structure and call tenant-aware adapters
    async def get_relation_info(self, src_entity: str, tgt_entity: str, folder_id: int, include_vector_data: bool = False) -> Dict[str, Any]:
        """Gets relation info within a specific folder."""
        if folder_id is None: raise ValueError("folder_id required")
        edge_data = await self.chunk_entity_relation_graph.get_edge(src_entity, tgt_entity, folder_id=folder_id)
        source_id = edge_data.get("source_id") if edge_data else None
        result = {"src_entity": src_entity, "tgt_entity": tgt_entity, "source_id": source_id, "graph_data": edge_data, "folder_id": folder_id}
        if include_vector_data:
            rel_id = compute_mdhash_id(src_entity + tgt_entity, prefix="rel-")
            # Assuming Milvus get_by_id takes folder_id
            vector_data = await self.relationships_vdb.get_by_id(rel_id, folder_id=folder_id)
            result["vector_data"] = vector_data
        return result

    # --- Edit Entity ---
    # Redefine sync wrapper
    def edit_entity(self, entity_name: str, updated_data: dict[str, Any], folder_id: int, allow_rename: bool = True) -> dict[str, Any]:
        """Synchronously edit an entity within a folder."""
        if folder_id is None: raise ValueError("folder_id required")
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aedit_entity(entity_name, updated_data, folder_id, allow_rename))

    # Keep the async implementation previously provided
    async def aedit_entity(self, entity_name: str, updated_data: dict[str, Any], folder_id: int, allow_rename: bool = True) -> dict[str, Any]:
        """Asynchronously edit an entity within a folder."""
        if folder_id is None: raise ValueError("folder_id required")
        try:
            # 1. Check existence & Get Data
            node_data = await self.chunk_entity_relation_graph.get_node(entity_name, folder_id=folder_id)
            if not node_data: raise ValueError(f"Entity '{entity_name}' not found in folder {folder_id}")

            # 2. Handle Renaming
            new_entity_name = updated_data.get("entity_name", entity_name)
            is_renaming = new_entity_name != entity_name
            new_node_data = {**node_data, **updated_data}
            new_node_data.pop("entity_name", None)
            new_node_data["entity_id"] = new_entity_name # For graph storage

            if is_renaming:
                if not allow_rename: raise ValueError("Renaming not allowed")
                new_name_exists = await self.chunk_entity_relation_graph.has_node(new_entity_name, folder_id=folder_id)
                if new_name_exists: raise ValueError(f"Entity name '{new_entity_name}' already exists in folder {folder_id}")
                logger.info(f"Renaming '{entity_name}' to '{new_entity_name}' in folder {folder_id}")

                # Create new node
                await self.chunk_entity_relation_graph.upsert_node(new_entity_name, new_node_data, folder_id=folder_id)

                # Re-link relationships
                edges = await self.chunk_entity_relation_graph.get_node_edges(entity_name, folder_id=folder_id)
                rels_to_update_in_vdb = []
                old_rel_vdb_ids_to_delete = set() # Use set for uniqueness
                if edges:
                    edge_detail_tasks = [self.chunk_entity_relation_graph.get_edge(src, tgt, folder_id=folder_id) for src, tgt in edges]
                    edge_details = await asyncio.gather(*edge_detail_tasks)
                    for (src, tgt), edge_data in zip(edges, edge_details):
                        if edge_data:
                            old_rel_vdb_ids_to_delete.add(compute_mdhash_id(src + tgt, prefix="rel-"))
                            old_rel_vdb_ids_to_delete.add(compute_mdhash_id(tgt + src, prefix="rel-"))
                            new_src = new_entity_name if src == entity_name else src
                            new_tgt = new_entity_name if tgt == entity_name else tgt
                            await self.chunk_entity_relation_graph.upsert_edge(new_src, new_tgt, edge_data, folder_id=folder_id)
                            rels_to_update_in_vdb.append((new_src, new_tgt, edge_data))

                # Delete old node (graph) & VDB entries
                await self.chunk_entity_relation_graph.delete_node(entity_name, folder_id=folder_id)
                old_entity_vdb_id = compute_mdhash_id(entity_name, prefix="ent-")
                await self.entities_vdb.delete([old_entity_vdb_id], folder_id=folder_id)
                if old_rel_vdb_ids_to_delete:
                    await self.relationships_vdb.delete(list(old_rel_vdb_ids_to_delete), folder_id=folder_id)

                # Update VDB for new relationships
                rel_vdb_updates = {}
                for src, tgt, data in rels_to_update_in_vdb:
                    rel_id = compute_mdhash_id(src + tgt, prefix="rel-")
                    rel_vdb_updates[rel_id] = { "content": f"{src}\t{tgt}\n{data.get('keywords','')}\n{data.get('description','')}", "src_id": src, "tgt_id": tgt, **data }
                if rel_vdb_updates: await self.relationships_vdb.upsert(rel_vdb_updates, folder_id=folder_id)

                entity_name = new_entity_name # Use new name for final VDB update

            else: # Not renaming
                await self.chunk_entity_relation_graph.upsert_node(entity_name, new_node_data, folder_id=folder_id)

            # 3. Update Entity VDB entry
            entity_vdb_id = compute_mdhash_id(entity_name, prefix="ent-")
            entity_data_for_vdb = {
                entity_vdb_id: { "content": f"{entity_name}\n{new_node_data.get('description','')}", "entity_name": entity_name, **new_node_data }
            }
            # Delete old entry first to ensure overwrite if ID is same but content changed
            await self.entities_vdb.delete([entity_vdb_id], folder_id=folder_id)
            await self.entities_vdb.upsert(entity_data_for_vdb, folder_id=folder_id)

            # 4. Callback
            await self._crud_done([self.chunk_entity_relation_graph, self.entities_vdb, self.relationships_vdb], folder_id=folder_id)
            logger.info(f"Entity '{entity_name}' updated in folder {folder_id}")
            return await self.get_entity_info(entity_name, folder_id=folder_id, include_vector_data=True)

        except Exception as e:
            logger.error(f"Error editing entity '{entity_name}' folder {folder_id}: {e}")
            raise


    # --- Create Entity ---
    # Redefine sync wrapper
    def create_entity(self, entity_name: str, entity_data: dict[str, Any], folder_id: int) -> dict[str, Any]:
        """Synchronously create an entity within a folder."""
        if folder_id is None: raise ValueError("folder_id required")
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.acreate_entity(entity_name, entity_data, folder_id))

    # Keep async implementation
    async def acreate_entity(self, entity_name: str, entity_data: dict[str, Any], folder_id: int) -> dict[str, Any]:
        """Asynchronously create an entity within a folder."""
        if folder_id is None: raise ValueError("folder_id required")
        try:
            # 1. Check existence
            node_exists = await self.chunk_entity_relation_graph.has_node(entity_name, folder_id=folder_id)
            if node_exists: raise ValueError(f"Entity '{entity_name}' already exists in folder {folder_id}")

            # 2. Prepare node data
            node_data = { "entity_id": entity_name, **entity_data } # Ensure entity_id is set

            # 3. Create in Graph Store
            await self.chunk_entity_relation_graph.upsert_node(entity_name, node_data, folder_id=folder_id)

            # 4. Create in VDB
            entity_vdb_id = compute_mdhash_id(entity_name, prefix="ent-")
            entity_data_for_vdb = {
                entity_vdb_id: { "content": f"{entity_name}\n{node_data.get('description','')}", "entity_name": entity_name, **node_data }
            }
            await self.entities_vdb.upsert(entity_data_for_vdb, folder_id=folder_id)

            # 5. Callback
            await self._crud_done([self.chunk_entity_relation_graph, self.entities_vdb], folder_id=folder_id)
            logger.info(f"Entity '{entity_name}' created in folder {folder_id}")
            return await self.get_entity_info(entity_name, folder_id=folder_id, include_vector_data=True)

        except Exception as e:
            logger.error(f"Error creating entity '{entity_name}' folder {folder_id}: {e}")
            raise

    # --- Create Relation ---
    # Redefine sync wrapper
    def create_relation(self, source_entity: str, target_entity: str, relation_data: dict[str, Any], folder_id: int) -> dict[str, Any]:
        """Synchronously create a relation within a folder."""
        if folder_id is None: raise ValueError("folder_id required")
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.acreate_relation(source_entity, target_entity, relation_data, folder_id))

    # Keep async implementation
    async def acreate_relation(self, source_entity: str, target_entity: str, relation_data: dict[str, Any], folder_id: int) -> dict[str, Any]:
        """Asynchronously create a relation within a folder."""
        if folder_id is None: raise ValueError("folder_id required")
        try:
            # 1. Check node existence
            source_exists = await self.chunk_entity_relation_graph.has_node(source_entity, folder_id=folder_id)
            target_exists = await self.chunk_entity_relation_graph.has_node(target_entity, folder_id=folder_id)
            if not source_exists: raise ValueError(f"Source '{source_entity}' not found in folder {folder_id}")
            if not target_exists: raise ValueError(f"Target '{target_entity}' not found in folder {folder_id}")

            # 2. Check edge existence
            edge_exists = await self.chunk_entity_relation_graph.has_edge(source_entity, target_entity, folder_id=folder_id)
            if edge_exists: raise ValueError(f"Relation {source_entity}-{target_entity} already exists in folder {folder_id}")

            # 3. Prepare edge data
            edge_data = {**relation_data} # Copy relation data

            # 4. Create in Graph Store
            await self.chunk_entity_relation_graph.upsert_edge(source_entity, target_entity, edge_data, folder_id=folder_id)

            # 5. Create in VDB
            rel_id = compute_mdhash_id(source_entity + target_entity, prefix="rel-")
            rel_data_for_vdb = {
                 rel_id: { "content": f"{source_entity}\t{target_entity}\n{edge_data.get('keywords','')}\n{edge_data.get('description','')}", "src_id": source_entity, "tgt_id": target_entity, **edge_data }
            }
            await self.relationships_vdb.upsert(rel_data_for_vdb, folder_id=folder_id)

            # 6. Callback
            await self._crud_done([self.chunk_entity_relation_graph, self.relationships_vdb], folder_id=folder_id)
            logger.info(f"Relation {source_entity}-{target_entity} created in folder {folder_id}")
            return await self.get_relation_info(source_entity, target_entity, folder_id=folder_id, include_vector_data=True)

        except Exception as e:
            logger.error(f"Error creating relation {source_entity}-{target_entity} folder {folder_id}: {e}")
            raise

    # --- Merge Entities ---
    # Redefine sync wrapper
    def merge_entities(self, source_entities: list[str], target_entity: str, folder_id: int, merge_strategy: dict[str, str] = None, target_entity_data: dict[str, Any] = None) -> dict[str, Any]:
        """Synchronously merge entities within a folder."""
        if folder_id is None: raise ValueError("folder_id required")
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.amerge_entities(source_entities, target_entity, folder_id, merge_strategy, target_entity_data))

    # Keep async implementation
    async def amerge_entities(self, source_entities: list[str], target_entity: str, folder_id: int, merge_strategy: dict[str, str] = None, target_entity_data: dict[str, Any] = None) -> dict[str, Any]:
        """Asynchronously merge entities within a folder."""
        if folder_id is None: raise ValueError("folder_id required")
        try:
             # Default merge strategy
             default_strategy = {"description": "concatenate", "entity_type": "keep_first", "source_id": "join_unique", "file_path": "join_unique"}
             merge_strategy = default_strategy if merge_strategy is None else {**default_strategy, **merge_strategy}
             target_entity_data = {} if target_entity_data is None else target_entity_data

             # 1. Check sources existence & Get data
             source_entities_data = {}
             fetch_source_tasks = [self.chunk_entity_relation_graph.get_node(name, folder_id=folder_id) for name in source_entities]
             source_nodes = await asyncio.gather(*fetch_source_tasks)
             for name, node_data in zip(source_entities, source_nodes):
                  if not node_data: raise ValueError(f"Source entity '{name}' not found in folder {folder_id}")
                  source_entities_data[name] = node_data

             # 2. Check target existence & Get data
             existing_target_entity_data = await self.chunk_entity_relation_graph.get_node(target_entity, folder_id=folder_id)
             target_exists = existing_target_entity_data is not None

             # 3. Merge attributes
             all_data_to_merge = list(source_entities_data.values()) + ([existing_target_entity_data] if target_exists else [])
             merged_entity_data = self._merge_entity_attributes(all_data_to_merge, merge_strategy)
             merged_entity_data.update(target_entity_data) # Apply explicit overrides
             merged_entity_data["entity_id"] = target_entity

             # 4. Get source relationships
             all_relations = []
             fetch_edge_tasks = [self.chunk_entity_relation_graph.get_node_edges(name, folder_id=folder_id) for name in source_entities]
             edges_per_source = await asyncio.gather(*fetch_edge_tasks)
             edge_tuples_to_fetch_details = set()
             for edges in edges_per_source:
                  if edges: edge_tuples_to_fetch_details.update(edges)
             if edge_tuples_to_fetch_details:
                  edge_detail_tasks = [self.chunk_entity_relation_graph.get_edge(src, tgt, folder_id=folder_id) for src, tgt in edge_tuples_to_fetch_details]
                  edge_details = await asyncio.gather(*edge_detail_tasks)
                  for (src, tgt), edge_data in zip(edge_tuples_to_fetch_details, edge_details):
                       if edge_data and (src in source_entities or tgt in source_entities): # Ensure it involves a source entity
                            all_relations.append((src, tgt, edge_data))


             # 5. Upsert target entity
             await self.chunk_entity_relation_graph.upsert_node(target_entity, merged_entity_data, folder_id=folder_id)

             # 6. Recreate relationships
             relation_updates = {}
             old_rel_vdb_ids_to_delete = set()
             for src, tgt, edge_data in all_relations:
                  old_rel_vdb_ids_to_delete.add(compute_mdhash_id(src + tgt, prefix="rel-"))
                  old_rel_vdb_ids_to_delete.add(compute_mdhash_id(tgt + src, prefix="rel-"))
                  new_src = target_entity if src in source_entities else src
                  new_tgt = target_entity if tgt in source_entities else tgt
                  if new_src == new_tgt: continue
                  relation_key = tuple(sorted((new_src, new_tgt)))
                  if relation_key in relation_updates:
                       existing = relation_updates[relation_key]["data"]
                       merged = self._merge_relation_attributes([existing, edge_data], {"description": "concatenate", "keywords": "join_unique", "source_id": "join_unique", "file_path": "join_unique", "weight": "max"})
                       relation_updates[relation_key]["data"] = merged
                  else:
                       relation_updates[relation_key] = {"src": new_src, "tgt": new_tgt, "data": edge_data.copy()}

             upsert_edge_tasks = [self.chunk_entity_relation_graph.upsert_edge(r["src"], r["tgt"], r["data"], folder_id=folder_id) for r in relation_updates.values()]
             await asyncio.gather(*upsert_edge_tasks)

             # 7. Update Target Entity VDB
             target_entity_vdb_id = compute_mdhash_id(target_entity, prefix="ent-")
             target_vdb_data = {target_entity_vdb_id: {"content": f"{target_entity}\n{merged_entity_data.get('description','')}", "entity_name": target_entity, **merged_entity_data}}
             # Delete first in case ID exists but content changes
             await self.entities_vdb.delete([target_entity_vdb_id], folder_id=folder_id)
             await self.entities_vdb.upsert(target_vdb_data, folder_id=folder_id)

             # 8. Update Relationship VDB entries
             if old_rel_vdb_ids_to_delete:
                 await self.relationships_vdb.delete(list(old_rel_vdb_ids_to_delete), folder_id=folder_id)
             rel_vdb_updates = {}
             for rel_data in relation_updates.values():
                  src, tgt, data = rel_data["src"], rel_data["tgt"], rel_data["data"]
                  rel_id = compute_mdhash_id(src + tgt, prefix="rel-")
                  rel_vdb_updates[rel_id] = {"content": f"{src}\t{tgt}\n{data.get('keywords','')}\n{data.get('description','')}", "src_id": src, "tgt_id": tgt, **data}
             if rel_vdb_updates: await self.relationships_vdb.upsert(rel_vdb_updates, folder_id=folder_id)

             # 9. Delete source entities
             tasks_graph_delete = []
             tasks_vdb_delete = []
             for entity_name in source_entities:
                  if entity_name == target_entity: continue
                  tasks_graph_delete.append(self.chunk_entity_relation_graph.delete_node(entity_name, folder_id=folder_id))
                  entity_vdb_id = compute_mdhash_id(entity_name, prefix="ent-")
                  tasks_vdb_delete.append(self.entities_vdb.delete([entity_vdb_id], folder_id=folder_id))
             await asyncio.gather(*tasks_graph_delete)
             await asyncio.gather(*tasks_vdb_delete)

             # 10. Callback
             await self._crud_done([self.chunk_entity_relation_graph, self.entities_vdb, self.relationships_vdb], folder_id=folder_id)
             logger.info(f"Merged entities into '{target_entity}' in folder {folder_id}")
             return await self.get_entity_info(target_entity, folder_id=folder_id, include_vector_data=True)

        except Exception as e:
             logger.error(f"Error merging entities folder {folder_id}: {e}"); raise
        

    # --- Export Data ---
    # Redefine sync wrapper
    def export_data(self, output_path: str, folder_id: int, file_format: str = "csv", include_vector_data: bool = False) -> None:
        """Synchronously export data for a specific folder."""
        if folder_id is None: raise ValueError("folder_id required")
        loop = always_get_an_event_loop()
        loop.run_until_complete(self.aexport_data(output_path, folder_id, file_format, include_vector_data))

    # Keep async implementation
    async def aexport_data(self, output_path: str, folder_id: int, file_format: str = "csv", include_vector_data: bool = False) -> None:
        """Asynchronously export data for a specific folder."""
        if folder_id is None: raise ValueError("folder_id required")
        logger.info(f"Exporting data folder {folder_id} to {output_path} (format: {file_format})")

        # --- Data Collection ---
        entities_data, relations_data, relationships_vdb_data = [], [], []
        all_entity_labels = await self.chunk_entity_relation_graph.get_all_labels(folder_id=folder_id)
        # Fetch entities
        entity_tasks = [self.get_entity_info(name, folder_id=folder_id, include_vector_data=include_vector_data) for name in all_entity_labels]
        entity_infos = await asyncio.gather(*entity_tasks)
        for info in entity_infos:
            row = {"entity_name": info["entity_name"], "source_id": info["source_id"], "graph_data": str(info.get("graph_data", {})),}
            if include_vector_data and "vector_data" in info: row["vector_data"] = str(info.get("vector_data", {}))
            entities_data.append(row)
        # Fetch relations (graph edges)
        seen_edge_pairs = set()
        edge_tasks = []
        for entity_name in all_entity_labels:
             edges = await self.chunk_entity_relation_graph.get_node_edges(entity_name, folder_id=folder_id)
             if edges:
                  for src, tgt in edges:
                       sorted_pair = tuple(sorted((src, tgt)))
                       if sorted_pair not in seen_edge_pairs:
                            edge_tasks.append(self.get_relation_info(src, tgt, folder_id=folder_id, include_vector_data=include_vector_data))
                            seen_edge_pairs.add(sorted_pair)
        relation_infos = await asyncio.gather(*edge_tasks)
        for info in relation_infos:
            row = {"src_entity": info["src_entity"], "tgt_entity": info["tgt_entity"], "source_id": info["source_id"], "graph_data": str(info.get("graph_data", {})),}
            if include_vector_data and "vector_data" in info: row["vector_data"] = str(info.get("vector_data", {}))
            relations_data.append(row)
        # Fetch relationships (VDB entries)
        if include_vector_data:
             try:
                  if hasattr(self.relationships_vdb, 'query_by_filter'):
                       all_rels_vdb = await self.relationships_vdb.query_by_filter(filter_expr=f'folder_id == {folder_id}', output_fields=list(self.relationships_vdb.meta_fields) + ["id"], limit=10000)
                       for rel in all_rels_vdb: relationships_vdb_data.append({"relationship_vdb_id": rel["id"], **{k:v for k,v in rel.items() if k!='id'}})
                  else: logger.warning("VDB relationship export skipped: query_by_filter not supported.")
             except Exception as e: logger.error(f"Error exporting VDB relationships folder {folder_id}: {e}")

        # --- Export Logic ---
        # ... (Keep the CSV, Excel, MD, TXT export logic from the previous correct version) ...
        if file_format == "csv":
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                if entities_data: f.write("# ENTITIES\n"); w = csv.DictWriter(f, entities_data[0].keys()); w.writeheader(); w.writerows(entities_data); f.write("\n\n")
                if relations_data: f.write("# RELATIONS (Graph Edges)\n"); w = csv.DictWriter(f, relations_data[0].keys()); w.writeheader(); w.writerows(relations_data); f.write("\n\n")
                if relationships_vdb_data: f.write("# RELATIONSHIPS (Vector DB Entries)\n"); w = csv.DictWriter(f, relationships_vdb_data[0].keys()); w.writeheader(); w.writerows(relationships_vdb_data)
        elif file_format == "excel":
            with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
                if entities_data: pd.DataFrame(entities_data).to_excel(writer, sheet_name="Entities", index=False)
                if relations_data: pd.DataFrame(relations_data).to_excel(writer, sheet_name="Relations (Graph)", index=False)
                if relationships_vdb_data: pd.DataFrame(relationships_vdb_data).to_excel(writer, sheet_name="Relationships (VDB)", index=False)
        # ... (Implement MD and TXT formats similarly) ...
        else: raise ValueError(f"Unsupported format: {file_format}")
        logger.info(f"Data export for folder {folder_id} completed.")
     

    async def get_entity_info(
        self, entity_name: str, folder_id: int, include_vector_data: bool = False
    ) -> dict[str, Any]: # 返回类型改为 Any
         """Gets entity info within a specific folder."""
         if folder_id is None: raise ValueError("folder_id required")
         node_data = await self.chunk_entity_relation_graph.get_node(entity_name, folder_id=folder_id)
         source_id = node_data.get("source_id") if node_data else None
         result = {
              "entity_name": entity_name, "source_id": source_id,
              "graph_data": node_data, "folder_id": folder_id
         }
         if include_vector_data:
              entity_vdb_id = compute_mdhash_id(entity_name, prefix="ent-")
              vector_data = await self.entities_vdb.get_by_id(entity_vdb_id, folder_id=folder_id)
              result["vector_data"] = vector_data
         return result

    # --- 为 aclear_cache 添加 folder_id ---
    async def aclear_cache(self, modes: list[str] | None = None, folder_id: int = None) -> None:
         """Clears cache for specific modes, optionally within a folder."""
         if not self.llm_response_cache:
              logger.warning("No cache storage configured")
              return
         if folder_id is None:
              logger.warning("Clearing cache globally as folder_id is not provided.")
              # 调用原始的全局清除逻辑或决定是否要阻止全局清除
              # For safety, let's prevent accidental global clear from tenant-aware class
              logger.error("Global cache clear attempted from tenant-aware instance. Use folder_id.")
              return
              # Or call super().aclear_cache(modes) if global clear is desired

         valid_modes = ["default", "naive", "local", "global", "hybrid", "mix"]
         if modes and not all(mode in valid_modes for mode in modes):
              raise ValueError(f"Invalid mode. Valid modes are: {valid_modes}")

         try:
              target_modes = modes if modes else valid_modes
              # 调用 tenant-aware 的 drop_cache_by_modes
              success = await self.llm_response_cache.drop_cache_by_modes(target_modes, folder_id=folder_id)
              if success:
                   logger.info(f"Cleared cache for modes {target_modes} in folder {folder_id}")
              else:
                   logger.warning(f"Failed to clear cache for modes {target_modes} in folder {folder_id}")
              await self.llm_response_cache.index_done_callback() # 刷新缓存
         except Exception as e:
              logger.error(f"Error clearing cache for folder {folder_id}: {e}")


    def clear_cache(self, modes: list[str] | None = None, folder_id: int = None) -> None:
         """Synchronous version of aclear_cache."""
         return always_get_an_event_loop().run_until_complete(self.aclear_cache(modes, folder_id))

     # --- Drop Tenant Data Method ---
    async def adrop_tenant_data(self, folder_id: int) -> dict:
         """Asynchronously drops all data associated with a specific folder_id."""
         if folder_id is None:
              return {"status": "error", "message": "folder_id is required"}

         logger.warning(f"!!! Initiating data drop for folder_id: {folder_id} !!!")
         results = {}
         all_success = True

         # List of tenant-aware storage instances that support drop_tenant
         tenant_storages: List[Union[TenantAwareRedisKVStorage, TenantAwareMilvusVectorDBStorage, TenantAwareNeo4JStorage]] = [
              self.llm_response_cache, self.full_docs, self.text_chunks, # Redis KV
              self.entities_vdb, self.relationships_vdb, self.chunks_vdb, # Milvus Vector
              self.chunk_entity_relation_graph # Neo4j Graph
         ]

         tasks = []
         storage_names = []
         for storage in tenant_storages:
              if hasattr(storage, 'drop_tenant'):
                   tasks.append(storage.drop_tenant(folder_id=folder_id))
                   storage_names.append(storage.__class__.__name__)
              else:
                   logger.warning(f"Storage {storage.__class__.__name__} does not support drop_tenant method.")

         if tasks:
              task_results = await asyncio.gather(*tasks, return_exceptions=True)
              for name, res in zip(storage_names, task_results):
                   if isinstance(res, Exception):
                        results[name] = {"status": "error", "message": str(res)}
                        all_success = False
                   else:
                        results[name] = res
                        if res.get("status") != "success":
                             all_success = False

         # Also delete relevant entries from DocStatus (if needed, based on file_id linkage)
         # This requires linking folder_id to file_id in your main DB first.
         # For now, we skip DocStatus cleanup related to folder drop.
         # logger.info(f"DocStatus cleanup for folder {folder_id} is currently skipped.")

         final_status = "success" if all_success else "partial_error"
         logger.info(f"Tenant data drop for folder {folder_id} finished with status: {final_status}")
         return {"overall_status": final_status, "details": results}

    def drop_tenant_data(self, folder_id: int) -> dict:
         """Synchronously drops all data associated with a specific folder_id."""
         loop = always_get_an_event_loop()
         return loop.run_until_complete(self.adrop_tenant_data(folder_id))


    # --- Helper methods (already provided, keep them) ---
    def _merge_entity_attributes(self, entity_data_list: list[dict[str, Any]], merge_strategy: dict[str, str]) -> dict[str, Any]:
        # ... (Keep implementation) ...
        merged_data = {}
        all_keys = set(k for data in entity_data_list for k in data.keys())
        for key in all_keys:
            values = [data.get(key) for data in entity_data_list if data.get(key)]
            if not values: continue
            strategy = merge_strategy.get(key, "keep_first")
            if strategy == "concatenate": merged_data[key] = "\n\n".join(map(str, values))
            elif strategy == "keep_first": merged_data[key] = values[0]
            elif strategy == "keep_last": merged_data[key] = values[-1]
            elif strategy == "join_unique":
                 unique_items = set(item for value in values for item in str(value).split(GRAPH_FIELD_SEP))
                 merged_data[key] = GRAPH_FIELD_SEP.join(sorted(list(unique_items)))
            else: merged_data[key] = values[0]
        return merged_data

    def _merge_relation_attributes(self, relation_data_list: list[dict[str, Any]], merge_strategy: dict[str, str]) -> dict[str, Any]:
        # ... (Keep implementation) ...
        merged_data = {}
        all_keys = set(k for data in relation_data_list for k in data.keys())
        for key in all_keys:
            values = [data.get(key) for data in relation_data_list if data.get(key) is not None]
            if not values: continue
            strategy = merge_strategy.get(key, "keep_first")
            if strategy == "concatenate": merged_data[key] = "\n\n".join(map(str, values))
            elif strategy == "keep_first": merged_data[key] = values[0]
            elif strategy == "keep_last": merged_data[key] = values[-1]
            elif strategy == "join_unique":
                 unique_items = set(item for value in values for item in str(value).split(GRAPH_FIELD_SEP))
                 merged_data[key] = GRAPH_FIELD_SEP.join(sorted(list(unique_items)))
            elif strategy == "max":
                 try: merged_data[key] = max(float(v) for v in values)
                 except (ValueError, TypeError): merged_data[key] = values[0]
            else: merged_data[key] = values[0]
        return merged_data
     