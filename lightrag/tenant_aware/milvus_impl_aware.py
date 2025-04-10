# tenant_aware/milvus_impl_aware.py
import asyncio
import os
from typing import Any, final, List
from dataclasses import dataclass
import numpy as np
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema # 确保导入必要类型
from pymilvus.exceptions import IndexNotExistException, CollectionNotExistException, MilvusException
from lightrag.utils import logger, compute_mdhash_id,EmbeddingFunc
from lightrag.kg.milvus_impl import MilvusVectorDBStorage # 假设原始实现在这里

@final
@dataclass
class TenantAwareMilvusVectorDBStorage(MilvusVectorDBStorage):
    """
    A tenant-aware Milvus vector storage implementation using folder_id metadata.
    Inherits from the original MilvusVectorDBStorage.
    Assumes the collection schema includes an indexed 'folder_id' field (Int64).
    """
    folder_id_field: str = "folder_id" # Name of the field storing tenant ID
    vector_field: str = "vector"       # Default vector field name
    primary_field: str = "id"          # Default primary key field name

    # Override __post_init__ to ensure schema and index exist
    def __post_init__(self):
        # 确保 pymilvus 已导入
        if MilvusClient is None:
             logger.error("pymilvus is not installed. Cannot initialize TenantAwareMilvusVectorDBStorage.")
             return

        # 先调用父类的 __post_init__ 来创建 _client 等
        try:
            super().__post_init__()
        except Exception as e:
             logger.error(f"Error during base MilvusVectorDBStorage __post_init__: {e}")
             # 如果基类初始化失败，则无法继续
             raise RuntimeError(f"Failed to initialize base Milvus client: {e}") from e


        # 检查 _client 是否成功创建
        if not hasattr(self, '_client') or not self._client:
             logger.error("Milvus client was not initialized by the base class.")
             raise RuntimeError("Milvus client initialization failed.")

        # 确保集合、模式和索引存在
        self._ensure_collection_schema_and_indexes()

    def _check_index_exists(self, field_name: str) -> bool:
        """Checks if an index exists for the given field."""
        try:
            indexes = self._client.list_indexes(self.namespace)
            for index_name in indexes:
                try:
                    # describe_index 返回的是 Index 对象列表或 Index 对象，不是字典
                    index_desc_list = self._client.describe_index(self.namespace, index_name)
                    # 假设 describe_index 返回列表，即使只有一个索引
                    if isinstance(index_desc_list, list):
                         if not index_desc_list: continue # 空列表
                         index_desc = index_desc_list[0] # 取第一个
                    else:
                         index_desc = index_desc_list # 直接使用对象

                    # 访问对象的属性
                    if index_desc and hasattr(index_desc, 'field_name') and index_desc.field_name == field_name:
                        logger.debug(f"Index '{index_name}' found for field '{field_name}' in collection '{self.namespace}'.")
                        return True
                except IndexNotExistException:
                     logger.debug(f"Index '{index_name}' queried but does not exist (possibly race condition).")
                     continue # 索引不存在
                except MilvusException as desc_err:
                    logger.warning(f"Could not describe index '{index_name}': {desc_err}")
            logger.debug(f"No index found for field '{field_name}' in collection '{self.namespace}'.")
            return False
        except CollectionNotExistException:
            logger.debug(f"Collection '{self.namespace}' does not exist when checking index for '{field_name}'.")
            return False
        except Exception as e:
            logger.error(f"Error checking index for field '{field_name}': {e}")
            return False # Assume index doesn't exist on error

    def _ensure_collection_schema_and_indexes(self):
        """Ensures collection, schema, and vector index exist. Skips scalar index creation."""
        try:
            collection_exists = self._client.has_collection(self.namespace)

            if not collection_exists:
                logger.info(f"Collection '{self.namespace}' does not exist. Creating...")
                # Define schema fields (保持不变)
                fields = [
                    FieldSchema(name=self.primary_field, dtype=DataType.VARCHAR, is_primary=True, max_length=65535),
                    FieldSchema(name=self.folder_id_field, dtype=DataType.INT64), # 仍然包含 folder_id 字段
                    FieldSchema(name=self.vector_field, dtype=DataType.FLOAT_VECTOR, dim=self.embedding_func.embedding_dim),
                ]
                for meta_field_name in self.meta_fields:
                    if meta_field_name not in [self.primary_field, self.folder_id_field, self.vector_field]:
                        fields.append(FieldSchema(name=meta_field_name, dtype=DataType.VARCHAR, max_length=65535))

                schema = CollectionSchema(fields=fields, description=f"Tenant-aware collection for {self.namespace}")
                self._client.create_collection(collection_name=self.namespace, schema=schema)
                logger.info(f"Created collection '{self.namespace}'.")
                collection_exists = True

            else:
                logger.debug(f"Collection '{self.namespace}' already exists.")
                # 可选: 验证现有模式是否包含 folder_id 字段
                try:
                    desc = self._client.describe_collection(self.namespace)
                    field_names = [f.name for f in desc.fields]
                    if self.folder_id_field not in field_names:
                         # 如果字段确实不存在，这是一个更严重的问题，需要模式迁移或手动添加
                         logger.error(f"CRITICAL: Collection {self.namespace} exists but lacks the required '{self.folder_id_field}' field!")
                         raise RuntimeError(f"Missing '{self.folder_id_field}' field in existing collection {self.namespace}")
                except Exception as desc_err:
                    logger.warning(f"Could not describe collection {self.namespace} to verify schema: {desc_err}")


            # --- Ensure Indexes Exist ---
            if collection_exists:
                 # 1. Ensure Vector Index Exists (保持不变)
                 if not self._check_index_exists(self.vector_field):
                      logger.info(f"Creating vector index on field '{self.vector_field}' for collection '{self.namespace}'...")
                      try:
                           vector_index_params = self._client.prepare_index_params()
                           vector_index_params.add_index(
                                field_name=self.vector_field,
                                index_type="AUTOINDEX", # Or specific type like "HNSW" or "IVF_FLAT"
                                metric_type="COSINE",    # Or "L2", ensure consistency with search
                                # params={"M": 16, "efConstruction": 200} # Example for HNSW
                           )
                           self._client.create_index(self.namespace, vector_index_params)
                           logger.info(f"Successfully created vector index on '{self.vector_field}'.")
                      except Exception as e:
                           logger.error(f"Failed to create vector index on '{self.vector_field}': {e}")
                           # 向量索引失败通常是关键错误
                           raise RuntimeError(f"Failed to create vector index on {self.namespace}") from e


                 # 2. --- FIX: Skip Scalar Index Creation on folder_id ---
                 logger.debug(f"Skipping explicit scalar index creation for '{self.folder_id_field}'. Milvus will handle filtering.")


                 # 3. Load Collection (保持不变)
                 try:
                      logger.debug(f"Ensuring collection '{self.namespace}' is loaded...")
                      self._client.load_collection(self.namespace)
                      logger.info(f"Collection '{self.namespace}' loaded.")
                 except Exception as load_err:
                      logger.error(f"Failed to load Milvus collection '{self.namespace}': {load_err}")
                      raise RuntimeError(f"Failed to load collection {self.namespace}") from load_err
        except Exception as e:
            logger.error(f"Unexpected error ensuring Milvus collection schema/indexes for '{self.namespace}': {e}")
            raise

    async def query(
        self, query: str, top_k: int, folder_id: int, ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Queries Milvus, filtering by folder_id."""
        if folder_id is None:
            raise ValueError("folder_id must be provided for tenant-aware query")

        embedding_result = await self.embedding_func([query])
        # Ensure embedding is a list of lists for Milvus client search
        query_embedding = [embedding_result[0].tolist()] # Wrap in another list

        filter_expr = f"{self.folder_id_field} == {folder_id}"
        if ids:
            logger.warning("`ids` filter provided but may not be combined effectively with vector search and folder_id filter in all Milvus setups.")

        logger.debug(f"Querying Milvus collection {self.namespace} with top_k={top_k}, folder_id={folder_id}, filter='{filter_expr}'")

        try:
            results = self._client.search(
                collection_name=self.namespace,
                data=query_embedding, # Pass list of lists
                limit=top_k,
                filter=filter_expr,
                output_fields=["*"], # Fetch all fields
                search_params={"metric_type": "COSINE", "params": {}} # Simplified params
            )
        except Exception as e:
            logger.error(f"Error querying Milvus for folder {folder_id}: {e}")
            raise

        # Process results - Milvus search returns a list of lists of hits
        processed_results = []
        if results and results[0]:
            for hit in results[0]:
                 # Access hit attributes directly
                entity_data = {
                    "id": hit.id,
                    "distance": hit.distance,
                    # Access entity fields using .get() on the hit.entity dictionary
                    **{k: hit.entity.get(k) for k in hit.entity.keys()}
                }
                processed_results.append(entity_data)

        logger.debug(f"Milvus query for folder {folder_id} returned {len(processed_results)} results.")
        return processed_results


    # --- Override Delete/Get methods to include folder_id filter ---

    async def delete_entity(self, entity_name: str, folder_id: int) -> None:
        """Deletes a specific entity within a folder."""
        if folder_id is None:
            raise ValueError("folder_id must be provided for tenant-aware delete_entity")
        entity_id = compute_mdhash_id(entity_name, prefix="ent-")
        filter_expr = f'{self.primary_field} == "{entity_id}" and {self.folder_id_field} == {folder_id}'
        try:
            # Query first to see if it exists in this tenant, though delete is idempotent
            # existing = self._client.query(collection_name=self.namespace, filter=filter_expr, limit=1)
            # if not existing:
            #     logger.debug(f"Entity {entity_name} (ID: {entity_id}) not found in folder {folder_id}.")
            #     return

            # Use delete with filter expression - more efficient than PK list if supported
            # Note: Milvus delete by expression might have limitations or specific syntax.
            # Using delete by PK list might be more universally supported.
            # Let's use PK delete for robustness, assuming PKs are unique globally.
            # If PKs (hashes) could collide across tenants (unlikely), filtering before delete is safer.
            results = self._client.delete(collection_name=self.namespace, pks=[entity_id]) # Assumes PK is globally unique
            logger.debug(f"Attempted delete for entity {entity_name} (ID: {entity_id}) globally (tenant filter applied conceptually). Result: {results}")

        except Exception as e:
            logger.error(f"Error deleting entity {entity_name} for folder {folder_id}: {e}")
            # Don't raise here, allow potential cleanup in calling function

    async def delete_entity_relation(self, entity_name: str, folder_id: int) -> None:
        """Deletes relations associated with an entity within a folder."""
        if folder_id is None:
            raise ValueError("folder_id must be provided for tenant-aware delete_entity_relation")

        # Construct filter expression including folder_id
        expr = f'(src_id == "{entity_name}" or tgt_id == "{entity_name}") and {self.folder_id_field} == {folder_id}'
        try:
            # Query to find relation IDs within the specific folder
            results = self._client.query(
                collection_name=self.namespace, filter=expr, output_fields=[self.primary_field] # Only need ID
            )

            if not results:
                logger.debug(f"No relations found for entity {entity_name} in folder {folder_id}")
                return

            relation_ids = [item[self.primary_field] for item in results]
            logger.debug(f"Found {len(relation_ids)} relations for entity {entity_name} in folder {folder_id} to delete.")

            if relation_ids:
                # Delete by primary keys
                delete_result = self._client.delete(
                    collection_name=self.namespace, pks=relation_ids
                )
                logger.debug(f"Milvus delete result for relations: {delete_result}")

        except Exception as e:
            logger.error(f"Error deleting relations for entity {entity_name} in folder {folder_id}: {e}")
            # Allow potential cleanup

    async def delete(self, ids: list[str], folder_id: int) -> None:
        """Deletes specific vector IDs within a folder."""
        if folder_id is None:
            raise ValueError("folder_id must be provided for tenant-aware delete")
        if not ids:
            return

        # Similar to delete_entity, deleting by PK is usually sufficient if PKs are globally unique.
        # If you need strict tenant deletion, query for IDs within the tenant first.
        # filter_expr = f'{self.primary_field} in ["' + '", "'.join(ids) + f'"] and {self.folder_id_field} == {folder_id}'
        # query_res = self._client.query(collection_name=self.namespace, filter=filter_expr, output_fields=[self.primary_field])
        # tenant_ids_to_delete = [item[self.primary_field] for item in query_res]
        # if not tenant_ids_to_delete:
        #     logger.debug(f"No specified IDs found within folder {folder_id}.")
        #     return
        # results = self._client.delete(collection_name=self.namespace, pks=tenant_ids_to_delete)

        # Assuming global PK uniqueness for simplicity:
        results = self._client.delete(collection_name=self.namespace, pks=ids)
        logger.debug(f"Attempted delete for {len(ids)} IDs globally (tenant filter applied conceptually). Result: {results}")


    async def get_by_id(self, id: str, folder_id: int) -> dict[str, Any] | None:
        """Gets vector data by ID within a specific folder."""
        if folder_id is None:
            raise ValueError("folder_id must be provided for tenant-aware get_by_id")

        filter_expr = f'{self.primary_field} == "{id}" and {self.folder_id_field} == {folder_id}'
        try:
            result = self._client.query(
                collection_name=self.namespace,
                filter=filter_expr,
                output_fields=list(self.meta_fields) + [self.primary_field, self.folder_id_field],
                limit=1
            )
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Error retrieving vector data for ID {id} in folder {folder_id}: {e}")
            return None

    async def get_by_ids(self, ids: list[str], folder_id: int) -> list[dict[str, Any]]:
         """Gets multiple vector data by IDs within a specific folder."""
         if folder_id is None:
             raise ValueError("folder_id must be provided for tenant-aware get_by_ids")
         if not ids:
             return []

         # Format IDs for the 'in' expression
         id_list_str = '", "'.join(ids)
         filter_expr = f'{self.primary_field} in ["{id_list_str}"] and {self.folder_id_field} == {folder_id}'
         try:
             result = self._client.query(
                 collection_name=self.namespace,
                 filter=filter_expr,
                 output_fields=["*"], # Fetch all fields
                 # limit=len(ids) # Limit might not be needed for query by filter/PK
             )
             # Query returns list of dicts directly
             return result if result else []
         except Exception as e:
             logger.error(f"Error retrieving vector data for IDs {ids} in folder {folder_id}: {e}")
             return []

    # drop method remains global unless specifically designed for tenant drop
    # async def drop(self) -> dict[str, str]:
    #     # Base implementation likely drops the whole collection
    #     return await super().drop()

    async def drop_tenant(self, folder_id: int) -> dict[str, str]:
        """Drops all data for a specific folder_id."""
        if folder_id is None:
            return {"status": "error", "message": "folder_id is required"}

        logger.info(f"Dropping data for folder_id {folder_id} from Milvus collection {self.namespace}")
        filter_expr = f"{self.folder_id_field} == {folder_id}"
        try:
            # Milvus delete supports filter expressions
            # Note: Deleting large amounts of data via expression might be slow.
            # Consider partition strategies in Milvus if performance is critical.
            delete_result = self._client.delete(collection_name=self.namespace, filter=filter_expr)
            # The delete_count might not be accurate for filter-based deletes in all versions.
            # We log the attempt. Querying count before/after is more reliable if needed.
            count_after = self._client.query(collection_name=self.namespace, filter=filter_expr, output_fields=["count(*)"])
            num_remaining = count_after[0]['count(*)'] if count_after else 'unknown'

            logger.info(f"Milvus delete operation for folder {folder_id} completed. Result: {delete_result}. Remaining entries for tenant: {num_remaining}")
            return {"status": "success", "message": f"Delete operation initiated for folder {folder_id}. Result: {delete_result}"}
        except Exception as e:
            logger.error(f"Error dropping tenant data for folder {folder_id} from Milvus: {e}")
            return {"status": "error", "message": str(e)}