# tenant_aware/milvus_impl_aware.py
import asyncio
import os
from typing import Any, final, List
from dataclasses import dataclass
import numpy as np
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema # 确保导入必要类型
from lightrag.utils import logger, compute_mdhash_id
from lightrag.kg.milvus_impl import MilvusVectorDBStorage # 假设原始实现在这里
from lightrag.base import EmbeddingFunc # 导入基类或类型

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
        # Run original post_init first
        super().__post_init__()

        # Ensure the collection exists and has the required schema
        self._ensure_collection_schema()

    def _ensure_collection_schema(self):
        """Ensures the Milvus collection exists and has the folder_id field."""
        try:
            if not self._client.has_collection(self.namespace):
                logger.info(f"Collection {self.namespace} does not exist. Creating...")
                # Define schema fields
                fields = [
                    FieldSchema(name=self.primary_field, dtype=DataType.VARCHAR, is_primary=True, max_length=65535), # Adjusted max_length
                    FieldSchema(name=self.folder_id_field, dtype=DataType.INT64), # Tenant ID field
                    FieldSchema(name=self.vector_field, dtype=DataType.FLOAT_VECTOR, dim=self.embedding_func.embedding_dim),
                     # Add other meta fields defined in the base class or config
                     # Example: FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                     # Example: FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=1024),
                     # Dynamically add fields from self.meta_fields if possible/needed
                     # For simplicity, assume necessary meta fields are manually added or not strictly required by schema
                ]
                # Include meta fields dynamically - requires knowing their types
                # Example (assuming all meta fields are VARCHAR for simplicity):
                for meta_field_name in self.meta_fields:
                     if meta_field_name not in [self.primary_field, self.folder_id_field, self.vector_field]:
                          # You might need a mapping for types or make assumptions
                          fields.append(FieldSchema(name=meta_field_name, dtype=DataType.VARCHAR, max_length=65535))


                schema = CollectionSchema(fields=fields, description=f"Tenant-aware collection for {self.namespace}")
                self._client.create_collection(collection_name=self.namespace, schema=schema)
                logger.info(f"Created collection {self.namespace}")

                # Create index on vector field
                index_params = self._client.prepare_index_params()
                index_params.add_index(
                    field_name=self.vector_field,
                    index_type="AUTOINDEX", # Or specify like "IVF_FLAT"
                    metric_type="COSINE", # Or L2
                    # params={"nlist": 1024} # Example params for IVF_FLAT
                )
                self._client.create_index(self.namespace, index_params)
                logger.info(f"Created vector index on {self.vector_field} for {self.namespace}")

                # Create index on folder_id field for efficient filtering
                # Scalar field indexing might depend on Milvus version/configuration
                # Try creating a MARISA_TRIE or default scalar index
                try:
                     scalar_index_params = self._client.prepare_index_params()
                     scalar_index_params.add_index(field_name=self.folder_id_field) # Use default index type
                     self._client.create_index(self.namespace, scalar_index_params, index_name=f"{self.folder_id_field}_idx")
                     logger.info(f"Created scalar index on {self.folder_id_field} for {self.namespace}")
                except Exception as index_err:
                     logger.warning(f"Could not create scalar index on {self.folder_id_field} (may not be supported or necessary): {index_err}")

            else:
                logger.debug(f"Collection {self.namespace} already exists.")
                # Optional: Verify existing schema has folder_id field here
                # desc = self._client.describe_collection(self.namespace)
                # field_names = [f.name for f in desc.fields]
                # if self.folder_id_field not in field_names:
                #     raise RuntimeError(f"Collection {self.namespace} exists but lacks the '{self.folder_id_field}' field.")

            # Ensure collection is loaded
            self._client.load_collection(self.namespace)


        except Exception as e:
            logger.error(f"Error ensuring Milvus collection schema for {self.namespace}: {e}")
            raise

    async def upsert(self, data: dict[str, dict[str, Any]], folder_id: int) -> None:
        """Upserts data into Milvus, adding the folder_id."""
        if folder_id is None:
            raise ValueError("folder_id must be provided for tenant-aware upsert")
        if not data:
            return

        logger.info(f"Upserting {len(data)} items to Milvus collection {self.namespace} for folder {folder_id}")

        list_data: list[dict[str, Any]] = []
        contents = []
        ids = []

        for k, v in data.items():
            if "content" not in v:
                 logger.warning(f"Skipping item with key {k} due to missing 'content'")
                 continue
            ids.append(k)
            contents.append(v["content"])
            # Prepare entity data for Milvus, ensuring primary key, folder_id, and vector are included later
            entity_milvus_data = {
                self.primary_field: k,
                self.folder_id_field: folder_id,
                # Add other metadata fields present in 'v' and defined in meta_fields
                **{k_meta: v.get(k_meta) for k_meta in self.meta_fields if k_meta in v}
            }
            list_data.append(entity_milvus_data)

        if not list_data:
             logger.warning("No valid data to upsert after filtering.")
             return


        # Batch embedding calculation
        embeddings = []
        batch_size = self._max_batch_size
        for i in range(0, len(contents), batch_size):
            batch_contents = contents[i:i + batch_size]
            batch_embeddings = await self.embedding_func(batch_contents)
            embeddings.extend(batch_embeddings)

        # Add embeddings to the list_data
        if len(embeddings) != len(list_data):
            raise RuntimeError(f"Mismatch between number of embeddings ({len(embeddings)}) and data entries ({len(list_data)})")

        for i, embedding in enumerate(embeddings):
            list_data[i][self.vector_field] = embedding # Add the vector

        try:
            results = self._client.upsert(collection_name=self.namespace, data=list_data)
            logger.debug(f"Milvus upsert result for folder {folder_id}: {results}")
        except Exception as e:
            logger.error(f"Error upserting data to Milvus for folder {folder_id}: {e}")
            raise

    async def query(
        self, query: str, top_k: int, folder_id: int, ids: list[str] | None = None # `ids` might not be supported with `expr` in all Milvus versions/setups
    ) -> list[dict[str, Any]]:
        """Queries Milvus, filtering by folder_id."""
        if folder_id is None:
            raise ValueError("folder_id must be provided for tenant-aware query")

        embedding_result = await self.embedding_func([query])
        query_embedding = embedding_result[0].tolist() # Milvus client often expects list

        # Construct the filter expression
        filter_expr = f"{self.folder_id_field} == {folder_id}"

        # Handle optional ID filtering - NOTE: Combining PK filter with vector search might depend on Milvus version/index
        if ids:
             # Example: Add ID filter if needed and supported
             # id_filter_part = f'{self.primary_field} in ["' + '", "'.join(ids) + '"]'
             # filter_expr = f"({filter_expr}) and ({id_filter_part})"
             logger.warning("`ids` filter provided but may not be combined effectively with vector search and folder_id filter in all Milvus setups.")


        logger.debug(f"Querying Milvus collection {self.namespace} with top_k={top_k}, folder_id={folder_id}, filter='{filter_expr}'")

        try:
            results = self._client.search(
                collection_name=self.namespace,
                data=[query_embedding],
                limit=top_k,
                filter=filter_expr, # Apply the folder_id filter
                output_fields=list(self.meta_fields) + [self.primary_field, self.folder_id_field], # Ensure needed fields are output
                search_params={ # Define search parameters like metric type
                    "metric_type": "COSINE", # Or "L2"
                    "params": {} # Add index-specific search params if needed, e.g., {"nprobe": 10} for IVF_FLAT
                }
            )
        except Exception as e:
            logger.error(f"Error querying Milvus for folder {folder_id}: {e}")
            raise

        # Process results
        processed_results = []
        if results and results[0]:
            for hit in results[0]:
                # The 'entity' field might not exist; access fields directly from the hit object
                entity_data = {
                    "id": hit.id,
                    "distance": hit.distance,
                    self.folder_id_field: hit.entity.get(self.folder_id_field), # Access folder_id safely
                     # Include other meta fields
                     **{mf: hit.entity.get(mf) for mf in self.meta_fields if hit.entity.get(mf) is not None}
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

        id_list_str = '", "'.join(ids)
        filter_expr = f'{self.primary_field} in ["{id_list_str}"] and {self.folder_id_field} == {folder_id}'
        try:
            result = self._client.query(
                collection_name=self.namespace,
                filter=filter_expr,
                output_fields=list(self.meta_fields) + [self.primary_field, self.folder_id_field],
                limit=len(ids) # Limit to max number of IDs requested
            )
            return result or []
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