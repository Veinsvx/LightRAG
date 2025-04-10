# tenant_aware/redis_impl_aware.py
import os
import json
from dataclasses import dataclass
from typing import Any, Set ,Dict
from redis.asyncio import Redis, ConnectionPool
from redis.exceptions import RedisError, ConnectionError
from lightrag.utils import logger
from lightrag.kg.redis_impl import RedisKVStorage # 假设原始实现在这里

# Constants from original file (if needed)
MAX_CONNECTIONS = 50
SOCKET_TIMEOUT = 5.0
SOCKET_CONNECT_TIMEOUT = 3.0

@dataclass
class TenantAwareRedisKVStorage(RedisKVStorage):
    """
    A tenant-aware Redis KV storage implementation that prefixes keys with folder_id.
    Inherits from the original RedisKVStorage.
    """

    # Override __init__ or __post_init__ if base class needs modification,
    # but likely we just need to override methods.
    # Keep the original __post_init__ logic for setting up the pool.

    def _get_tenant_key(self, key: str, folder_id: int) -> str:
        """Constructs the Redis key with folder_id prefix."""
        if folder_id is None:
            raise ValueError("folder_id is required for tenant-aware Redis operations")
        # Ensure key is a string
        safe_key = str(key)
        return f"folder:{folder_id}:{self.namespace}:{safe_key}"

    async def get_by_id(self, id: str, folder_id: int) -> dict[str, Any] | None:
        """Gets value by ID within a specific folder."""
        tenant_key = self._get_tenant_key(id, folder_id)
        # Call the original get_by_id logic but pass the prefixed key
        # Need to adapt if the base class method doesn't just take the key string.
        # Assuming base class get method takes the raw key:
        async with self._get_redis_connection() as redis:
            try:
                data = await redis.get(tenant_key)
                return json.loads(data) if data else None
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error for key {tenant_key}: {e}")
                return None
            except Exception as e:
                logger.error(f"Error in get_by_id for key {tenant_key}: {e}")
                raise

    async def get_by_ids(self, ids: list[str], folder_id: int) -> list[dict[str, Any]]:
        """Gets multiple values by IDs within a specific folder."""
        if not ids:
            return []
        async with self._get_redis_connection() as redis:
            try:
                pipe = redis.pipeline()
                tenant_keys = [self._get_tenant_key(id, folder_id) for id in ids]
                for tenant_key in tenant_keys:
                    pipe.get(tenant_key)
                results = await pipe.execute()
                # Deserialize results, handling potential None values and JSON errors
                deserialized_results = []
                for result in results:
                    if result:
                        try:
                            deserialized_results.append(json.loads(result))
                        except json.JSONDecodeError:
                            deserialized_results.append(None) # Handle malformed JSON
                    else:
                        deserialized_results.append(None)
                return deserialized_results
            except Exception as e:
                logger.error(f"Error in get_by_ids for folder {folder_id}: {e}")
                raise # Re-raise after logging

    async def filter_keys(self, keys: Set[str], folder_id: int) -> Set[str]:
        """Returns keys that DO NOT exist within the specific folder."""
        if not keys:
            return set()
        async with self._get_redis_connection() as redis:
            pipe = redis.pipeline()
            tenant_keys_map = {self._get_tenant_key(key, folder_id): key for key in keys}
            for tenant_key in tenant_keys_map.keys():
                pipe.exists(tenant_key)
            results = await pipe.execute()

            existing_tenant_keys = [list(tenant_keys_map.keys())[i] for i, exists in enumerate(results) if exists]
            existing_original_keys = {tenant_keys_map[tk] for tk in existing_tenant_keys}

            return keys - existing_original_keys # Return keys that were NOT found

    async def upsert(self, data: dict[str, dict[str, Any]], folder_id: int) -> None:
        """Upserts data within a specific folder."""
        if not data:
            return
        logger.info(f"Upserting {len(data)} items to namespace {self.namespace} for folder {folder_id}")
        async with self._get_redis_connection() as redis:
            try:
                pipe = redis.pipeline()
                for k, v in data.items():
                    tenant_key = self._get_tenant_key(k, folder_id)
                    # Ensure value is serializable
                    try:
                         serialized_value = json.dumps(v)
                    except TypeError as e:
                         logger.error(f"Cannot serialize value for key {k} in folder {folder_id}: {e}")
                         # Decide how to handle: skip? raise error? log and continue?
                         continue # Skip this key-value pair
                    pipe.set(tenant_key, serialized_value)
                await pipe.execute()
            except json.JSONEncodeError as e:
                logger.error(f"JSON encode error during upsert for folder {folder_id}: {e}")
                raise
            except Exception as e:
                logger.error(f"Error in upsert for folder {folder_id}: {e}")
                raise

    async def delete(self, ids: list[str], folder_id: int) -> None:
        """Deletes entries by IDs within a specific folder."""
        if not ids:
            return
        async with self._get_redis_connection() as redis:
            pipe = redis.pipeline()
            tenant_keys = [self._get_tenant_key(id, folder_id) for id in ids]
            if tenant_keys:
                 pipe.delete(*tenant_keys) # Unpack keys for delete
                 results = await pipe.execute()
                 # Redis returns the number of keys deleted per call,
                 # For pipeline, it's often a list of results per command.
                 # Assuming delete returns 1 if key existed, 0 otherwise.
                 deleted_count = sum(results) if isinstance(results, list) else results
                 logger.info(f"Attempted to delete {len(ids)} keys for folder {folder_id}. Redis deleted {deleted_count} keys.")
            else:
                 logger.info(f"No keys to delete for folder {folder_id}.")


    async def drop_cache_by_modes(self, modes: list[str] | None = None, folder_id: int = None) -> bool:
        """Deletes cache entries for specific modes within a folder."""
        if not modes or folder_id is None:
             logger.warning("Modes or folder_id missing for drop_cache_by_modes")
             return False
        # In Redis implementation, modes might correspond to top-level keys
        # or prefixes. Here, we assume modes are treated as keys directly.
        tenant_mode_keys = [self._get_tenant_key(mode, folder_id) for mode in modes]
        try:
            await self.delete(modes, folder_id) # Reuse the tenant-aware delete
            logger.info(f"Successfully dropped cache modes {modes} for folder {folder_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to drop cache modes {modes} for folder {folder_id}: {e}")
            return False

    async def drop(self) -> dict[str, str]:
        """Drops ALL keys across ALL folders for the specific namespace. Use with caution."""
        logger.warning(f"Executing global drop for namespace {self.namespace}. This will affect ALL folders.")
        # Call the original drop method which likely deletes based on namespace pattern
        return await super().drop()

    async def drop_tenant(self, folder_id: int) -> dict[str, str]:
         """Drops all keys for a SPECIFIC folder within the namespace."""
         if folder_id is None:
              return {"status": "error", "message": "folder_id is required to drop a tenant"}

         pattern = f"folder:{folder_id}:{self.namespace}:*"
         logger.info(f"Dropping all keys matching pattern: {pattern}")
         async with self._get_redis_connection() as redis:
              try:
                   keys_to_delete = [key async for key in redis.scan_iter(match=pattern)]
                   if not keys_to_delete:
                        logger.info(f"No keys found for folder {folder_id} in namespace {self.namespace}")
                        return {"status": "success", "message": "no keys found for tenant"}

                   pipe = redis.pipeline()
                   for key in keys_to_delete:
                        pipe.delete(key)
                   results = await pipe.execute()
                   deleted_count = sum(results) # Assuming delete returns 1 per deleted key
                   logger.info(f"Dropped {deleted_count} keys for folder {folder_id} in namespace {self.namespace}")
                   return {"status": "success", "message": f"{deleted_count} keys dropped for folder {folder_id}"}
              except Exception as e:
                   logger.error(f"Error dropping tenant data for folder {folder_id}: {e}")
                   return {"status": "error", "message": str(e)}

    # get_all_in_folder 为了防止lightrag在adelete_by_doc_id中调用时报错，额外添加，但是应该重新实现，目前功能效果有问题。
    async def get_all_in_folder(self, folder_id: int) -> Dict[str, Any]:
        """Fetches all key-value pairs for a specific folder within the namespace."""
        if folder_id is None:
            raise ValueError("folder_id required for get_all_in_folder")

        all_data = {}
        pattern = f"folder:{folder_id}:{self.namespace}:*" # 构建匹配模式
        async with self._get_redis_connection() as redis:
            try:
                async for key in redis.scan_iter(match=pattern):
                    # 去掉前缀，得到原始 key
                    original_key_parts = key.split(':')
                    original_key = original_key_parts[-1] if len(original_key_parts) > 2 else key # 简单的去前缀逻辑
                    try:
                        value_str = await redis.get(key)
                        if value_str:
                            all_data[original_key] = json.loads(value_str)
                        else:
                            all_data[original_key] = None # 或者跳过 None 值
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode JSON for key: {key}")
                        all_data[original_key] = None # 或标记为错误
                    except Exception as e:
                        logger.error(f"Error fetching/decoding key {key}: {e}")
            except Exception as e:
                logger.error(f"Error during SCAN for pattern {pattern}: {e}")
                raise
        return all_data

    # --- Add embedding_func parameter to match base class ---
    # The base class __init__ might handle this, but let's ensure it's present.
    #embedding_func: EmbeddingFunc | None = None # Allow None if not always needed

    # We might need to explicitly call the superclass __init__ or __post_init__
    # if it does important setup. Let's assume the @dataclass handles it for now.
    # If errors occur, we might need:
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     # Add any tenant-specific init logic here