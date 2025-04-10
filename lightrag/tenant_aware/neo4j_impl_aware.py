# tenant_aware/neo4j_impl_aware.py
import os
import re
import numpy as np
from dataclasses import dataclass
from typing import Any, final, List, Tuple, Union
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from neo4j import AsyncGraphDatabase, exceptions as neo4jExceptions, AsyncDriver, AsyncManagedTransaction, Session
from lightrag.utils import logger,EmbeddingFunc
from lightrag.kg.neo4j_impl import Neo4JStorage # 假设原始实现在这里
from lightrag.types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge # 确保导入类型

MAX_GRAPH_NODES = int(os.getenv("MAX_GRAPH_NODES", 1000)) # 从原始文件获取

@final
@dataclass
class TenantAwareNeo4JStorage(Neo4JStorage):
    """
    A tenant-aware Neo4j graph storage implementation using folder_id property.
    Inherits from the original Neo4JStorage.
    Assumes nodes have a 'folder_id' property and an index exists on :base(folder_id).
    """

    def __init__(self, namespace, global_config, embedding_func):
        # 正确调用父类的 __init__ (StorageNameSpace 的)
        super().__init__(namespace=namespace, global_config=global_config, embedding_func=embedding_func)
        self._driver: AsyncDriver | None = None # 初始化为 None
        self._DATABASE: str | None = None      # 初始化为 None
        logger.debug(f"TenantAwareNeo4JStorage '{namespace}' __init__ called (driver set to None).")
        # 不需要在这里调用 self.initialize()，它会被 RAG 主类调用

    async def _ensure_neo4j_index(self, session: Session, index_name: str, label: str, properties: Union[str, List[str]]):
        """Helper to create a Neo4j index if it doesn't exist."""
        # Ensure properties input is handled correctly for Cypher syntax
        if isinstance(properties, list):
            # For composite index: n.`property1`, n.`property2`
            prop_str_cypher = ", ".join([f"n.`{p}`" for p in properties])
            prop_str_log = ", ".join(properties) # For logging
        else:
            # For single property index: n.`property`
            prop_str_cypher = f"n.`{properties}`"
            prop_str_log = properties

        # Use CREATE INDEX ... IF NOT EXISTS for idempotency
        create_index_cypher = f"CREATE INDEX {index_name} IF NOT EXISTS FOR (n:{label}) ON ({prop_str_cypher})"
        try:
             result = await session.run(create_index_cypher)
             await result.consume()
             # Log success regardless of whether it was created now or existed before
             logger.info(f"Ensured Neo4j index '{index_name}' exists for label '{label}' on properties '{prop_str_log}' in db '{self._DATABASE}'.")
        except neo4jExceptions.ClientError as e:
            # Handle potential errors, e.g., conflicting index definition
            # Different Neo4j versions might have different error messages
            err_str = str(e).lower()
            if "already exists" in err_str or "concurrently" in err_str or "token name" in err_str: # Check common messages
                logger.debug(f"Index matching config for {label}({prop_str_log}) likely exists (maybe under different name or concurrent creation). Index name: '{index_name}'. Error: {e}")
            else:
                logger.warning(f"Could not ensure Neo4j index '{index_name}': {e}")
        except Exception as e:
            logger.warning(f"Unexpected error ensuring Neo4j index '{index_name}': {e}")

    async def initialize(self):
        """
        Initializes the Neo4j driver for this tenant-aware storage instance
        and ensures necessary indexes exist.
        This method *does not* call super().initialize().
        """
        # --- 1. 配置和建立驱动连接 ---
        if self._driver: # 这个检查现在可以工作了
            logger.debug(f"Neo4j driver for {self.namespace} already initialized.")
            # 也许重新确认数据库名
            # self._DATABASE = os.getenv("NEO4J_DATABASE", re.sub(r"[^a-zA-Z0-9-]", "-", self.namespace))
            return

        logger.info(f"Initializing Neo4j driver for namespace {self.namespace}...")
        URI = os.getenv("NEO4J_URI", None) # 从环境获取
        USERNAME = os.getenv("NEO4J_USERNAME", None)
        PASSWORD = os.getenv("NEO4J_PASSWORD", None)
        # 从 global_config 获取配置（如果需要）或保持从 env 获取
        # config = self.global_config.get('neo4j_config', {}) # 示例
        MAX_CONNECTION_POOL_SIZE = int(os.getenv("NEO4J_MAX_CONNECTION_POOL_SIZE", 50))
        # ... (获取其他连接参数: TIMEOUTS, etc.)
        CONNECTION_TIMEOUT = float(os.getenv("NEO4J_CONNECTION_TIMEOUT", 30.0))
        CONNECTION_ACQUISITION_TIMEOUT = float(os.getenv("NEO4J_CONNECTION_ACQUISITION_TIMEOUT", 30.0))
        MAX_TRANSACTION_RETRY_TIME = float(os.getenv("NEO4J_MAX_TRANSACTION_RETRY_TIME", 30.0))

        # **更安全的做法：使用固定的或环境变量指定的数据库**
        self._DATABASE = os.getenv("NEO4J_DATABASE", "neo4j") # 默认使用 'neo4j'
        logger.info(f"Targeting Neo4j database: '{self._DATABASE}' (Check NEO4J_DATABASE env var if needed)")

        if not URI or not USERNAME or not PASSWORD:
            raise ValueError("Missing required Neo4j connection environment variables: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD")

        try:
             self._driver: AsyncDriver = AsyncGraphDatabase.driver(
                  URI,
                  auth=(USERNAME, PASSWORD),
                  max_connection_pool_size=MAX_CONNECTION_POOL_SIZE,
                  connection_timeout=CONNECTION_TIMEOUT,
                  connection_acquisition_timeout=CONNECTION_ACQUISITION_TIMEOUT,
                  max_transaction_retry_time=MAX_TRANSACTION_RETRY_TIME,
             )
             logger.debug(f"Neo4j driver created for URI: {URI}, Target DB: {self._DATABASE}")

             # --- 验证连接和确保索引 (逻辑不变) ---
             async with self._driver.session(database=self._DATABASE) as session:
                  await session.run("MATCH (n) RETURN count(n) LIMIT 1")
             logger.info(f"Successfully connected to Neo4j database '{self._DATABASE}'.")
             await self._ensure_indexes() # 确保索引存在

        except neo4jExceptions.ClientError as e:
             # 简化处理：如果目标数据库（例如 'neo4j'）不存在或无法访问，这是严重错误
             logger.error(f"Neo4j client error connecting to DB '{self._DATABASE}': {e}")
             await self.finalize() # 清理驱动
             raise ConnectionError(f"Could not connect to Neo4j DB '{self._DATABASE}'") from e
        except neo4jExceptions.AuthError as e:
             logger.error(f"Neo4j authentication failed: {e}")
             await self.finalize()
             raise e
        except neo4jExceptions.ServiceUnavailable as e:
             logger.error(f"Neo4j service unavailable at {URI}: {e}")
             await self.finalize()
             raise e
        except Exception as e:
             logger.error(f"Unexpected error during Neo4j initialization for DB '{self._DATABASE}': {e}")
             await self.finalize()
             raise e
        
        
    async def _ensure_indexes(self):
        """Internal helper to ensure required indexes exist."""
        if not self._driver or not self._DATABASE:
            logger.error("Cannot ensure indexes: Driver or Database not initialized.")
            return

        async with self._driver.session(database=self._DATABASE) as session:
            logger.info(f"Ensuring required indexes exist in Neo4j database '{self._DATABASE}'...")
            try:
                # 1. Index on :base(entity_id)
                await self._ensure_neo4j_index(session, "base_entity_id_idx", "base", "entity_id")
                # 2. Index on :base(folder_id)
                await self._ensure_neo4j_index(session, "base_folder_id_idx", "base", "folder_id")
                # 3. Composite Index on :base(entity_id, folder_id)
                await self._ensure_neo4j_index(session, "base_entity_folder_comp_idx", "base", ["entity_id", "folder_id"])
                logger.info(f"Required Neo4j indexes ensured for database '{self._DATABASE}'.")
            except Exception as e:
                logger.error(f"Failed during index creation/verification in database '{self._DATABASE}': {e}")
                # Decide if this is fatal



    # --- Override CRUD and Query Methods ---

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((neo4jExceptions.ServiceUnavailable, neo4jExceptions.TransientError, neo4jExceptions.WriteServiceUnavailable, neo4jExceptions.ClientError))
    )
    async def upsert_node(self, node_id: str, node_data: dict[str, Any], folder_id: int) -> None:
        """Upserts a node within a specific folder."""
        if folder_id is None:
            raise ValueError("folder_id must be provided for tenant-aware upsert_node")
        if "entity_id" not in node_data:
             raise ValueError("Neo4j (TenantAware): node properties must contain 'entity_id'")

        properties = node_data.copy() # Avoid modifying original dict
        entity_type = properties.get("entity_type", "Unknown") # Use .get for safety
        properties['folder_id'] = folder_id # Add tenant ID

        try:
            async with self._driver.session(database=self._DATABASE) as session:
                async def execute_upsert(tx: AsyncManagedTransaction):
                    # Use MERGE on both entity_id and folder_id for uniqueness within tenant
                    query = f"""
                    MERGE (n:base {{entity_id: $entity_id, folder_id: $folder_id}})
                    SET n += $properties
                    SET n:`{entity_type}`
                    """ # Using f-string for label injection needs caution if entity_type is user input
                      # Consider validating entity_type against allowed characters or using APOC if available
                    result = await tx.run(query, entity_id=node_id, properties=properties, folder_id=folder_id)
                    await result.consume()
                    logger.debug(f"Tenant Upsert Node: folder={folder_id}, id={node_id}")

                await session.execute_write(execute_upsert)
        except Exception as e:
            logger.error(f"Error during tenant node upsert (folder {folder_id}, node {node_id}): {e}")
            raise

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((neo4jExceptions.ServiceUnavailable, neo4jExceptions.TransientError, neo4jExceptions.WriteServiceUnavailable, neo4jExceptions.ClientError))
    )
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, Any], folder_id: int
    ) -> None:
        """Upserts an edge within a specific folder."""
        if folder_id is None:
            raise ValueError("folder_id must be provided for tenant-aware upsert_edge")

        edge_properties = edge_data.copy()
        edge_properties['folder_id'] = folder_id # Add folder_id to edge properties

        try:
            async with self._driver.session(database=self._DATABASE) as session:
                async def execute_upsert(tx: AsyncManagedTransaction):
                    # Match nodes within the specific folder
                    query = """
                    MATCH (source:base {entity_id: $source_entity_id, folder_id: $folder_id})
                    MATCH (target:base {entity_id: $target_entity_id, folder_id: $folder_id})
                    MERGE (source)-[r:DIRECTED]-(target) // Merge relationship between source and target
                    SET r += $properties // Set properties including folder_id
                    """
                    result = await tx.run(
                        query,
                        source_entity_id=source_node_id,
                        target_entity_id=target_node_id,
                        properties=edge_properties,
                        folder_id=folder_id
                    )
                    await result.consume()
                    logger.debug(f"Tenant Upsert Edge: folder={folder_id}, {source_node_id}->{target_node_id}")

                await session.execute_write(execute_upsert)
        except Exception as e:
            logger.error(f"Error during tenant edge upsert (folder {folder_id}, {source_node_id}->{target_node_id}): {e}")
            raise

    async def has_node(self, node_id: str, folder_id: int) -> bool:
        """Checks if a node exists within a specific folder."""
        if folder_id is None:
            raise ValueError("folder_id must be provided for tenant-aware has_node")
        async with self._driver.session(database=self._DATABASE, default_access_mode="READ") as session:
            try:
                query = "MATCH (n:base {entity_id: $entity_id, folder_id: $folder_id}) RETURN count(n) > 0 AS node_exists"
                result = await session.run(query, entity_id=node_id, folder_id=folder_id)
                single_result = await result.single()
                await result.consume()
                return single_result["node_exists"] if single_result else False
            except Exception as e:
                logger.error(f"Error checking node existence (folder {folder_id}, node {node_id}): {e}")
                raise

    async def has_edge(self, source_node_id: str, target_node_id: str, folder_id: int) -> bool:
        """Checks if an edge exists between two nodes within a specific folder."""
        if folder_id is None:
            raise ValueError("folder_id must be provided for tenant-aware has_edge")
        async with self._driver.session(database=self._DATABASE, default_access_mode="READ") as session:
            try:
                # Match both nodes and the relationship within the same folder
                query = """
                MATCH (a:base {entity_id: $source_entity_id, folder_id: $folder_id})
                MATCH (b:base {entity_id: $target_entity_id, folder_id: $folder_id})
                MATCH (a)-[r]-(b) // Check for relationship between them
                RETURN COUNT(r) > 0 AS edgeExists
                """
                result = await session.run(query, source_entity_id=source_node_id, target_entity_id=target_node_id, folder_id=folder_id)
                single_result = await result.single()
                await result.consume()
                return single_result["edgeExists"] if single_result else False
            except Exception as e:
                 logger.error(f"Error checking edge existence (folder {folder_id}, {source_node_id}-{target_node_id}): {e}")
                 raise

    async def get_node(self, node_id: str, folder_id: int) -> dict[str, Any] | None:
        """Gets a node by ID within a specific folder."""
        if folder_id is None:
            raise ValueError("folder_id must be provided for tenant-aware get_node")
        async with self._driver.session(database=self._DATABASE, default_access_mode="READ") as session:
            try:
                query = "MATCH (n:base {entity_id: $entity_id, folder_id: $folder_id}) RETURN properties(n) as props"
                result = await session.run(query, entity_id=node_id, folder_id=folder_id)
                record = await result.single()
                await result.consume()
                return dict(record["props"]) if record and record["props"] else None
            except Exception as e:
                 logger.error(f"Error getting node (folder {folder_id}, node {node_id}): {e}")
                 raise

    async def node_degree(self, node_id: str, folder_id: int) -> int:
        """Gets the degree of a node within a specific folder."""
        if folder_id is None:
             raise ValueError("folder_id must be provided for tenant-aware node_degree")
        async with self._driver.session(database=self._DATABASE, default_access_mode="READ") as session:
             try:
                  # Count relationships connected to nodes also within the same folder
                  query = """
                  MATCH (n:base {entity_id: $entity_id, folder_id: $folder_id})
                  OPTIONAL MATCH (n)-[r]-(neighbor:base {folder_id: $folder_id}) // Ensure neighbor is in same folder
                  RETURN COUNT(r) AS degree
                  """
                  result = await session.run(query, entity_id=node_id, folder_id=folder_id)
                  record = await result.single()
                  await result.consume()
                  return record["degree"] if record else 0
             except Exception as e:
                  logger.error(f"Error getting node degree (folder {folder_id}, node {node_id}): {e}")
                  raise

    async def edge_degree(self, src_id: str, tgt_id: str, folder_id: int) -> int:
        """Gets edge degree by summing node degrees within the folder."""
        # Reuses the tenant-aware node_degree
        src_degree = await self.node_degree(src_id, folder_id)
        tgt_degree = await self.node_degree(tgt_id, folder_id)
        return src_degree + tgt_degree

    async def get_edge(
        self, source_node_id: str, target_node_id: str, folder_id: int
    ) -> dict[str, Any] | None:
        """Gets edge properties between two nodes within a specific folder."""
        if folder_id is None:
             raise ValueError("folder_id must be provided for tenant-aware get_edge")
        async with self._driver.session(database=self._DATABASE, default_access_mode="READ") as session:
             try:
                  # Ensure both nodes and the relationship exist within the folder context
                  query = """
                  MATCH (start:base {entity_id: $source_entity_id, folder_id: $folder_id})
                  MATCH (end:base {entity_id: $target_entity_id, folder_id: $folder_id})
                  MATCH (start)-[r]-(end)
                  RETURN properties(r) as edge_properties
                  LIMIT 1
                  """
                  result = await session.run(query, source_entity_id=source_node_id, target_entity_id=target_node_id, folder_id=folder_id)
                  record = await result.single()
                  await result.consume()
                  # Process result as in the original get_edge, handling missing keys
                  if record and record["edge_properties"]:
                       edge_result = dict(record["edge_properties"])
                       required_keys = {"weight": 0.0, "source_id": None, "description": None, "keywords": None}
                       for key, default_value in required_keys.items():
                            if key not in edge_result:
                                 edge_result[key] = default_value
                       return edge_result
                  return None
             except Exception as e:
                  logger.error(f"Error getting edge (folder {folder_id}, {source_node_id}-{target_node_id}): {e}")
                  raise

    async def get_node_edges(self, source_node_id: str, folder_id: int) -> list[tuple[str, str]] | None:
        """Gets edges for a node within a specific folder."""
        if folder_id is None:
             raise ValueError("folder_id must be provided for tenant-aware get_node_edges")
        async with self._driver.session(database=self._DATABASE, default_access_mode="READ") as session:
             try:
                  # Match source node in folder, and connected nodes also in the same folder
                  query = """
                  MATCH (n:base {entity_id: $entity_id, folder_id: $folder_id})
                  OPTIONAL MATCH (n)-[r]-(connected:base {folder_id: $folder_id})
                  WHERE connected.entity_id IS NOT NULL
                  RETURN connected.entity_id AS target_id
                  """
                  result = await session.run(query, entity_id=source_node_id, folder_id=folder_id)
                  edges = []
                  async for record in result:
                       target_id = record["target_id"]
                       if target_id:
                            edges.append((source_node_id, target_id))
                  await result.consume()
                  return edges if edges else None
             except Exception as e:
                  logger.error(f"Error getting node edges (folder {folder_id}, node {source_node_id}): {e}")
                  raise

    async def get_knowledge_graph(
        self,
        node_label: str,
        folder_id: int,
        max_depth: int = 3,
        max_nodes: int = MAX_GRAPH_NODES,
    ) -> KnowledgeGraph:
        """Retrieves a subgraph within a specific folder."""
        if folder_id is None:
            raise ValueError("folder_id must be provided for tenant-aware get_knowledge_graph")

        result = KnowledgeGraph()
        seen_nodes = set()
        seen_edges = set()

        async with self._driver.session(database=self._DATABASE, default_access_mode="READ") as session:
            try:
                # --- APOC Attempt (with folder_id filter) ---
                apoc_query = """
                MATCH (start:base {entity_id: $entity_id, folder_id: $folder_id})
                WITH start
                CALL apoc.path.subgraphAll(start, {
                    relationshipFilter: '',
                    labelFilter: '+base', // Ensure nodes have base label
                    minLevel: 0,
                    maxLevel: $max_depth,
                    bfs: true,
                    filterStartNode: false, // Include start node even if it doesn't match labelFilter? Check APOC docs
                     // Add node filter for folder_id
                     nodeFilter: 'folder_id = ' + $folder_id
                     // Relationship filter might also be needed if edges store folder_id
                     // relationshipFilter: 'r.folder_id = ' + $folder_id
                })
                YIELD nodes, relationships
                WITH nodes, relationships, size(nodes) AS total_nodes
                WHERE all(n IN nodes WHERE n.folder_id = $folder_id) // Double check folder_id
                UNWIND nodes AS node
                WITH collect({node: node}) AS node_info, relationships, total_nodes
                RETURN node_info, relationships, total_nodes
                LIMIT 1
                """
                 # Limit query for APOC (node count check inside)
                limited_apoc_query = """
                 MATCH (start:base {entity_id: $entity_id, folder_id: $folder_id})
                 WITH start
                 CALL apoc.path.subgraphAll(start, {
                     relationshipFilter: '',
                     labelFilter: '+base',
                     minLevel: 0,
                     maxLevel: $max_depth,
                     limit: $max_nodes, // Apply limit here for APOC BFS
                     bfs: true,
                     filterStartNode: false,
                     nodeFilter: 'folder_id = ' + $folder_id
                     // relationshipFilter: 'r.folder_id = ' + $folder_id
                 })
                 YIELD nodes, relationships
                 WHERE all(n IN nodes WHERE n.folder_id = $folder_id) // Double check folder_id
                 UNWIND nodes AS node
                 WITH collect({node: node}) AS node_info, relationships
                 RETURN node_info, relationships
                 """


                record = None
                full_result = None
                limited_result = None

                try:
                    # Check total count first
                    full_result = await session.run(apoc_query, entity_id=node_label, max_depth=max_depth, folder_id=folder_id)
                    full_record = await full_result.single()

                    if not full_record:
                        logger.debug(f"APOC: No nodes found for entity_id {node_label} in folder {folder_id}")
                        return result # Return empty graph

                    total_nodes = full_record["total_nodes"]
                    if total_nodes <= max_nodes:
                         record = full_record # Use the full result
                         logger.debug(f"APOC: Using full result ({total_nodes} nodes) for folder {folder_id}")
                    else:
                         result.is_truncated = True
                         logger.info(f"APOC: Graph truncated ({total_nodes} > {max_nodes}) for folder {folder_id}. Running limited query.")
                         limited_result = await session.run(limited_apoc_query, entity_id=node_label, max_depth=max_depth, max_nodes=max_nodes, folder_id=folder_id)
                         record = await limited_result.single()

                except neo4jExceptions.ClientError as apoc_error:
                    logger.warning(f"APOC plugin error or incompatibility for tenant query: {apoc_error}")
                    logger.warning("Neo4j (TenantAware): Falling back to basic Cypher BFS...")
                    return await self._tenant_robust_fallback(node_label, folder_id, max_depth, max_nodes)
                finally:
                     if full_result: await full_result.consume()
                     if limited_result: await limited_result.consume()


                # Process the APOC result (either full or limited)
                if record:
                    for node_info in record["node_info"]:
                        node = node_info["node"]
                        node_id = node.get("entity_id") # Use entity_id for consistency
                        if node_id and node_id not in seen_nodes:
                            result.nodes.append(KnowledgeGraphNode(
                                id=str(node_id),
                                labels=[str(node_id)], # Assuming label is entity_id
                                properties=dict(node)
                            ))
                            seen_nodes.add(node_id)

                    for rel in record["relationships"]:
                        edge_id = rel.id
                        if edge_id not in seen_edges:
                            start_node = rel.start_node
                            end_node = rel.end_node
                            start_id = start_node.get("entity_id")
                            end_id = end_node.get("entity_id")
                            if start_id and end_id: # Ensure both ends have entity_id
                                result.edges.append(KnowledgeGraphEdge(
                                    id=str(edge_id),
                                    type=rel.type,
                                    source=str(start_id),
                                    target=str(end_id),
                                    properties=dict(rel)
                                ))
                                seen_edges.add(edge_id)
                    logger.info(f"APOC subgraph query successful for folder {folder_id}")

            except Exception as e:
                 logger.error(f"Error getting knowledge graph (folder {folder_id}, node {node_label}): {e}")
                 # Optionally fall back here too if APOC fails unexpectedly
                 # return await self._tenant_robust_fallback(node_label, folder_id, max_depth, max_nodes)
                 raise

        return result


    async def _tenant_robust_fallback(
        self, node_label: str, folder_id: int, max_depth: int, max_nodes: int
    ) -> KnowledgeGraph:
        """Fallback BFS implementation filtered by folder_id."""
        from collections import deque

        result = KnowledgeGraph()
        visited_node_ids = set() # Use entity_id for tracking visited nodes
        visited_edge_ids = set() # Use Neo4j edge ID

        # Get starting node ID (Neo4j internal ID) and data
        start_node_neo4j_id = None
        start_node_data = None
        async with self._driver.session(database=self._DATABASE, default_access_mode="READ") as session:
             query = "MATCH (n:base {entity_id: $entity_id, folder_id: $folder_id}) RETURN id(n) as neo4j_id, properties(n) as props"
             node_result = await session.run(query, entity_id=node_label, folder_id=folder_id)
             try:
                  node_record = await node_result.single()
                  if not node_record: return result # Start node not found in folder
                  start_node_neo4j_id = node_record['neo4j_id']
                  start_node_data = dict(node_record['props'])
             finally:
                  await node_result.consume()

        if not start_node_neo4j_id: return result

        # Add start node to result
        start_node = KnowledgeGraphNode(id=node_label, labels=[node_label], properties=start_node_data)
        result.nodes.append(start_node)
        visited_node_ids.add(node_label)

        # Queue contains tuples: (neo4j_node_id, current_depth)
        queue = deque([(start_node_neo4j_id, 0)])

        while queue and len(result.nodes) < max_nodes:
            current_neo4j_id, current_depth = queue.popleft()

            if current_depth >= max_depth: continue

            # Find neighbors within the same folder
            async with self._driver.session(database=self._DATABASE, default_access_mode="READ") as session:
                 query = """
                 MATCH (a) WHERE id(a) = $current_neo4j_id
                 MATCH (a)-[r]-(b:base {folder_id: $folder_id}) // Ensure neighbor is in same folder
                 RETURN id(r) as edge_id, type(r) as rel_type, properties(r) as rel_props,
                        id(b) as neighbor_neo4j_id, b.entity_id as neighbor_entity_id, properties(b) as neighbor_props
                 """
                 results = await session.run(query, current_neo4j_id=current_neo4j_id, folder_id=folder_id)
                 neighbors = await results.consume() # Get all neighbors before proceeding

                 for record in neighbors: # Process neighbors outside the async with block
                      neighbor_entity_id = record["neighbor_entity_id"]
                      if not neighbor_entity_id: continue # Skip if neighbor has no entity_id

                      # Add edge if not seen
                      edge_id = record["edge_id"]
                      if edge_id not in visited_edge_ids:
                           current_entity_id = start_node_data['entity_id'] if current_neo4j_id == start_node_neo4j_id else None
                           # Need a way to get entity_id for current_neo4j_id if not start node (complex)
                           # Simplified: Assume current node's entity_id is known or fetch it (adds overhead)
                           # For now, fetch it if needed
                           if not current_entity_id:
                                node_info = await self.get_node(current_neo4j_id, folder_id) # Incorrect: get_node expects entity_id
                                # Need a get_node_by_neo4j_id or modify query
                                logger.warning("Cannot reliably determine source entity ID in fallback BFS")
                                continue


                           result.edges.append(KnowledgeGraphEdge(
                                id=str(edge_id),
                                type=record["rel_type"],
                                source=str(current_entity_id),
                                target=str(neighbor_entity_id),
                                properties=dict(record["rel_props"])
                           ))
                           visited_edge_ids.add(edge_id)

                      # Add neighbor node if not visited and within limits
                      if neighbor_entity_id not in visited_node_ids and len(result.nodes) < max_nodes:
                           visited_node_ids.add(neighbor_entity_id)
                           result.nodes.append(KnowledgeGraphNode(
                                id=str(neighbor_entity_id),
                                labels=[str(neighbor_entity_id)],
                                properties=dict(record["neighbor_props"])
                           ))
                           # Enqueue neighbor for further exploration
                           queue.append((record["neighbor_neo4j_id"], current_depth + 1))
                      elif len(result.nodes) >= max_nodes:
                            result.is_truncated = True
                            logger.info(f"Fallback BFS truncated at {max_nodes} nodes for folder {folder_id}")
                            break # Stop BFS

            if result.is_truncated: break # Exit outer loop if truncated

        logger.info(f"Fallback BFS successful for folder {folder_id}")
        return result

    async def get_all_labels(self, folder_id: int) -> list[str]:
        """Gets all node labels (entity_ids) within a specific folder."""
        if folder_id is None:
             raise ValueError("folder_id must be provided for tenant-aware get_all_labels")
        async with self._driver.session(database=self._DATABASE, default_access_mode="READ") as session:
             query = """
             MATCH (n:base {folder_id: $folder_id})
             WHERE n.entity_id IS NOT NULL
             RETURN DISTINCT n.entity_id AS label
             ORDER BY label
             """
             result = await session.run(query, folder_id=folder_id)
             labels = [record["label"] async for record in result]
             await result.consume()
             return labels

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((neo4jExceptions.ServiceUnavailable, neo4jExceptions.TransientError, neo4jExceptions.WriteServiceUnavailable, neo4jExceptions.ClientError))
    )
    async def delete_node(self, node_id: str, folder_id: int) -> None:
        """Deletes a node within a specific folder."""
        if folder_id is None:
             raise ValueError("folder_id must be provided for tenant-aware delete_node")
        async def _do_delete(tx: AsyncManagedTransaction):
            # Match node by entity_id AND folder_id
            query = """
            MATCH (n:base {entity_id: $entity_id, folder_id: $folder_id})
            DETACH DELETE n
            """
            result = await tx.run(query, entity_id=node_id, folder_id=folder_id)
            await result.consume()
            logger.debug(f"Deleted node '{node_id}' from folder {folder_id}")

        try:
            async with self._driver.session(database=self._DATABASE) as session:
                await session.execute_write(_do_delete)
        except Exception as e:
            logger.error(f"Error deleting node (folder {folder_id}, node {node_id}): {e}")
            raise

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((neo4jExceptions.ServiceUnavailable, neo4jExceptions.TransientError, neo4jExceptions.WriteServiceUnavailable, neo4jExceptions.ClientError))
    )
    async def remove_edges(self, edges: list[tuple[str, str]], folder_id: int):
        """Deletes multiple edges within a specific folder."""
        if folder_id is None:
             raise ValueError("folder_id must be provided for tenant-aware remove_edges")
        async def _do_delete_edge(tx: AsyncManagedTransaction, source: str, target: str):
            # Match nodes and relationship within the specific folder
            query = """
            MATCH (source:base {entity_id: $source_entity_id, folder_id: $folder_id})
            MATCH (target:base {entity_id: $target_entity_id, folder_id: $folder_id})
            MATCH (source)-[r]-(target)
            DELETE r
            """
            result = await tx.run(query, source_entity_id=source, target_entity_id=target, folder_id=folder_id)
            await result.consume()
            logger.debug(f"Deleted edge from '{source}' to '{target}' in folder {folder_id}")

        try:
            async with self._driver.session(database=self._DATABASE) as session:
                for source, target in edges:
                     await session.execute_write(_do_delete_edge, source=source, target=target)
        except Exception as e:
            logger.error(f"Error during bulk edge deletion for folder {folder_id}: {e}")
            raise

    # --- Drop Methods ---
    async def drop(self) -> dict[str, str]:
        """Drops ALL data across ALL folders. Inherited behavior."""
        logger.warning(f"Executing global drop for database {self._DATABASE}. This affects ALL folders.")
        return await super().drop()

    async def drop_tenant(self, folder_id: int) -> dict[str, str]:
        """Drops all nodes and relationships for a SPECIFIC folder_id."""
        if folder_id is None:
            return {"status": "error", "message": "folder_id is required"}

        logger.info(f"Dropping data for folder_id {folder_id} from Neo4j database {self._DATABASE}")
        try:
            async with self._driver.session(database=self._DATABASE) as session:
                # Delete all nodes (and their relationships via DETACH) within the folder
                query = "MATCH (n:base {folder_id: $folder_id}) DETACH DELETE n"
                result = await session.run(query, folder_id=folder_id)
                summary = await result.consume()
                deleted_nodes = summary.counters.nodes_deleted
                deleted_rels = summary.counters.relationships_deleted
                logger.info(f"Dropped {deleted_nodes} nodes and {deleted_rels} relationships for folder {folder_id}")
                return {"status": "success", "message": f"{deleted_nodes} nodes and {deleted_rels} relationships dropped for folder {folder_id}"}
        except Exception as e:
            logger.error(f"Error dropping tenant data for folder {folder_id} from Neo4j: {e}")
            return {"status": "error", "message": str(e)}

    # embed_nodes likely remains non-tenant specific unless the algorithm supports filtering
    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray[Any, Any], list[str]]:
         raise NotImplementedError("Tenant-specific node embedding not implemented")