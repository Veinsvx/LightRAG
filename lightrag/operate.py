from __future__ import annotations

import asyncio
import traceback
import json
import re
import os
from typing import Any, AsyncIterator, List, Tuple, Dict, Union, Callable
from collections import Counter, defaultdict


from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
    compute_args_hash,
    handle_cache,#utils中已修改为多租户形式
    save_to_cache,#utils中已修改为多租户形式
    CacheData,
    statistic_data,
    get_conversation_turns,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS
import time
from dotenv import load_dotenv

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


def chunking_by_token_size(
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
    tiktoken_model: str = "gpt-4o",
) -> list[dict[str, Any]]:
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results: list[dict[str, Any]] = []
    if split_by_character:
        raw_chunks = content.split(split_by_character)
        new_chunks = []
        if split_by_character_only:
            for chunk in raw_chunks:
                _tokens = encode_string_by_tiktoken(chunk, model_name=tiktoken_model)
                new_chunks.append((len(_tokens), chunk))
        else:
            for chunk in raw_chunks:
                _tokens = encode_string_by_tiktoken(chunk, model_name=tiktoken_model)
                if len(_tokens) > max_token_size:
                    for start in range(
                        0, len(_tokens), max_token_size - overlap_token_size
                    ):
                        chunk_content = decode_tokens_by_tiktoken(
                            _tokens[start : start + max_token_size],
                            model_name=tiktoken_model,
                        )
                        new_chunks.append(
                            (min(max_token_size, len(_tokens) - start), chunk_content)
                        )
                else:
                    new_chunks.append((len(_tokens), chunk))
        for index, (_len, chunk) in enumerate(new_chunks):
            results.append(
                {
                    "tokens": _len,
                    "content": chunk.strip(),
                    "chunk_order_index": index,
                }
            )
    else:
        for index, start in enumerate(
            range(0, len(tokens), max_token_size - overlap_token_size)
        ):
            chunk_content = decode_tokens_by_tiktoken(
                tokens[start : start + max_token_size], model_name=tiktoken_model
            )
            results.append(
                {
                    "tokens": min(max_token_size, len(tokens) - start),
                    "content": chunk_content.strip(),
                    "chunk_order_index": index,
                }
            )
    return results


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    """Handle entity relation summary
    For each entity or relation, input is the combined description of already existing description and new description.
    If too long, use LLM to summarize.
    """
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    file_path: str = "unknown_source",
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None

    # Clean and validate entity name
    entity_name = clean_str(record_attributes[1]).strip('"')
    if not entity_name.strip():
        logger.warning(
            f"Entity extraction error: empty entity name in: {record_attributes}"
        )
        return None

    # Clean and validate entity type
    entity_type = clean_str(record_attributes[2]).strip('"')
    if not entity_type.strip() or entity_type.startswith('("'):
        logger.warning(
            f"Entity extraction error: invalid entity type in: {record_attributes}"
        )
        return None

    # Clean and validate description
    entity_description = clean_str(record_attributes[3]).strip('"')
    if not entity_description.strip():
        logger.warning(
            f"Entity extraction error: empty description for entity '{entity_name}' of type '{entity_type}'"
        )
        return None

    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=chunk_key,
        file_path=file_path,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
    file_path: str = "unknown_source",
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1]).strip('"')
    target = clean_str(record_attributes[2]).strip('"')
    edge_description = clean_str(record_attributes[3]).strip('"')
    edge_keywords = clean_str(record_attributes[4]).strip('"')
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1].strip('"'))
        if is_float_regex(record_attributes[-1])
        else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id,
        file_path=file_path,
    )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict], # list of node data dicts for the same entity from different chunks
    knowledge_graph_inst: BaseGraphStorage, # Expects TenantAwareNeo4JStorage instance
    global_config: dict,
    folder_id: int, # <--- 新增参数
) -> dict:
    """
    Gets existing node from graph store using entity_name and folder_id,
    merges properties with new data, summarizes description if needed,
    and upserts the node back into the graph store for the specific folder.

    Args:
        entity_name: The name (and usually ID) of the entity.
        nodes_data: A list of new property dictionaries for this entity extracted
                    from different chunks (within the same folder).
        knowledge_graph_inst: The tenant-aware graph storage instance.
        global_config: Global configuration dictionary.
        folder_id: The ID of the folder (tenant) this operation belongs to.

    Returns:
        The final merged and upserted node data dictionary.
    """
    if folder_id is None:
        raise ValueError("_merge_nodes_then_upsert requires a folder_id")
    if not nodes_data:
         logger.warning(f"Received empty nodes_data for entity '{entity_name}' in folder {folder_id}. Skipping merge/upsert.")
         # Decide return value: maybe fetch existing node? or return empty?
         # Fetching existing node seems reasonable if it exists.
         existing_node = await knowledge_graph_inst.get_node(entity_name, folder_id=folder_id)
         return existing_node if existing_node else {"entity_name": entity_name, "error": "No new data and node not found"}


    # --- 获取已存在的节点数据 (特定于 folder_id) ---
    already_entity_types = []
    already_source_ids = []
    already_description = []
    already_file_paths = []

    # 调用 tenant-aware 的 get_node
    already_node = await knowledge_graph_inst.get_node(entity_name, folder_id=folder_id)

    if already_node:
        logger.debug(f"Found existing node '{entity_name}' in folder {folder_id}.")
        # 使用 .get() 安全访问，以防属性缺失
        if already_node.get("entity_type"):
            already_entity_types.append(already_node["entity_type"])
        if already_node.get("source_id"):
            already_source_ids.extend(
                split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
            )
        if already_node.get("file_path"):
            already_file_paths.extend(
                split_string_by_multi_markers(already_node["file_path"], [GRAPH_FIELD_SEP])
            )
        if already_node.get("description"):
            already_description.append(already_node["description"])
    else:
        logger.debug(f"Node '{entity_name}' not found in folder {folder_id}, creating new.")

    # --- 合并新旧属性 ---
    # 实体类型: 选择最常见的类型
    all_types = [dp.get("entity_type", "UNKNOWN") for dp in nodes_data] + already_entity_types
    if all_types:
         entity_type = Counter(all_types).most_common(1)[0][0]
    else:
         entity_type = "UNKNOWN" # 如果完全没有类型信息

    # 描述: 合并并去重
    all_descriptions = [dp.get("description", "") for dp in nodes_data if dp.get("description")] + already_description
    # 使用 set 去重，然后排序（可选）再合并
    unique_descriptions = sorted(list(set(filter(None, all_descriptions)))) # filter(None, ...) 移除空字符串
    merged_description = GRAPH_FIELD_SEP.join(unique_descriptions)


    # Source ID: 合并并去重
    all_source_ids = [dp.get("source_id", "") for dp in nodes_data if dp.get("source_id")] + already_source_ids
    unique_source_ids = sorted(list(set(filter(None, all_source_ids))))
    source_id = GRAPH_FIELD_SEP.join(unique_source_ids)

    # File Path: 合并并去重
    all_file_paths = [dp.get("file_path", "") for dp in nodes_data if dp.get("file_path")] + already_file_paths
    unique_file_paths = sorted(list(set(filter(None, all_file_paths))))
    file_path = GRAPH_FIELD_SEP.join(unique_file_paths)

    # --- 总结描述 (如果过长) ---
    # 调用原始的 _handle_entity_relation_summary
    final_description = await _handle_entity_relation_summary(
        entity_name, merged_description, global_config
    )

    # --- 准备最终节点数据 ---
    # 确保包含 entity_id，因为 Neo4J upsert 需要它
    node_data = dict(
        entity_id=entity_name, # Neo4J 实现中需要这个字段
        entity_type=entity_type,
        description=final_description,
        source_id=source_id,
        file_path=file_path,
        # folder_id 将在 TenantAwareNeo4JStorage.upsert_node 内部添加
    )
    # 可以添加其他可能从 already_node 或 nodes_data 继承的属性
    # 例如，如果原始节点有创建时间戳，可能需要保留


    # --- 调用 tenant-aware 的 upsert_node ---
    await knowledge_graph_inst.upsert_node(
        node_id=entity_name, # node_id 通常就是 entity_name
        node_data=node_data,
        folder_id=folder_id, # <--- 传递 folder_id
    )

    # 返回最终的节点数据 (可能用于后续处理，例如 VDB 更新)
    # 在返回的数据中也包含 entity_name，方便外部使用
    final_node_data_for_return = node_data.copy()
    final_node_data_for_return["entity_name"] = entity_name
    # 可以选择性地移除内部使用的 entity_id (如果外部不需要)
    # if "entity_id" in final_node_data_for_return:
    #     del final_node_data_for_return["entity_id"]

    return final_node_data_for_return


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict], # List of new edge data dicts between src/tgt from different chunks
    knowledge_graph_inst: BaseGraphStorage, # Expects TenantAwareNeo4JStorage
    global_config: dict,
    folder_id: int, # <--- 新增参数
) -> dict:
    """
    Gets existing edge between src_id and tgt_id within folder_id,
    merges properties with new data, summarizes description if needed,
    and upserts the edge back into the graph store for the specific folder.
    Handles potential missing nodes by creating them if necessary.

    Args:
        src_id: The source entity name/ID.
        tgt_id: The target entity name/ID.
        edges_data: A list of new property dictionaries for this edge extracted
                    from different chunks (within the same folder).
        knowledge_graph_inst: The tenant-aware graph storage instance.
        global_config: Global configuration dictionary.
        folder_id: The ID of the folder (tenant) this operation belongs to.

    Returns:
        The final merged and upserted edge data dictionary.
    """
    if folder_id is None:
        raise ValueError("_merge_edges_then_upsert requires a folder_id")
    if not edges_data:
        logger.warning(f"Received empty edges_data for edge {src_id}->{tgt_id} in folder {folder_id}. Skipping merge/upsert.")
        # Fetch existing edge if it exists
        existing_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id, folder_id=folder_id)
        return existing_edge if existing_edge else {"src_id": src_id, "tgt_id": tgt_id, "error": "No new data and edge not found"}


    # --- 获取已存在的边数据 (特定于 folder_id) ---
    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []
    already_file_paths = []

    # 调用 tenant-aware 的 has_edge 和 get_edge
    if await knowledge_graph_inst.has_edge(src_id, tgt_id, folder_id=folder_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id, folder_id=folder_id)
        if already_edge:
            logger.debug(f"Found existing edge {src_id}->{tgt_id} in folder {folder_id}.")
            # 安全访问属性
            if already_edge.get("weight") is not None:
                try:
                    already_weights.append(float(already_edge["weight"]))
                except (ValueError, TypeError):
                    logger.warning(f"Invalid existing weight for edge {src_id}->{tgt_id}: {already_edge['weight']}. Using 0.0.")
                    already_weights.append(0.0)
            if already_edge.get("source_id"):
                already_source_ids.extend(
                    split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
                )
            if already_edge.get("file_path"):
                already_file_paths.extend(
                    split_string_by_multi_markers(already_edge["file_path"], [GRAPH_FIELD_SEP])
                )
            if already_edge.get("description"):
                already_description.append(already_edge["description"])
            if already_edge.get("keywords"):
                already_keywords.extend(
                     split_string_by_multi_markers(already_edge["keywords"], [GRAPH_FIELD_SEP]) # Use extend for keywords list
                )
        else:
             logger.debug(f"Edge {src_id}->{tgt_id} reported by has_edge but get_edge returned None in folder {folder_id}.")
    else:
         logger.debug(f"Edge {src_id}->{tgt_id} not found in folder {folder_id}, creating new.")


    # --- 合并新旧属性 ---
    # Weight: 求和
    new_weights = []
    for dp in edges_data:
        try:
             new_weights.append(float(dp.get("weight", 0.0))) # Default to 0.0 if missing
        except (ValueError, TypeError):
             logger.warning(f"Invalid new weight in edge data for {src_id}->{tgt_id}: {dp.get('weight')}. Using 0.0.")
             new_weights.append(0.0)
    weight = sum(new_weights + already_weights)

    # Description: 合并并去重
    all_descriptions = [dp.get("description", "") for dp in edges_data if dp.get("description")] + already_description
    unique_descriptions = sorted(list(set(filter(None, all_descriptions))))
    merged_description = GRAPH_FIELD_SEP.join(unique_descriptions)

    # Keywords: 合并并去重
    # Keywords can be space-separated or comma-separated within each entry. We split, flatten, unique, sort, join.
    all_keywords_flat = set(already_keywords)
    for dp in edges_data:
        if dp.get("keywords"):
            # Split by common delimiters and filter empty strings
            kws = filter(None, re.split(r'[,\s]+', dp["keywords"]))
            all_keywords_flat.update(kws)
    keywords = GRAPH_FIELD_SEP.join(sorted(list(all_keywords_flat)))


    # Source ID: 合并并去重
    all_source_ids = [dp.get("source_id", "") for dp in edges_data if dp.get("source_id")] + already_source_ids
    unique_source_ids = sorted(list(set(filter(None, all_source_ids))))
    source_id = GRAPH_FIELD_SEP.join(unique_source_ids)

    # File Path: 合并并去重
    all_file_paths = [dp.get("file_path", "") for dp in edges_data if dp.get("file_path")] + already_file_paths
    unique_file_paths = sorted(list(set(filter(None, all_file_paths))))
    file_path = GRAPH_FIELD_SEP.join(unique_file_paths)

    # --- 确保源节点和目标节点存在 (在同一 folder_id 下) ---
    # 检查并可能创建节点是 upsert_edge 的责任，但这里可以先检查
    for node_id_to_check in [src_id, tgt_id]:
        if not await knowledge_graph_inst.has_node(node_id_to_check, folder_id=folder_id):
            logger.warning(f"Node '{node_id_to_check}' for edge {src_id}->{tgt_id} not found in folder {folder_id}. Creating placeholder.")
            # 创建一个最小化的节点，以便边可以创建。更好的做法可能是在 extract_entities 阶段就确保节点存在。
            placeholder_node_data = {
                "entity_id": node_id_to_check,
                "entity_type": "UNKNOWN_PLACEHOLDER",
                "description": "Placeholder node created during edge upsert.",
                "source_id": source_id, # Use edge's source_id
                "file_path": file_path, # Use edge's file_path
            }
            await knowledge_graph_inst.upsert_node(
                 node_id_to_check, placeholder_node_data, folder_id=folder_id
            )


    # --- 总结描述 (如果需要) ---
    final_description = await _handle_entity_relation_summary(
        f"({src_id}, {tgt_id})", merged_description, global_config
    )

    # --- 准备最终边数据 ---
    final_edge_data = dict(
        weight=weight,
        description=final_description,
        keywords=keywords,
        source_id=source_id,
        file_path=file_path,
        # folder_id 将在 TenantAwareNeo4JStorage.upsert_edge 内部添加
    )

    # --- 调用 tenant-aware 的 upsert_edge ---
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=final_edge_data,
        folder_id=folder_id, # <--- 传递 folder_id
    )

    # --- 返回最终的边数据 (用于 VDB 更新等) ---
    # 添加 src_id 和 tgt_id 以方便外部使用
    final_edge_data_for_return = final_edge_data.copy()
    final_edge_data_for_return["src_id"] = src_id
    final_edge_data_for_return["tgt_id"] = tgt_id

    return final_edge_data_for_return


async def extract_entities(
    chunks: dict[str, Any], # 使用 Any 替代 TextChunkSchema 以简化
    knowledge_graph_inst: BaseGraphStorage, # Expects TenantAwareNeo4JStorage
    entity_vdb: BaseVectorStorage,          # Expects TenantAwareMilvusVectorDBStorage
    relationships_vdb: BaseVectorStorage,   # Expects TenantAwareMilvusVectorDBStorage
    global_config: dict[str, Any],
    folder_id: int, # <--- 新增参数
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None, # Expects TenantAwareRedisKVStorage
) -> None:
    """
    Extracts entities and relationships from text chunks for a specific folder,
    updates the graph store and vector databases accordingly.

    Args:
        chunks: Dictionary where keys are chunk IDs and values are chunk data (including content and file_path).
        knowledge_graph_inst: Tenant-aware graph storage instance.
        entity_vdb: Tenant-aware entity vector database instance.
        relationships_vdb: Tenant-aware relationship vector database instance.
        global_config: Global configuration dictionary.
        folder_id: The ID of the folder (tenant) this operation belongs to.
        pipeline_status: Optional dictionary for pipeline status updates.
        pipeline_status_lock: Optional asyncio Lock for pipeline status.
        llm_response_cache: Optional tenant-aware cache storage instance.
    """
    if folder_id is None:
        raise ValueError("extract_entities requires a folder_id")

    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]
    enable_llm_cache_for_entity_extract: bool = global_config.get(
        "enable_llm_cache_for_entity_extract", True # 使用 .get() 安全访问
    )

    ordered_chunks = list(chunks.items())
    # ... (构建 LLM prompt 的逻辑，包括 language, entity_types, examples 不变)
    language = global_config["addon_params"].get("language", PROMPTS["DEFAULT_LANGUAGE"])
    entity_types = global_config["addon_params"].get("entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"])
    example_number = global_config["addon_params"].get("example_number", None)
    # ... (构建 examples 字符串)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        examples = "\n".join(PROMPTS["entity_extraction_examples"][: int(example_number)])
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=", ".join(entity_types),
        language=language,
    )
    examples = examples.format(**example_context_base) # 格式化 examples

    entity_extract_prompt_template = PROMPTS["entity_extraction"] # 获取模板
    context_base = dict( # 用于模板的上下文
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=", ".join(entity_types), # 确保是逗号分隔的字符串
        examples=examples,
        language=language,
    )
    continue_prompt = PROMPTS["entity_continue_extraction"].format(**context_base)
    if_loop_prompt = PROMPTS["entity_if_loop_extraction"] # 这个通常不需要 format

    processed_chunks = 0
    total_chunks = len(ordered_chunks)
    total_entities_count = 0
    total_relations_count = 0

    # 获取图数据库锁
    from .kg.shared_storage import get_graph_db_lock # 假设路径正确
    graph_db_lock = get_graph_db_lock(enable_logging=False)

    # 内部函数：使用带缓存的 LLM 调用 (需要传递 folder_id 给缓存操作)
    async def _user_llm_func_with_cache(
        input_text: str, history_messages: list[dict[str, str]] = None
    ) -> str:
        # 仅当启用了提取缓存并且提供了缓存实例时才进行缓存操作
        if enable_llm_cache_for_entity_extract and llm_response_cache:
            _prompt = json.dumps(history_messages, ensure_ascii=False) + "\n" + input_text if history_messages else input_text
            # 使用 'default' 模式进行提取缓存，并传递 folder_id
            arg_hash = compute_args_hash(_prompt) # Hash 保持不变

            # 调用修改后的 handle_cache，传入 folder_id
            cached_return, _, _, _ = await handle_cache(
                llm_response_cache,
                arg_hash,
                _prompt,
                mode="default", # 提取缓存使用 'default' 模式
                cache_type="extract", # 指定缓存类型
                folder_id=folder_id, # <--- 传递 folder_id
            )
            if cached_return:
                logger.debug(f"Found extract cache for hash {arg_hash} in folder {folder_id}")
                statistic_data["llm_cache"] += 1
                return cached_return

        # 如果缓存未命中或未启用缓存，则调用 LLM
        statistic_data["llm_call"] += 1
        if history_messages:
            res: str = await use_llm_func(input_text, history_messages=history_messages)
        else:
            res: str = await use_llm_func(input_text)

        # 如果启用了缓存，保存结果 (传递 folder_id)
        if enable_llm_cache_for_entity_extract and llm_response_cache:
            # 调用修改后的 save_to_cache，传入 folder_id
            await save_to_cache(
                llm_response_cache,
                CacheData(
                    args_hash=arg_hash,
                    content=res,
                    prompt=_prompt, # 缓存时使用组合后的 prompt
                    # embedding 相关字段为 None，因为这是基于 hash 的缓存
                    quantized=None, min_val=None, max_val=None,
                    mode="default", # 保存到 'default' 模式
                    cache_type="extract", # 指定缓存类型
                ),
                folder_id=folder_id, # <--- 传递 folder_id
            )
        return res

    # 内部函数：处理 LLM 的提取结果 (这个函数本身不需要 folder_id)
    async def _process_extraction_result(
        result: str, chunk_key: str, file_path: str = "unknown_source"
    ):
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        records = split_string_by_multi_markers(
            result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )
        for record in records:
            # ... (解析实体的逻辑不变)
            record_match = re.search(r"\((.*)\)", record)
            if record_match is None: continue
            record_content = record_match.group(1)
            record_attributes = split_string_by_multi_markers(record_content, [context_base["tuple_delimiter"]])

            if_entities = await _handle_single_entity_extraction(record_attributes, chunk_key, file_path)
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(record_attributes, chunk_key, file_path)
            if if_relation is not None:
                 # 确保边的方向一致性，例如总是按字母排序存储
                 # sorted_edge_key = tuple(sorted((if_relation["src_id"], if_relation["tgt_id"])))
                 # maybe_edges[sorted_edge_key].append(if_relation)
                 # 或者保持原始方向，由 _merge_edges_then_upsert 处理
                 edge_key = (if_relation["src_id"], if_relation["tgt_id"])
                 maybe_edges[edge_key].append(if_relation)

        return maybe_nodes, maybe_edges

    # 内部函数：处理单个 chunk (这是核心的并行任务单元)
    async def _process_single_content(chunk_key_dp: tuple[str, Any]): # 使用 Any
        nonlocal processed_chunks, total_entities_count, total_relations_count # 允许修改外部计数器
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        file_path = chunk_dp.get("file_path", "unknown_source") # 获取文件路径

        # --- LLM 调用提取 ---
        # 格式化初始提取的 prompt
        hint_prompt = entity_extract_prompt_template.format(**context_base, input_text=content)
        final_result = await _user_llm_func_with_cache(hint_prompt) # 调用带缓存的 LLM 函数
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result) # 构建历史

        # 处理初始结果
        maybe_nodes, maybe_edges = await _process_extraction_result(final_result, chunk_key, file_path)

        # --- 多轮提取 (Gleaning) ---
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await _user_llm_func_with_cache(continue_prompt, history_messages=history)
            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            glean_nodes, glean_edges = await _process_extraction_result(glean_result, chunk_key, file_path)
            # 合并结果
            for entity_name, entities in glean_nodes.items(): maybe_nodes[entity_name].extend(entities)
            for edge_key, edges in glean_edges.items(): maybe_edges[edge_key].extend(edges)

            if now_glean_index == entity_extract_max_gleaning - 1: break # 到达最大轮次

            # 检查是否需要继续
            if_loop_result: str = await _user_llm_func_with_cache(if_loop_prompt, history_messages=history)
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes": break # LLM 说不需要继续了

        # --- 更新处理进度和日志 ---
        processed_chunks += 1
        entities_count = len(maybe_nodes)
        relations_count = len(maybe_edges)
        log_message = f"Folder {folder_id} Chk {processed_chunks}/{total_chunks}: extracted {entities_count} Ent + {relations_count} Rel"
        logger.info(log_message)
        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

        # --- 将提取结果写入存储 (使用锁和传递 folder_id) ---
        chunk_entities_data = []
        chunk_relationships_data = []

        async with graph_db_lock: # 保证图数据库操作的原子性
            # 处理和更新实体 (调用修改后的 _merge_nodes_then_upsert)
            for entity_name, entities in maybe_nodes.items():
                entity_data = await _merge_nodes_then_upsert(
                    entity_name, entities, knowledge_graph_inst, global_config,
                    folder_id=folder_id # <--- 传递 folder_id
                )
                if entity_data and "error" not in entity_data: # 确保成功合并/创建
                     chunk_entities_data.append(entity_data)

            # 处理和更新关系 (调用修改后的 _merge_edges_then_upsert)
            for edge_key, edges in maybe_edges.items():
                # _merge_edges_then_upsert 内部应该处理好方向性或去重逻辑
                edge_data = await _merge_edges_then_upsert(
                    edge_key[0], edge_key[1], edges, knowledge_graph_inst, global_config,
                    folder_id=folder_id # <--- 传递 folder_id
                )
                if edge_data and "error" not in edge_data:
                     chunk_relationships_data.append(edge_data)

            # --- 更新向量数据库 (在锁内部，确保与图数据库一致) ---
            # 更新实体 VDB (调用修改后的 VDB upsert)
            if entity_vdb is not None and chunk_entities_data:
                data_for_vdb = {
                    compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                        "entity_name": dp["entity_name"],
                        "entity_type": dp.get("entity_type", "UNKNOWN"),
                        "content": f"{dp['entity_name']}\n{dp.get('description', '')}",
                        "source_id": dp.get("source_id", ""),
                        "file_path": dp.get("file_path", "unknown_source"),
                        # 保留其他元数据 (如果存在于 dp 中且 meta_fields 允许)
                         **{k_meta: dp[k_meta] for k_meta in entity_vdb.meta_fields if k_meta in dp and k_meta not in ['entity_name', 'entity_type', 'content', 'source_id', 'file_path']}
                    }
                    for dp in chunk_entities_data # 使用成功处理的实体数据
                }
                if data_for_vdb: # 确保有数据再调用 upsert
                     await entity_vdb.upsert(data_for_vdb, folder_id=folder_id) # <--- 传递 folder_id

            # 更新关系 VDB (调用修改后的 VDB upsert)
            if relationships_vdb is not None and chunk_relationships_data:
                data_for_vdb = {
                    compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                        "src_id": dp["src_id"],
                        "tgt_id": dp["tgt_id"],
                        "keywords": dp.get("keywords", ""),
                        "content": f'{dp.get("src_id", "")}\t{dp.get("tgt_id", "")}\n{dp.get("keywords", "")}\n{dp.get("description", "")}',
                        "source_id": dp.get("source_id", ""),
                        "description": dp.get("description", ""),
                        "weight": dp.get("weight", 1.0), # 确保 weight 存在
                        "file_path": dp.get("file_path", "unknown_source"),
                         # 保留其他元数据
                         **{k_meta: dp[k_meta] for k_meta in relationships_vdb.meta_fields if k_meta in dp and k_meta not in ['src_id', 'tgt_id', 'keywords', 'content', 'source_id', 'description', 'weight', 'file_path']}
                    }
                    for dp in chunk_relationships_data # 使用成功处理的关系数据
                }
                if data_for_vdb:
                    await relationships_vdb.upsert(data_for_vdb, folder_id=folder_id) # <--- 传递 folder_id

            # 在锁内部更新全局计数器 (如果这些计数器需要精确)
            total_entities_count += len(chunk_entities_data)
            total_relations_count += len(chunk_relationships_data)
        # --- 锁结束 ---

    # --- 并行处理所有 chunks ---
    tasks = [_process_single_content(c) for c in ordered_chunks]
    await asyncio.gather(*tasks)

    # --- 最终日志 ---
    log_message = f"Folder {folder_id} Extraction Complete: {total_entities_count} entities + {total_relations_count} relationships (total)"
    logger.info(log_message)
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = log_message
            pipeline_status["history_messages"].append(log_message)


async def kg_query(
    query: str,
    knowledge_graph_inst: BaseGraphStorage, # TenantAwareNeo4JStorage
    entities_vdb: BaseVectorStorage,       # TenantAwareMilvusVectorDBStorage
    relationships_vdb: BaseVectorStorage, # TenantAwareMilvusVectorDBStorage
    text_chunks_db: BaseKVStorage,         # TenantAwareRedisKVStorage
    query_param: QueryParam,
    global_config: dict[str, Any], # 使用 Any 允许更多类型
    folder_id: int, # <--- 新增参数
    hashing_kv: BaseKVStorage | None = None, # TenantAwareRedisKVStorage
    system_prompt: str | None = None,
) -> Union[str, AsyncIterator[str]]:
    """
    Performs a knowledge graph based query within a specific folder.
    Assumes underlying storage instances are tenant-aware.
    """
    if folder_id is None:
        raise ValueError("kg_query requires folder_id")

    # --- 1. 缓存处理 (传递 folder_id) ---
    use_model_func = (
        query_param.model_func
        if query_param.model_func
        else global_config["llm_model_func"]
    )
    # 缓存键包含模式和查询文本
    args_hash = compute_args_hash(query_param.mode, query, folder_id, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv,
        args_hash,
        query,
        query_param.mode,
        cache_type="query",
        folder_id=folder_id # <--- 传递 folder_id 给 handle_cache
    )
    if cached_response is not None:
        logger.info(f"Cache hit for kg_query in folder {folder_id}")
        return cached_response
    logger.info(f"Cache miss for kg_query in folder {folder_id}")


    # --- 2. 获取关键词 (传递 folder_id 给其内部的缓存调用) ---
    # 假设 get_keywords_from_query 内部调用 extract_keywords_only,
    # 且 extract_keywords_only 内部调用 handle_cache 时传递了 folder_id
    hl_keywords, ll_keywords = await get_keywords_from_query(
        query, query_param, global_config, hashing_kv, folder_id=folder_id # <--- 传递 folder_id
    )

    logger.debug(f"Folder {folder_id} - High-level keywords: {hl_keywords}")
    logger.debug(f"Folder {folder_id} - Low-level keywords: {ll_keywords}")

    # 处理空关键词 (逻辑不变)
    original_mode = query_param.mode
    mode_changed = False
    if not hl_keywords and not ll_keywords:
        logger.warning(f"Folder {folder_id}: Both high-level and low-level keywords are empty.")
        return PROMPTS["fail_response"]
    if not ll_keywords and query_param.mode in ["local", "hybrid"]:
        logger.warning(
            f"Folder {folder_id}: low_level_keywords empty, switching from {query_param.mode} to global"
        )
        query_param.mode = "global"
        mode_changed = True
    if not hl_keywords and query_param.mode in ["global", "hybrid"]:
        logger.warning(
            f"Folder {folder_id}: high_level_keywords empty, switching from {query_param.mode} to local"
        )
        query_param.mode = "local"
        mode_changed = True

    # 如果模式因关键词缺失而改变，确保至少有一组关键词存在
    if mode_changed and not hl_keywords and not ll_keywords:
         logger.error(f"Folder {folder_id}: Mode changed but still no keywords available. Cannot proceed.")
         query_param.mode = original_mode # 恢复原始模式以防混淆
         return PROMPTS["fail_response"]


    ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
    hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""

    # --- 3. 构建上下文 (传递 folder_id) ---
    # 假设 _build_query_context 已被修改以接受 folder_id
    context = await _build_query_context(
        ll_keywords_str,
        hl_keywords_str,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        folder_id=folder_id # <--- 传递 folder_id
    )

    # 恢复可能被修改的模式，以便后续缓存保存时使用正确的模式 key
    if mode_changed:
        logger.debug(f"Restoring query mode from {query_param.mode} to {original_mode} for caching.")
        query_param.mode = original_mode


    if query_param.only_need_context:
        return context if context else "No context generated." # 返回空字符串或提示信息
    if context is None:
        logger.warning(f"Failed to build query context for folder {folder_id}")
        return PROMPTS["fail_response"]

    # --- 4. 处理对话历史 (逻辑不变) ---
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    # --- 5. 构建最终提示 (逻辑不变) ---
    sys_prompt_temp = system_prompt if system_prompt else PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context,
        response_type=query_param.response_type,
        history=history_context,
    )

    if query_param.only_need_prompt:
        return sys_prompt

    len_of_prompts = len(encode_string_by_tiktoken(query + sys_prompt))
    logger.debug(f"[kg_query - Folder {folder_id}] Prompt Tokens: {len_of_prompts}")

    # --- 6. 调用 LLM (逻辑不变) ---
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
        # 可以在这里传递 folder_id 给 LLM 函数吗？取决于 use_model_func 的实现
        # 如果 LLM 本身也需要知道租户信息，可以在 kwargs 中传递
        # folder_id=folder_id
    )

    # --- 7. 处理和保存缓存 (传递 folder_id) ---
    # 对于非流式响应
    if isinstance(response, str):
        # 清理响应内容 (逻辑不变)
        if len(response) > len(sys_prompt):
             response = (
                  response.replace(sys_prompt, "")
                  .replace("user", "").replace("model", "")
                  .replace(query, "").replace("<system>", "").replace("</system>", "")
                  .strip()
             )
        # 保存到缓存 (传递 folder_id)
        await save_to_cache(
            hashing_kv,
            CacheData(
                args_hash=args_hash,
                content=response,
                prompt=query,
                quantized=quantized, # 从 handle_cache 获取
                min_val=min_val,     # 从 handle_cache 获取
                max_val=max_val,     # 从 handle_cache 获取
                mode=query_param.mode, # 使用原始（或最终确定）的模式
                cache_type="query",
            ),
            folder_id=folder_id # <--- 传递 folder_id 给 save_to_cache
        )
        return response
    # 对于流式响应 (AsyncIterator)
    elif hasattr(response, "__aiter__"):
         # 流式响应通常不直接缓存，或者需要特殊处理来收集完整响应后再缓存
         logger.debug(f"Returning stream for kg_query in folder {folder_id}. Caching skipped for stream.")
         return response
    else:
         logger.error(f"Unexpected response type from LLM for folder {folder_id}: {type(response)}")
         return PROMPTS["fail_response"]


async def get_keywords_from_query(
    query: str,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    folder_id: int | None = None, # <--- 新增参数
) -> tuple[list[str], list[str]]:
    """
    Retrieves high-level and low-level keywords for RAG operations, tenant-aware.

    Checks query_param first, otherwise calls tenant-aware keyword extraction.

    Args:
        query: The user's query text.
        query_param: Query parameters that may contain pre-defined keywords.
        global_config: Global configuration dictionary.
        hashing_kv: Optional key-value storage for caching results.
        folder_id: The tenant identifier. Required if keywords need extraction.

    Returns:
        A tuple containing (high_level_keywords, low_level_keywords).
    """
    # Check if pre-defined keywords are already provided (no folder_id needed here)
    if query_param.hl_keywords or query_param.ll_keywords:
        logger.debug("Using pre-defined keywords from QueryParam.")
        return query_param.hl_keywords, query_param.ll_keywords

    # If keywords need extraction, folder_id is required for caching
    if folder_id is None:
        raise ValueError("folder_id is required when extracting keywords (for cache isolation).")

    # Extract keywords using tenant-aware extract_keywords_only
    logger.debug(f"Extracting keywords for query in folder {folder_id}.")
    hl_keywords, ll_keywords = await extract_keywords_only(
        query,
        query_param,
        global_config,
        hashing_kv,
        folder_id=folder_id # <--- 传递 folder_id
    )
    return hl_keywords, ll_keywords


async def extract_keywords_only(
    text: str,
    param: QueryParam,
    global_config: dict[str, Any], # Use Any for flexibility
    hashing_kv: BaseKVStorage | None = None,
    folder_id: int | None = None, # <--- 新增参数
) -> tuple[list[str], list[str]]:
    """
    Extract high-level and low-level keywords from the given 'text' using the LLM,
    considering tenant isolation for caching.
    This method ONLY extracts keywords (hl_keywords, ll_keywords).
    """
    # 对于关键词提取，folder_id 是必需的，因为缓存是隔离的
    if hashing_kv is not None and folder_id is None:
         logger.warning("folder_id is required for tenant-aware keyword caching. Cache will be skipped.")
         # 或者可以直接报错: raise ValueError("folder_id is required for cached keyword extraction")

    # 1. Handle cache if needed - pass folder_id
    args_hash = compute_args_hash(param.mode, text, folder_id, cache_type="keywords")
    cached_response = None
    quantized = min_val = max_val = None # Initialize these for potential saving later

    if hashing_kv is not None and folder_id is not None: # 只有提供了 kv 和 folder_id 才尝试缓存
        cached_response, quantized, min_val, max_val = await handle_cache(
            hashing_kv,
            args_hash,
            text,
            param.mode,
            cache_type="keywords",
            folder_id=folder_id, # <--- 传递 folder_id
        )

    if cached_response is not None:
        try:
            keywords_data = json.loads(cached_response)
            # Ensure lists are returned even if keys are missing in cache
            hl_keywords = keywords_data.get("high_level_keywords", [])
            ll_keywords = keywords_data.get("low_level_keywords", [])
            logger.debug(f"Keyword cache hit for folder {folder_id}")
            return hl_keywords, ll_keywords
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(
                f"Invalid cache format for keywords (folder {folder_id}, hash {args_hash}), proceeding with extraction: {e}"
            )
            # Fall through to extraction if cache is invalid

    logger.debug(f"Keyword cache miss for folder {folder_id}")

    # 2. Build the examples (logic remains the same)
    example_number = global_config.get("addon_params", {}).get("example_number", None)
    if example_number and example_number < len(PROMPTS["keywords_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["keywords_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["keywords_extraction_examples"])
    language = global_config.get("addon_params", {}).get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    # 3. Process conversation history (logic remains the same)
    history_context = ""
    if param.conversation_history:
        history_context = get_conversation_turns(
            param.conversation_history, param.history_turns
        )

    # 4. Build the keyword-extraction prompt (logic remains the same)
    kw_prompt = PROMPTS["keywords_extraction"].format(
        query=text, examples=examples, language=language, history=history_context
    )

    # 5. Call the LLM for keyword extraction
    # Determine which LLM function to use (from param or global config)
    llm_func_to_use: Callable = param.model_func if param.model_func else global_config.get("llm_model_func")
    if not llm_func_to_use:
        logger.error("LLM model function not configured.")
        return [], [] # Return empty lists if LLM func is missing

    try:
        # Ensure keyword_extraction=True forces JSON format if required by LLM func implementation
        # The actual forcing of JSON might happen within the llm_func (like openai_complete_if_cache)
        result_str = await llm_func_to_use(kw_prompt, keyword_extraction=True, hashing_kv=hashing_kv) # Pass hashing_kv if llm_func expects it for internal caching
    except Exception as e:
        logger.error(f"LLM call for keyword extraction failed: {e}")
        return [], []

    # 6. Parse out JSON from the LLM response (logic remains the same)
    match = re.search(r"\{.*\}", result_str, re.DOTALL | re.MULTILINE) # Added MULTILINE flag
    hl_keywords = []
    ll_keywords = []
    if not match:
        logger.error(f"No JSON structure found in LLM response for keyword extraction: {result_str[:500]}...")
    else:
        json_str = match.group(0)
        try:
            # Attempt to clean common issues before parsing
            cleaned_json_str = json_str.replace('\\n', '').replace('\n', '').replace("'", '"')
            # Handle potential trailing commas before closing brace/bracket
            cleaned_json_str = re.sub(r",\s*(\}|\])", r"\1", cleaned_json_str)
            keywords_data = json.loads(cleaned_json_str)
            hl_keywords = keywords_data.get("high_level_keywords", [])
            ll_keywords = keywords_data.get("low_level_keywords", [])
            # Basic validation: ensure they are lists
            if not isinstance(hl_keywords, list): hl_keywords = []
            if not isinstance(ll_keywords, list): ll_keywords = []

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error during keyword extraction: {e}. JSON string: {json_str[:500]}...")
            # Keep keywords as empty lists on parsing error

    # 7. Cache the extracted keywords - pass folder_id
    if (hl_keywords or ll_keywords) and hashing_kv is not None and folder_id is not None:
        cache_content = {
            "high_level_keywords": hl_keywords,
            "low_level_keywords": ll_keywords,
        }
        try:
            await save_to_cache(
                hashing_kv,
                CacheData(
                    args_hash=args_hash,
                    content=json.dumps(cache_content, ensure_ascii=False), # Store as JSON string
                    prompt=text, # Use original text as prompt context for cache
                    quantized=quantized, # Pass quantization data if available from handle_cache
                    min_val=min_val,
                    max_val=max_val,
                    mode=param.mode,
                    cache_type="keywords", # Explicitly set cache type
                ),
                folder_id=folder_id, # <--- 传递 folder_id
            )
            logger.debug(f"Saved keywords to cache for folder {folder_id}")
        except Exception as e:
            logger.error(f"Failed to save keywords to cache for folder {folder_id}: {e}")


    return hl_keywords, ll_keywords


async def mix_kg_vector_query(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,   # TenantAwareNeo4JStorage
    entities_vdb: BaseVectorStorage,          # TenantAwareMilvusVectorDBStorage
    relationships_vdb: BaseVectorStorage,     # TenantAwareMilvusVectorDBStorage
    chunks_vdb: BaseVectorStorage,            # TenantAwareMilvusVectorDBStorage
    text_chunks_db: BaseKVStorage,            # TenantAwareRedisKVStorage
    query_param: QueryParam,
    global_config: dict[str, Any], # 改为 Any 避免严格类型问题
    folder_id: int, # <--- 新增参数
    hashing_kv: BaseKVStorage | None = None,  # TenantAwareRedisKVStorage
    system_prompt: str | None = None,
) -> Union[str, AsyncIterator[str]]: # 使用 Union
    """
    Tenant-aware hybrid retrieval implementation combining KG and vector search.
    """
    # 1. 参数检查
    if folder_id is None:
        raise ValueError("folder_id must be provided for tenant-aware mix_kg_vector_query")

    # 2. 缓存处理 (传递 folder_id 和正确的 mode)
    use_model_func = (
        query_param.model_func
        if query_param.model_func
        else global_config["llm_model_func"]
    )
    args_hash = compute_args_hash("mix", query, folder_id, cache_type="query") # 在哈希中包含 folder_id
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, "mix", cache_type="query", folder_id=folder_id # 传递 folder_id
    )
    if cached_response is not None:
        return cached_response # 返回缓存结果

    # 3. 处理对话历史 (不变)
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    # 4. 并行执行 KG 上下文获取和向量上下文获取 (传递 folder_id)
    async def get_kg_context(target_folder_id: int): # 内部函数接收 folder_id
        try:
            # get_keywords_from_query 可能内部调用缓存，也需要 folder_id
            hl_keywords, ll_keywords = await get_keywords_from_query(
                query, query_param, global_config, hashing_kv, folder_id=target_folder_id # 传递 folder_id
            )

            if not hl_keywords and not ll_keywords:
                logger.warning(f"Folder {target_folder_id}: Both keyword types empty for KG context.")
                return None

            ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
            hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""

            # 创建临时的 QueryParam副本，避免修改原始对象影响其他调用
            kg_query_param = QueryParam(**query_param.__dict__) # 复制参数

            # 根据关键词调整模式 (这部分逻辑可以保留，但查询本身是 tenant-aware 的)
            if not ll_keywords_str and not hl_keywords_str: return None
            elif not ll_keywords_str: kg_query_param.mode = "global"
            elif not hl_keywords_str: kg_query_param.mode = "local"
            else: kg_query_param.mode = "hybrid" # 混合模式内部仍然需要 local/global

            # 调用修改后的 _build_query_context
            context = await _build_query_context(
                ll_keywords_str,
                hl_keywords_str,
                knowledge_graph_inst,
                entities_vdb,
                relationships_vdb,
                text_chunks_db,
                kg_query_param, # 使用副本
                folder_id=target_folder_id, # <--- 传递 folder_id
            )
            return context

        except Exception as e:
            logger.error(f"Error in get_kg_context for folder {target_folder_id}: {e}")
            traceback.print_exc()
            return None

    async def get_vector_context(target_folder_id: int): # 内部函数接收 folder_id
        augmented_query = query
        # if history_context: # 对话历史增强查询文本的逻辑保持不变
        #     augmented_query = f"{history_context}\n{query}"
        try:
            mix_topk = min(10, query_param.top_k) # 向量部分 top_k 可以小一些

            # 调用 tenant-aware 的向量查询
            results = await chunks_vdb.query(
                augmented_query, top_k=mix_topk, folder_id=target_folder_id, ids=query_param.ids # <--- 传递 folder_id
            )
            if not results:
                logger.debug(f"Folder {target_folder_id}: No vector results found.")
                return None

            chunks_ids = [r["id"] for r in results]
            # 调用 tenant-aware 的 KV 存储查询
            chunks = await text_chunks_db.get_by_ids(chunks_ids, folder_id=target_folder_id) # <--- 传递 folder_id

            valid_chunks_data = []
            chunk_id_to_result_map = {r["id"]: r for r in results} # 用于快速查找 result 元数据

            for chunk_data in chunks:
                 # 确保 chunk_data 不为 None 且包含内容和 ID
                 if chunk_data and "content" in chunk_data and "_id" in chunk_data:
                      chunk_id = chunk_data["_id"]
                      original_result = chunk_id_to_result_map.get(chunk_id)
                      # 合并块内容和来自原始向量搜索结果的时间戳等元数据
                      merged_chunk = {
                           "content": chunk_data["content"],
                           "file_path": chunk_data.get("file_path", "unknown_source"), # 从 text_chunks 获取文件路径
                           "created_at": original_result.get("created_at") if original_result else None, # 从 VDB 结果获取时间戳
                           # 可以添加 score 等其他信息
                           "score": original_result.get("distance") if original_result else None,
                      }
                      valid_chunks_data.append(merged_chunk)


            if not valid_chunks_data:
                 logger.debug(f"Folder {target_folder_id}: No valid chunks found after DB retrieval.")
                 return None

            # 按分数排序（如果可用且需要）
            # valid_chunks_data.sort(key=lambda x: x.get("score", 0), reverse=True) # Milvus distance 越小越好

            # 截断块
            maybe_trun_chunks = truncate_list_by_token_size(
                valid_chunks_data,
                key=lambda x: x["content"],
                max_token_size=query_param.max_token_for_text_unit,
            )

            if not maybe_trun_chunks:
                 logger.debug(f"Folder {target_folder_id}: No chunks left after truncation.")
                 return None

            # 格式化输出，包含文件路径和时间戳
            formatted_chunks = []
            for c in maybe_trun_chunks:
                # 优先使用 text_chunks_db 中的 file_path
                file_info = f"File path: {c.get('file_path', 'unknown_source')}"
                time_info = ""
                if c.get("created_at"):
                    try:
                        # 假设 created_at 是 Unix 时间戳
                        time_info = f"[Created at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(float(c['created_at'])))}]"
                    except (ValueError, TypeError):
                         time_info = f"[Created at: {c.get('created_at')}]" # 原样输出无法解析的时间戳
                score_info = f"[Score: {c.get('score'):.4f}]" if c.get('score') is not None else ""

                chunk_text = f"{file_info}\n{time_info}{score_info}\n{c['content']}"
                formatted_chunks.append(chunk_text)


            logger.debug(f"Folder {target_folder_id}: Truncated vector chunks to {len(formatted_chunks)}")
            return "\n--New Chunk--\n".join(formatted_chunks)
        except Exception as e:
            logger.error(f"Error in get_vector_context for folder {target_folder_id}: {e}")
            traceback.print_exc()
            return None

    # 并行执行，传入 folder_id
    kg_context_result, vector_context_result = await asyncio.gather(
        get_kg_context(folder_id), get_vector_context(folder_id)
    )

    # 5. 合并上下文 (不变)
    if kg_context_result is None and vector_context_result is None:
         logger.warning(f"Folder {folder_id}: Both KG and vector contexts are empty for query '{query}'")
         return PROMPTS["fail_response"]

    # 如果只需要上下文，返回字典
    if query_param.only_need_context:
        return {"kg_context": kg_context_result, "vector_context": vector_context_result}

    # 6. 构建混合提示 (不变)
    sys_prompt_template = system_prompt if system_prompt else PROMPTS["mix_rag_response"]
    # 确保模板格式化时能处理 None 值
    final_sys_prompt = sys_prompt_template.format(
        kg_context=kg_context_result if kg_context_result else "No relevant knowledge graph information found.",
        vector_context=vector_context_result if vector_context_result else "No relevant text information found.",
        response_type=query_param.response_type,
        history=history_context,
    )


    if query_param.only_need_prompt:
        return final_sys_prompt # 返回最终构建的提示

    len_of_prompts = len(encode_string_by_tiktoken(query + final_sys_prompt))
    logger.debug(f"[mix_kg_vector_query folder {folder_id}] Prompt Tokens: {len_of_prompts}")

    # 7. 生成响应 (不变)
    response = await use_model_func(
        query,
        system_prompt=final_sys_prompt,
        stream=query_param.stream,
    )

    # 8. 清理响应内容 (不变，但需要确保流式响应处理正确)
    if isinstance(response, str): # 仅对非流式响应进行清理和缓存
         if len(response) > len(final_sys_prompt):
              # 尝试更安全地移除提示，避免意外删除部分响应
              # 这假设响应以提示开头，如果 LLM 行为不同，可能需要调整
              if response.startswith(final_sys_prompt):
                   response = response[len(final_sys_prompt):]
              # 移除可能的额外标记
              response = response.replace("user", "").replace("model", "").replace(query, "")
              response = response.replace("<system>", "").replace("</system>", "").strip()


         # 9. 保存到缓存 (传递 folder_id)
         await save_to_cache(
              hashing_kv,
              CacheData(
                   args_hash=args_hash,
                   content=response,
                   prompt=query, # 使用原始查询作为提示缓存键
                   quantized=quantized,
                   min_val=min_val,
                   max_val=max_val,
                   mode="mix",
                   cache_type="query",
              ),
              folder_id=folder_id # <--- 传递 folder_id
         )
    # 对于流式响应，直接返回迭代器
    # 注意：流式响应目前无法直接缓存整个结果

    return response


async def _build_query_context(
    ll_keywords: str,
    hl_keywords: str,
    knowledge_graph_inst: BaseGraphStorage, # Should be TenantAwareNeo4JStorage instance
    entities_vdb: BaseVectorStorage,       # Should be TenantAwareMilvusVectorDBStorage instance
    relationships_vdb: BaseVectorStorage,  # Should be TenantAwareMilvusVectorDBStorage instance
    text_chunks_db: BaseKVStorage,         # Should be TenantAwareRedisKVStorage instance
    query_param: QueryParam,
    folder_id: int, # <--- 新增参数
):
    """Builds the context string for the query based on mode, filtered by folder_id."""
    if folder_id is None:
        raise ValueError("_build_query_context requires folder_id")

    logger.info(f"Process {os.getpid()} building query context for folder {folder_id}...")

    entities_context = ""
    relations_context = ""
    text_units_context = ""

    # Call the appropriate tenant-aware data fetching function based on mode
    if query_param.mode == "local":
        entities_context, relations_context, text_units_context = await _get_node_data(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
            folder_id=folder_id, # <--- Pass folder_id
        )
    elif query_param.mode == "global":
        entities_context, relations_context, text_units_context = await _get_edge_data(
            hl_keywords,
            knowledge_graph_inst,
            relationships_vdb,
            text_chunks_db,
            query_param,
            folder_id=folder_id, # <--- Pass folder_id
        )
    elif query_param.mode == "hybrid": # hybrid mode requires special handling
        # Run both local and global fetches concurrently, passing folder_id to both
        ll_data_task = _get_node_data(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
            folder_id=folder_id, # <--- Pass folder_id
        )
        hl_data_task = _get_edge_data(
            hl_keywords,
            knowledge_graph_inst,
            relationships_vdb,
            text_chunks_db,
            query_param,
            folder_id=folder_id, # <--- Pass folder_id
        )
        ll_data, hl_data = await asyncio.gather(ll_data_task, hl_data_task)

        # Unpack results
        (
            ll_entities_context,
            ll_relations_context,
            ll_text_units_context,
        ) = ll_data
        (
            hl_entities_context,
            hl_relations_context,
            hl_text_units_context,
        ) = hl_data

        # Combine contexts (assuming combine_contexts logic remains the same)
        # You might need to import process_combine_contexts if it's not already available
        from .utils import process_combine_contexts
        entities_context = process_combine_contexts(hl_entities_context, ll_entities_context)
        relations_context = process_combine_contexts(hl_relations_context, ll_relations_context)
        text_units_context = process_combine_contexts(hl_text_units_context, ll_text_units_context)

    else:
         # Should not happen if called from kg_query, but handle defensively
         logger.error(f"Invalid mode '{query_param.mode}' passed to _build_query_context")
         return None

    # Check if any context was actually retrieved
    # Use .strip() to handle cases where functions return empty CSV headers
    if not entities_context.strip() and not relations_context.strip() and not text_units_context.strip():
        logger.warning(f"No context generated for folder {folder_id} with mode {query_param.mode}")
        return None # Return None if no meaningful context was built

    result = f"""
    -----Entities-----
    ```csv
    {entities_context}
    ```
    -----Relationships-----
    ```csv
    {relations_context}
    ```
    -----Sources-----
    ```csv
    {text_units_context}
    ```
    """.strip()
    return result


async def _get_node_data(
    query: str, # low-level keywords string
    knowledge_graph_inst: BaseGraphStorage, # TenantAwareNeo4JStorage
    entities_vdb: BaseVectorStorage,       # TenantAwareMilvusVectorDBStorage
    text_chunks_db: BaseKVStorage,         # TenantAwareRedisKVStorage
    query_param: QueryParam,
    folder_id: int, # <--- 新增参数
) -> Tuple[str, str, str]: # 返回 (entities_csv, relations_csv, chunks_csv)
    """
    Retrieves node-centric context for a query within a specific folder.
    Uses low-level keywords to find relevant entities via vector search,
    then gathers their properties, related edges, and text chunks from the graph store.

    Args:
        query: The low-level keywords string for entity vector search.
        knowledge_graph_inst: Tenant-aware graph storage instance.
        entities_vdb: Tenant-aware entity vector storage instance.
        text_chunks_db: Tenant-aware text chunk key-value storage instance.
        query_param: Query parameters.
        folder_id: The ID of the current folder/tenant.

    Returns:
        A tuple containing CSV strings for entities, relationships, and text units context.
        Returns ("", "", "") if no relevant entities are found.
    """
    if folder_id is None:
        raise ValueError("_get_node_data requires a folder_id")

    # 1. 查询相似实体 (tenant-aware)
    logger.info(
        f"Folder {folder_id} - Querying nodes: top_k={query_param.top_k}, query='{query[:50]}...'"
    )
    # 调用 tenant-aware query
    results = await entities_vdb.query(
        query, top_k=query_param.top_k, folder_id=folder_id, ids=query_param.ids
    )

    if not results:
        logger.info(f"Folder {folder_id}: No similar entities found for query.")
        return "", "", ""

    # 2. 获取实体属性和度数 (tenant-aware)
    entity_names_found = [r.get("entity_name") for r in results if r.get("entity_name")]
    if not entity_names_found:
        logger.warning(f"Folder {folder_id}: VDB results missing 'entity_name'. Results: {results}")
        return "", "", ""

    node_tasks = [knowledge_graph_inst.get_node(name, folder_id=folder_id) for name in entity_names_found]
    degree_tasks = [knowledge_graph_inst.node_degree(name, folder_id=folder_id) for name in entity_names_found]

    node_properties_list, node_degrees_list = await asyncio.gather(
         asyncio.gather(*node_tasks),
         asyncio.gather(*degree_tasks),
    )

    # 合并 VDB 结果、节点属性和度数，过滤掉未找到的节点
    node_datas_merged = []
    for vdb_result, props, degree in zip(results, node_properties_list, node_degrees_list):
         if props: # 仅当在图数据库中找到节点时才包括
              # 从 VDB 结果获取创建时间 (如果 Milvus 查询返回了)
              # 注意：这依赖于 Milvus _get_node_data 实现中是否配置 output_fields
              created_at_vdb = vdb_result.get("__created_at__") # 或者 Milvus 返回的字段名

              node_datas_merged.append({
                   **props, # 图数据库属性
                   "entity_name": props.get("entity_id"), # 确保 entity_name 在这里
                   "rank": degree,
                   "created_at": props.get("created_at", created_at_vdb) # 优先用图数据库的，否则用 VDB 的
                   # "file_path" 应该已经在 props 中了
              })
         # else: # 记录 VDB 中找到但在图中未找到的实体
              # logger.warning(f"Folder {folder_id}: Entity '{vdb_result.get('entity_name')}' found in VDB but not in Graph DB.")


    if not node_datas_merged:
         logger.info(f"Folder {folder_id}: No valid entities found in graph store after VDB search.")
         return "", "", ""

    # 3. 获取相关文本块和边 (tenant-aware)
    # 调用修改后的辅助函数，传递 folder_id
    use_text_units, use_relations = await asyncio.gather(
        _find_most_related_text_unit_from_entities(
            node_datas_merged, query_param, text_chunks_db, knowledge_graph_inst, folder_id=folder_id
        ),
        _find_most_related_edges_from_entities(
            node_datas_merged, query_param, knowledge_graph_inst, folder_id=folder_id
        ),
    )

    # 4. 截断节点数据
    len_before_truncate = len(node_datas_merged)
    truncated_node_datas = truncate_list_by_token_size(
        node_datas_merged,
        key=lambda x: x.get("description", ""), # 安全访问
        max_token_size=query_param.max_token_for_local_context,
    )
    logger.debug(
        f"Folder {folder_id}: Truncated entities from {len_before_truncate} to {len(truncated_node_datas)} (max tokens:{query_param.max_token_for_local_context})"
    )

    logger.info(
        f"Folder {folder_id} - Local query context: {len(truncated_node_datas)} entities, {len(use_relations)} relations, {len(use_text_units)} chunks"
    )

    # 5. 构建 CSV 上下文
    entities_section_list = [
        ["id", "entity", "type", "description", "rank", "created_at", "file_path"]
    ]
    for i, n in enumerate(truncated_node_datas):
        created_at = n.get("created_at", "UNKNOWN")
        if isinstance(created_at, (int, float)): # 格式化时间戳
            try:
                created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))
            except ValueError: # 处理无效时间戳
                 created_at = "Invalid Timestamp"

        file_path = n.get("file_path", "unknown_source") # file_path 来自图数据库

        entities_section_list.append([
            i,
            n.get("entity_name", "UNKNOWN"),
            n.get("entity_type", "UNKNOWN"),
            n.get("description", "UNKNOWN"),
            n.get("rank", 0),
            created_at,
            file_path,
        ])
    entities_context = list_of_list_to_csv(entities_section_list)

    relations_section_list = [
        ["id", "source", "target", "description", "keywords", "weight", "rank", "created_at", "file_path"]
    ]
    for i, e in enumerate(use_relations):
         created_at = e.get("created_at", "UNKNOWN")
         if isinstance(created_at, (int, float)):
              try:
                   created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))
              except ValueError:
                   created_at = "Invalid Timestamp"

         file_path = e.get("file_path", "unknown_source") # file_path 来自图数据库

         relations_section_list.append([
              i,
              e["src_tgt"][0], # src_tgt 是元组 (src, tgt)
              e["src_tgt"][1],
              e.get("description", "UNKNOWN"),
              e.get("keywords", "UNKNOWN"),
              e.get("weight", 0.0),
              e.get("rank", 0),
              created_at,
              file_path,
         ])
    relations_context = list_of_list_to_csv(relations_section_list)

    text_units_section_list = [["id", "content", "file_path"]]
    for i, t in enumerate(use_text_units):
        # text_units 的 file_path 来自 text_chunks_db 获取的数据
        text_units_section_list.append([
            i,
            t.get("content", ""),
            t.get("file_path", "unknown_source")
        ])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    return entities_context, relations_context, text_units_context


async def _find_most_related_text_unit_from_entities( # 需要添加 folder_id
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage, # TenantAwareRedisKVStorage
    knowledge_graph_inst: BaseGraphStorage, # TenantAwareNeo4JStorage
    folder_id: int, # <--- 新增参数
):
    """Finds related text chunks for entities within a specific folder."""
    if folder_id is None:
        raise ValueError("_find_most_related_text_unit_from_entities requires folder_id")
    if not node_datas:
        return []

    # 提取 source_id (通常是 chunk_ids)
    text_units_per_node = []
    for dp in node_datas:
        source_ids = dp.get("source_id")
        if source_ids:
             text_units_per_node.append(split_string_by_multi_markers(source_ids, [GRAPH_FIELD_SEP]))
        else:
             text_units_per_node.append([]) # 添加空列表以保持长度一致

    # 获取每个节点的边 (在指定 folder 内)
    edge_tasks = [knowledge_graph_inst.get_node_edges(dp["entity_name"], folder_id=folder_id) for dp in node_datas]
    edges_per_node = await asyncio.gather(*edge_tasks)

    # 获取所有一跳邻居节点 (在指定 folder 内)
    all_one_hop_entity_names = set()
    for this_edges in edges_per_node:
        if this_edges: # this_edges 是 [(src, tgt), ...]
            # 提取与当前节点相连的邻居 entity_name
            current_node_name = node_datas[edges_per_node.index(this_edges)]["entity_name"]
            neighbors = {tgt if src == current_node_name else src for src, tgt in this_edges}
            all_one_hop_entity_names.update(neighbors)

    # 获取这些邻居节点的 source_id (chunk_ids)
    one_hop_node_list = list(all_one_hop_entity_names)
    one_hop_node_data_tasks = [knowledge_graph_inst.get_node(name, folder_id=folder_id) for name in one_hop_node_list]
    all_one_hop_nodes_data_list = await asyncio.gather(*one_hop_node_data_tasks)

    all_one_hop_text_units_lookup = {}
    for name, node_data in zip(one_hop_node_list, all_one_hop_nodes_data_list):
        if node_data and node_data.get("source_id"):
            all_one_hop_text_units_lookup[name] = set(split_string_by_multi_markers(node_data["source_id"], [GRAPH_FIELD_SEP]))
        # else: # 如果邻居节点没有 source_id，则忽略
        #     all_one_hop_text_units_lookup[name] = set()


    # --- 获取 Text Chunks 内容 (特定于 folder_id) ---
    all_text_units_to_fetch = set()
    chunk_metadata_map = {} # 存储 order 和 edges 信息

    for index, (this_node_text_units, this_edges) in enumerate(zip(text_units_per_node, edges_per_node)):
         current_node_name = node_datas[index]["entity_name"]
         for chunk_id in this_node_text_units:
              if chunk_id not in chunk_metadata_map:
                   all_text_units_to_fetch.add(chunk_id)
                   chunk_metadata_map[chunk_id] = {"order": index, "relation_counts": 0, "edges": this_edges or []} # 存储边的信息

    # 批量获取 chunk 内容 (特定于 folder_id)
    list_to_fetch = list(all_text_units_to_fetch)
    if not list_to_fetch:
         logger.warning(f"No text chunks found for nodes in folder {folder_id}")
         return []

    # 调用 tenant-aware 的 get_by_ids
    fetched_chunks_data = await text_chunks_db.get_by_ids(list_to_fetch, folder_id=folder_id)

    # --- 组合数据并计算关系计数 ---
    valid_fetched_chunks = {}
    for chunk_id, chunk_data in zip(list_to_fetch, fetched_chunks_data):
        if chunk_data and "content" in chunk_data:
             valid_fetched_chunks[chunk_id] = chunk_data
        # else: # 记录未找到或无效的 chunk
        #     logger.warning(f"Chunk {chunk_id} not found or invalid in folder {folder_id}")

    final_chunk_list = []
    for chunk_id, metadata in chunk_metadata_map.items():
        if chunk_id in valid_fetched_chunks:
            # 计算关系计数：检查这个 chunk 的 source_id 是否也出现在其连接的邻居节点的 source_id 中
            relation_counts = 0
            current_node_name = node_datas[metadata["order"]]["entity_name"]
            if metadata["edges"]:
                 neighbors_of_current = {tgt if src == current_node_name else src for src, tgt in metadata["edges"]}
                 for neighbor_name in neighbors_of_current:
                      if neighbor_name in all_one_hop_text_units_lookup:
                           if chunk_id in all_one_hop_text_units_lookup[neighbor_name]:
                                relation_counts += 1

            final_chunk_list.append({
                "id": chunk_id,
                "data": valid_fetched_chunks[chunk_id],
                "order": metadata["order"],
                "relation_counts": relation_counts,
            })

    if not final_chunk_list:
        logger.warning(f"No valid text units found after processing for folder {folder_id}")
        return []

    # 排序和截断
    final_chunk_list = sorted(
        final_chunk_list, key=lambda x: (x["order"], -x["relation_counts"])
    )

    truncated_chunks = truncate_list_by_token_size(
        final_chunk_list,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    logger.debug(
        f"Folder {folder_id}: Truncate chunks from {len(final_chunk_list)} to {len(truncated_chunks)} (max tokens:{query_param.max_token_for_text_unit})"
    )

    # 只返回 chunk 数据本身
    return [t["data"] for t in truncated_chunks]


async def _find_most_related_edges_from_entities( # 需要添加 folder_id
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage, # TenantAwareNeo4JStorage
    folder_id: int, # <--- 新增参数
):
    """Finds related edges for entities within a specific folder."""
    if folder_id is None:
        raise ValueError("_find_most_related_edges_from_entities requires folder_id")
    if not node_datas:
        return []

    all_entity_names = [dp["entity_name"] for dp in node_datas]
    edge_tasks = [knowledge_graph_inst.get_node_edges(name, folder_id=folder_id) for name in all_entity_names]
    all_related_edges_per_node = await asyncio.gather(*edge_tasks)

    all_edges_tuples = set() # 使用 set 存储 (src, tgt) 元组以自动去重
    seen_sorted_pairs = set() # 跟踪已排序的边对以处理双向

    for this_node_edges in all_related_edges_per_node:
        if this_node_edges:
            for src, tgt in this_node_edges:
                sorted_pair = tuple(sorted((src, tgt)))
                if sorted_pair not in seen_sorted_pairs:
                    all_edges_tuples.add((src, tgt)) # 添加原始方向的边
                    seen_sorted_pairs.add(sorted_pair)

    list_of_edge_tuples = list(all_edges_tuples)
    if not list_of_edge_tuples:
        return []

    # 获取边的属性和度数 (特定于 folder_id)
    edge_data_tasks = [knowledge_graph_inst.get_edge(src, tgt, folder_id=folder_id) for src, tgt in list_of_edge_tuples]
    edge_degree_tasks = [knowledge_graph_inst.edge_degree(src, tgt, folder_id=folder_id) for src, tgt in list_of_edge_tuples]

    all_edges_pack, all_edges_degree = await asyncio.gather(
        asyncio.gather(*edge_data_tasks),
        asyncio.gather(*edge_degree_tasks)
    )

    all_edges_data = []
    for edge_tuple, edge_props, degree in zip(list_of_edge_tuples, all_edges_pack, all_edges_degree):
        if edge_props: # 只处理成功获取到属性的边
             # 确保 weight 是 float
             weight = edge_props.get("weight", 0.0)
             try:
                  weight = float(weight)
             except (ValueError, TypeError):
                  weight = 0.0
             edge_props["weight"] = weight # 更新字典中的 weight

             all_edges_data.append({
                  "src_tgt": edge_tuple,
                  "rank": degree, # rank is based on combined node degrees
                  **edge_props # 合并边的所有属性
             })
        # else: # 记录未找到或无效的边
        #      logger.warning(f"Edge {edge_tuple} not found or invalid in folder {folder_id}")

    # 排序和截断
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x.get("rank", 0), x.get("weight", 0.0)), reverse=True
    )

    truncated_edges = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x.get("description", ""), # 安全访问 description
        max_token_size=query_param.max_token_for_global_context,
    )

    logger.debug(
        f"Folder {folder_id}: Truncate relations from {len(all_edges_data)} to {len(truncated_edges)} (max tokens:{query_param.max_token_for_global_context})"
    )

    return truncated_edges


async def _get_edge_data(
    keywords: str,
    knowledge_graph_inst: BaseGraphStorage, # TenantAwareNeo4JStorage
    relationships_vdb: BaseVectorStorage, # TenantAwareMilvusVectorDBStorage
    text_chunks_db: BaseKVStorage,       # TenantAwareRedisKVStorage
    query_param: QueryParam,
    folder_id: int, # <--- 新增参数
):
    """
    Retrieves edge-centric data (relationships, related entities, related text chunks)
    filtered by folder_id.
    """
    if folder_id is None:
        raise ValueError("_get_edge_data requires folder_id")

    logger.info(
        f"Tenant Query edges (folder {folder_id}): {keywords}, top_k: {query_param.top_k}, cosine: {relationships_vdb.cosine_better_than_threshold}"
    )

    # 调用 tenant-aware 的向量查询
    results = await relationships_vdb.query(
        keywords, top_k=query_param.top_k, folder_id=folder_id, ids=query_param.ids
    )

    if not len(results):
        return "", "", "" # Return empty strings for entity, relation, text contexts

    # --- 获取边数据和度数，使用 folder_id ---
    edge_tasks = []
    degree_tasks = []
    valid_results = [] # Store results for which both src and tgt IDs exist

    for r in results:
        # Basic check for required keys before proceeding
        if "src_id" in r and "tgt_id" in r:
            edge_tasks.append(knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"], folder_id=folder_id))
            degree_tasks.append(knowledge_graph_inst.edge_degree(r["src_id"], r["tgt_id"], folder_id=folder_id))
            valid_results.append(r) # Keep track of results corresponding to tasks
        else:
            logger.warning(f"Skipping result due to missing 'src_id' or 'tgt_id': {r}")


    # Ensure tasks are not empty before calling gather
    if not edge_tasks:
         logger.warning(f"No valid relationship results found after filtering for folder {folder_id}")
         return "", "", ""

    edge_datas_results, edge_degree_results = await asyncio.gather(
        asyncio.gather(*edge_tasks),
        asyncio.gather(*degree_tasks),
    )

    # 组合结果，过滤掉 get_edge 返回 None 的情况
    edge_datas = []
    for k, v, d in zip(valid_results, edge_datas_results, edge_degree_results):
         if v is not None: # Check if edge data was found
              edge_datas.append({
                   "src_id": k["src_id"],
                   "tgt_id": k["tgt_id"],
                   "rank": d, # edge_degree
                   "created_at": k.get("__created_at__"), # 从 Milvus 结果获取（如果存在）
                   **v, # 合并从 Neo4j 获取的属性 (description, keywords, weight, file_path etc.)
              })
         else:
              logger.warning(f"Edge data not found in graph for {k['src_id']}->{k['tgt_id']} in folder {folder_id}")


    if not edge_datas:
        logger.warning(f"No valid edge data found in graph storage for folder {folder_id} based on VDB results.")
        return "", "", ""

    # --- 排序和截断 ---
    edge_datas = sorted(
        edge_datas, key=lambda x: (x.get("rank", 0), x.get("weight", 0.0)), reverse=True
    )
    original_edge_count = len(edge_datas)
    edge_datas = truncate_list_by_token_size(
        edge_datas,
        key=lambda x: x.get("description", ""), # 使用 .get 以防万一
        max_token_size=query_param.max_token_for_global_context,
    )
    logger.debug(
        f"Tenant Truncate relations (folder {folder_id}) from {original_edge_count} to {len(edge_datas)} (max tokens:{query_param.max_token_for_global_context})"
    )

    # --- 获取相关的实体和文本单元，传递 folder_id ---
    # 假设 _find_most_related_entities_from_relationships 和
    # _find_related_text_unit_from_relationships 已被修改
    entity_tasks = _find_most_related_entities_from_relationships(
        edge_datas, query_param, knowledge_graph_inst, folder_id=folder_id
    )
    text_unit_tasks = _find_related_text_unit_from_relationships(
        edge_datas, query_param, text_chunks_db, knowledge_graph_inst, folder_id=folder_id
    )

    use_entities, use_text_units = await asyncio.gather(
        entity_tasks,
        text_unit_tasks,
    )
    logger.info(
        f"Tenant Global query (folder {folder_id}) uses {len(use_entities)} entities, {len(edge_datas)} relations, {len(use_text_units)} chunks"
    )

    # --- 构建上下文 CSV 字符串 ---
    # Relations Context
    relations_section_list = [
        ["id", "source", "target", "description", "keywords", "weight", "rank", "created_at", "file_path"]
    ]
    for i, e in enumerate(edge_datas):
        created_at_val = e.get("created_at")
        created_at_str = "Unknown"
        if isinstance(created_at_val, (int, float)):
            try:
                 created_at_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(created_at_val))
            except ValueError: # Handle potential invalid timestamp values
                 created_at_str = "Invalid Timestamp"
        elif isinstance(created_at_val, str): # If already a string
             created_at_str = created_at_val

        relations_section_list.append([
            i,
            e.get("src_id", "Unknown"),
            e.get("tgt_id", "Unknown"),
            e.get("description", ""),
            e.get("keywords", ""),
            e.get("weight", 0.0),
            e.get("rank", 0),
            created_at_str,
            e.get("file_path", "unknown_source"),
        ])
    relations_context = list_of_list_to_csv(relations_section_list)

    # Entities Context
    entites_section_list = [
        ["id", "entity", "type", "description", "rank", "created_at", "file_path"]
    ]
    for i, n in enumerate(use_entities):
         created_at_val = n.get("created_at")
         created_at_str = "Unknown"
         if isinstance(created_at_val, (int, float)):
              try:
                   created_at_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(created_at_val))
              except ValueError:
                   created_at_str = "Invalid Timestamp"
         elif isinstance(created_at_val, str):
              created_at_str = created_at_val

         entites_section_list.append([
              i,
              n.get("entity_name", "Unknown"),
              n.get("entity_type", "UNKNOWN"),
              n.get("description", ""),
              n.get("rank", 0),
              created_at_str,
              n.get("file_path", "unknown_source"),
         ])
    entities_context = list_of_list_to_csv(entites_section_list)

    # Text Units Context
    text_units_section_list = [["id", "content", "file_path"]]
    for i, t in enumerate(use_text_units):
        # Ensure 't' is a dictionary and has 'content'
        if isinstance(t, dict):
             text_units_section_list.append([
                  i,
                  t.get("content", "[Content Missing]"),
                  t.get("file_path", "unknown_source")
             ])
        else:
             logger.warning(f"Skipping invalid text unit item: {t}")
    text_units_context = list_of_list_to_csv(text_units_section_list)

    return entities_context, relations_context, text_units_context


async def _find_most_related_entities_from_relationships(
    edge_datas: List[Dict[str, Any]], # 输入的边数据列表
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage, # TenantAwareNeo4JStorage
    folder_id: int, # <--- 新增参数
) -> List[Dict[str, Any]]:
    """
    Finds related entities based on a list of edge data within a specific folder,
    retrieves their details, ranks them by degree, and truncates the list by token size.
    """
    if folder_id is None:
        raise ValueError("_find_most_related_entities_from_relationships requires folder_id")

    entity_names_seen = set() # 使用集合来自动处理重复的实体名称

    # 从边数据中收集所有相关的实体名称
    for e in edge_datas:
        src_id = e.get("src_id")
        tgt_id = e.get("tgt_id")
        if src_id:
            entity_names_seen.add(src_id)
        if tgt_id:
            entity_names_seen.add(tgt_id)

    entity_names = list(entity_names_seen) # 转换为列表

    if not entity_names:
        logger.debug(f"No entity names derived from edge data for folder {folder_id}.")
        return [] # 如果没有实体名称，直接返回空列表

    # --- 并行获取节点数据和度数，使用 folder_id ---
    node_tasks = []
    degree_tasks = []
    for entity_name in entity_names:
        node_tasks.append(knowledge_graph_inst.get_node(entity_name, folder_id=folder_id))
        degree_tasks.append(knowledge_graph_inst.node_degree(entity_name, folder_id=folder_id))

    node_datas_results, node_degrees_results = await asyncio.gather(
        asyncio.gather(*node_tasks),
        asyncio.gather(*degree_tasks),
    )

    # --- 组合结果，过滤掉未找到的节点 ---
    node_datas = []
    for k, n, d in zip(entity_names, node_datas_results, node_degrees_results):
        if n is not None: # 确保节点存在于当前 folder 中
            # 从 Neo4j 返回的 properties(n) 中提取所需字段
            node_entry = {
                 "entity_name": k, # 这里的 k 就是 entity_id/node_label
                 "rank": d, # 度数
                 "entity_type": n.get("entity_type", "UNKNOWN"),
                 "description": n.get("description", ""),
                 "source_id": n.get("source_id", ""),
                 "file_path": n.get("file_path", "unknown_source"),
                 "created_at": n.get("__created_at__"), # 如果 Neo4j 存储了这个元数据
                 # 可以添加从 'n' 字典中获取的其他属性
                 # **n # 或者直接合并所有属性，但要注意可能存在的冗余或冲突
            }
            node_datas.append(node_entry)
        else:
             logger.warning(f"Entity '{k}' referenced by edges not found in folder {folder_id}")


    if not node_datas:
         logger.debug(f"No valid entity data found for derived entities in folder {folder_id}.")
         return []

    # --- 按度数排序并截断 ---
    # 注意：这里没有按权重排序，只按度数（rank）
    node_datas = sorted(node_datas, key=lambda x: x.get('rank', 0), reverse=True)

    len_node_datas_before_trunc = len(node_datas)
    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x.get("description", ""), # 按描述的 token 数截断
        max_token_size=query_param.max_token_for_local_context,
    )
    logger.debug(
        f"Tenant Truncate entities (from edges) in folder {folder_id} from {len_node_datas_before_trunc} to {len(node_datas)} (max tokens:{query_param.max_token_for_local_context})"
    )

    return node_datas


async def _find_related_text_unit_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage, # Should be TenantAwareRedisKVStorage instance
    knowledge_graph_inst: BaseGraphStorage, # May not be needed here, check usage
    folder_id: int, # <--- 新增参数
) -> list[TextChunkSchema]: # Return type hint added for clarity
    """Finds related text chunks based on source IDs from relationship data within a folder."""
    if folder_id is None:
        raise ValueError("_find_related_text_unit_from_relationships requires folder_id")

    # Extract all unique source chunk IDs from the edge data
    all_chunk_ids = set()
    for edge_data in edge_datas:
        source_id_str = edge_data.get("source_id")
        if source_id_str:
            # Split in case multiple source chunks are associated with a relationship
            chunk_ids = split_string_by_multi_markers(source_id_str, [GRAPH_FIELD_SEP])
            all_chunk_ids.update(chunk_ids)

    if not all_chunk_ids:
        logger.warning("No source chunk IDs found in relationship data.")
        return []

    unique_chunk_id_list = list(all_chunk_ids)
    logger.debug(f"Found {len(unique_chunk_id_list)} unique chunk IDs from relationships for folder {folder_id}.")

    # Fetch chunk data using the tenant-aware get_by_ids (or batched get_by_id)
    # Note: text_chunks_db.get_by_ids requires folder_id
    # If get_by_ids is not implemented in TenantAwareRedisKVStorage, use batched get_by_id
    if hasattr(text_chunks_db, 'get_by_ids'):
         chunks_data = await text_chunks_db.get_by_ids(unique_chunk_id_list, folder_id=folder_id)
    else:
         # Fallback to fetching one by one if get_by_ids is not available
         logger.warning("text_chunks_db does not have get_by_ids, fetching chunks individually.")
         fetch_tasks = [text_chunks_db.get_by_id(c_id, folder_id=folder_id) for c_id in unique_chunk_id_list]
         chunks_data = await asyncio.gather(*fetch_tasks)


    # Filter out None results and ensure 'content' exists
    valid_chunks_data = [
        chunk for chunk in chunks_data if chunk is not None and "content" in chunk
    ]

    if not valid_chunks_data:
        logger.warning(f"No valid text chunk data found for the retrieved IDs in folder {folder_id}.")
        return []

    logger.debug(f"Retrieved {len(valid_chunks_data)} valid text chunks for relationships in folder {folder_id}.")

    # Truncate the list based on token size
    # Assuming valid_chunks_data contains dicts with a 'content' key
    truncated_text_units = truncate_list_by_token_size(
        valid_chunks_data,
        key=lambda x: x.get("content", ""), # Use .get for safety
        max_token_size=query_param.max_token_for_text_unit,
    )

    logger.debug(
        f"Truncated chunks from {len(valid_chunks_data)} to {len(truncated_text_units)} "
        f"(max tokens: {query_param.max_token_for_text_unit}) for folder {folder_id}"
    )

    # The return type should match TextChunkSchema expected by callers
    # Ensure the dictionaries have the necessary keys.
    # For now, we return the list of dictionaries.
    return truncated_text_units


def combine_contexts(entities, relationships, sources):
    # Function to extract entities, relationships, and sources from context strings
    hl_entities, ll_entities = entities[0], entities[1]
    hl_relationships, ll_relationships = relationships[0], relationships[1]
    hl_sources, ll_sources = sources[0], sources[1]
    # Combine and deduplicate the entities
    combined_entities = process_combine_contexts(hl_entities, ll_entities)

    # Combine and deduplicate the relationships
    combined_relationships = process_combine_contexts(
        hl_relationships, ll_relationships
    )

    # Combine and deduplicate the sources
    combined_sources = process_combine_contexts(hl_sources, ll_sources)

    return combined_entities, combined_relationships, combined_sources


async def naive_query(
    query: str,
    chunks_vdb: BaseVectorStorage, # 应该是 TenantAwareMilvusVectorDBStorage 实例
    text_chunks_db: BaseKVStorage, # 应该是 TenantAwareRedisKVStorage 实例
    query_param: QueryParam,
    global_config: dict[str, Any], # 使用 Any 避免严格的字符串类型
    folder_id: int, # <--- 新增参数
    hashing_kv: BaseKVStorage | None = None, # 应该是 TenantAwareRedisKVStorage 实例
    system_prompt: str | None = None,
) -> Union[str, AsyncIterator[str]]:
    """Performs a naive vector search query within a specific folder."""
    if folder_id is None:
        raise ValueError("folder_id must be provided for naive_query operation")

    # 1. Handle cache (传递 folder_id)
    use_model_func = (
        query_param.model_func
        if query_param.model_func
        else global_config["llm_model_func"]
    )
    # 使用 query_param.mode 作为缓存模式键
    args_hash = compute_args_hash(query_param.mode, query, folder_id, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv,
        args_hash,
        query,
        query_param.mode, # 使用参数中的 mode
        cache_type="query",
        folder_id=folder_id, # <--- 传递 folder_id
    )
    if cached_response is not None:
        return cached_response

    # 2. Query vector database (传递 folder_id)
    results = await chunks_vdb.query(
        query, top_k=query_param.top_k, folder_id=folder_id, ids=query_param.ids
    )
    if not len(results):
        logger.warning(f"Naive query for folder {folder_id} returned no vector results.")
        return PROMPTS["fail_response"]

    # 3. Get text chunks (传递 folder_id)
    chunks_ids = [r["id"] for r in results]
    # 假设 text_chunks_db 是 TenantAwareRedisKVStorage 实例
    chunks = await text_chunks_db.get_by_ids(chunks_ids, folder_id=folder_id)

    # Filter out invalid chunks (None or missing 'content')
    valid_chunks = [
        chunk for chunk in chunks if chunk is not None and "content" in chunk
    ]

    if not valid_chunks:
        logger.warning(f"No valid text chunks found for folder {folder_id} after vector search.")
        return PROMPTS["fail_response"]

    # 4. Truncate chunks
    maybe_trun_chunks = truncate_list_by_token_size(
        valid_chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    if not maybe_trun_chunks:
        logger.warning(f"No chunks left after truncation for folder {folder_id}.")
        return PROMPTS["fail_response"]

    logger.debug(
        f"Truncated chunks from {len(valid_chunks)} to {len(maybe_trun_chunks)} for folder {folder_id} (max tokens:{query_param.max_token_for_text_unit})"
    )

    # 5. Build context string
    section = "\n--New Chunk--\n".join(
        [
            # 优先使用 chunk 中存储的 file_path，如果没有则使用 results 中的
            "File path: " + chunk.get("file_path", result.get("file_path", "unknown_source")) + "\n" + chunk["content"]
            for chunk, result in zip(maybe_trun_chunks, results) # 假设 maybe_trun_chunks 和 results 的顺序对应
        ]
    )


    if query_param.only_need_context:
        return section

    # 6. Process conversation history (逻辑不变)
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    # 7. Build system prompt (逻辑不变)
    sys_prompt_temp = system_prompt if system_prompt else PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=section,
        response_type=query_param.response_type,
        history=history_context,
    )

    if query_param.only_need_prompt:
        return sys_prompt

    len_of_prompts = len(encode_string_by_tiktoken(query + sys_prompt))
    logger.debug(f"[naive_query folder:{folder_id}] Prompt Tokens: {len_of_prompts}")

    # 8. Call LLM (逻辑不变)
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream # 支持流式输出
    )

    # 9. Process and cache response (传递 folder_id)
    # 对于流式响应，需要在调用端收集完整响应后再缓存
    if isinstance(response, str):
         if len(response) > len(sys_prompt): # 简单清理，可能不完美
              cleaned_response = (
                   response.replace(sys_prompt, "")
                   .replace("user", "").replace("model", "")
                   .replace(query, "").replace("<system>", "").replace("</system>", "")
                   .strip()
              )
         else:
              cleaned_response = response.strip()

         # 保存到缓存 (传递 folder_id)
         await save_to_cache(
              hashing_kv,
              CacheData(
                   args_hash=args_hash,
                   content=cleaned_response, # 缓存清理后的响应
                   prompt=query,
                   quantized=quantized,
                   min_val=min_val,
                   max_val=max_val,
                   mode=query_param.mode, # 使用参数中的 mode
                   cache_type="query",
              ),
              folder_id=folder_id, # <--- 传递 folder_id
         )
         return cleaned_response
    elif hasattr(response, "__aiter__"):
         # 对于异步迭代器（流式响应），直接返回，由调用者处理缓存
         logger.debug(f"Returning async iterator for naive_query folder:{folder_id}")
         return response
    else:
         logger.error(f"Unexpected response type from LLM for naive_query folder:{folder_id}: {type(response)}")
         return PROMPTS["fail_response"]


async def kg_query_with_keywords(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,    # TenantAwareNeo4JStorage
    entities_vdb: BaseVectorStorage,          # TenantAwareMilvusVectorDBStorage
    relationships_vdb: BaseVectorStorage,     # TenantAwareMilvusVectorDBStorage
    text_chunks_db: BaseKVStorage,            # TenantAwareRedisKVStorage
    query_param: QueryParam,
    global_config: dict[str, Any],            # 使用 Any 避免严格类型
    folder_id: int,                           # <--- 新增参数
    hashing_kv: BaseKVStorage | None = None,  # TenantAwareRedisKVStorage
    system_prompt: str | None = None,         # <--- 添加 system_prompt 参数
) -> str | AsyncIterator[str]:
    """
    Refactored kg_query that does NOT extract keywords by itself, now tenant-aware.
    It expects hl_keywords and ll_keywords to be set in query_param, or defaults to empty.
    Then it uses those to build context and produce a final LLM response within a folder.

    Args:
        query: The user's query text.
        knowledge_graph_inst: Tenant-aware graph storage instance.
        entities_vdb: Tenant-aware entity vector storage instance.
        relationships_vdb: Tenant-aware relationship vector storage instance.
        text_chunks_db: Tenant-aware text chunk KV storage instance.
        query_param: Query parameters, expected to contain hl_keywords/ll_keywords.
        global_config: Global configuration dictionary.
        folder_id: The ID of the folder to query within.
        hashing_kv: Tenant-aware cache storage instance.
        system_prompt: Optional system prompt override.

    Returns:
        The query result (string or async iterator).
    """
    if folder_id is None:
        raise ValueError("folder_id must be provided for kg_query_with_keywords")

    # ---------------------------
    # 1) Handle potential cache for query results (Tenant-Aware)
    # ---------------------------
    use_model_func = (
        query_param.model_func
        if query_param.model_func
        else global_config.get("llm_model_func") # 使用 .get 更安全
    )
    if not use_model_func:
         # 如果没有找到模型函数，需要处理错误
         logger.error("LLM model function not configured.")
         return PROMPTS["fail_response"] # 或者抛出异常

    # 使用 folder_id 调用 handle_cache
    args_hash = compute_args_hash(query_param.mode, query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query", folder_id=folder_id # 传递 folder_id
    )
    if cached_response is not None:
        return cached_response

    # ---------------------------
    # 2) RETRIEVE KEYWORDS FROM query_param (逻辑不变)
    # ---------------------------
    hl_keywords = getattr(query_param, "hl_keywords", []) or []
    ll_keywords = getattr(query_param, "ll_keywords", []) or []

    if not hl_keywords and not ll_keywords:
        logger.warning(f"No keywords found in query_param for folder {folder_id}.")
        return PROMPTS["fail_response"]
    if not ll_keywords and query_param.mode in ["local", "hybrid"]:
        logger.warning(f"Low-level keywords empty for folder {folder_id}, switching to global.")
        query_param.mode = "global"
    if not hl_keywords and query_param.mode in ["global", "hybrid"]:
        logger.warning(f"High-level keywords empty for folder {folder_id}, switching to local.")
        query_param.mode = "local"

    # 展平关键词列表 (逻辑不变)
    ll_keywords_flat = [item for sublist in ll_keywords if isinstance(sublist, list) for item in sublist] if any(isinstance(i, list) for i in ll_keywords) else ll_keywords
    hl_keywords_flat = [item for sublist in hl_keywords if isinstance(sublist, list) for item in sublist] if any(isinstance(i, list) for i in hl_keywords) else hl_keywords

    ll_keywords_str = ", ".join(map(str, ll_keywords_flat)) if ll_keywords_flat else ""
    hl_keywords_str = ", ".join(map(str, hl_keywords_flat)) if hl_keywords_flat else ""


    # ---------------------------
    # 3) BUILD CONTEXT (Tenant-Aware)
    # ---------------------------
    # 调用修改后的 _build_query_context 并传递 folder_id
    context = await _build_query_context(
        ll_keywords_str,
        hl_keywords_str,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        folder_id=folder_id, # <--- 传递 folder_id
    )
    if not context:
        logger.warning(f"Failed to build context for folder {folder_id}")
        return PROMPTS["fail_response"]

    if query_param.only_need_context:
        return context

    # ---------------------------
    # 4) BUILD THE SYSTEM PROMPT + CALL LLM (逻辑不变, 但使用正确的 system_prompt)
    # ---------------------------
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    # 使用传入的 system_prompt 或默认值
    sys_prompt_temp = system_prompt if system_prompt else PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context,
        response_type=query_param.response_type,
        history=history_context,
    )

    if query_param.only_need_prompt:
        return sys_prompt

    len_of_prompts = len(encode_string_by_tiktoken(query + sys_prompt))
    logger.debug(f"[kg_query_with_keywords] Folder {folder_id} Prompt Tokens: {len_of_prompts}")

    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
        # 可以在这里传递 mode 给 LLM 函数，如果它需要的话
        # mode=query_param.mode
    )

    # --- 处理和缓存结果 (Tenant-Aware) ---
    if isinstance(response, str):
        # 清理响应内容的逻辑不变
        if len(response) > len(sys_prompt):
            response = (
                response.replace(sys_prompt, "")
                .replace("user", "").replace("model", "")
                .replace(query, "").replace("<system>", "").replace("</system>", "")
                .strip()
            )
        # 使用 folder_id 调用 save_to_cache
        await save_to_cache(
            hashing_kv,
            CacheData(
                args_hash=args_hash, content=response, prompt=query,
                quantized=quantized, min_val=min_val, max_val=max_val,
                mode=query_param.mode, cache_type="query"
            ),
            folder_id=folder_id # <--- 传递 folder_id
        )
    elif hasattr(response, "__aiter__"):
        # 对于流式响应，缓存需要在外部完成（例如，在调用端收集完所有块后）
        # 或者修改 save_to_cache 以支持异步迭代器（更复杂）
        # 目前我们不在流式传输时缓存
        logger.debug(f"Streaming response for folder {folder_id}, caching will be skipped in this function.")
        pass # 返回异步迭代器

    return response



async def query_with_keywords(
    query: str,
    prompt: str, # 这个 prompt 参数的用途需要明确
    param: QueryParam,
    folder_id: int, # <--- 新增参数
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    global_config: dict[str, Any], # 使用 Any
    hashing_kv: BaseKVStorage | None = None,
) -> str | AsyncIterator[str]:
    """
    Async: Extract keywords from the query and then use them for retrieving information within a specific folder.

    1. Extracts high-level and low-level keywords from the query.
    2. Formats the query with the extracted keywords and prompt? (See Note)
    3. Uses the appropriate query method based on param.mode.

    Args:
        query: The user's query.
        prompt: Additional prompt provided. Its usage depends on how underlying functions interpret it.
                Typically, system prompts are handled separately.
        param: Query parameters.
        folder_id: The ID of the folder to operate within.
        knowledge_graph_inst: Knowledge graph storage.
        entities_vdb: Entities vector database.
        relationships_vdb: Relationships vector database.
        chunks_vdb: Document chunks vector database.
        text_chunks_db: Text chunks storage.
        global_config: Global configuration.
        hashing_kv: Cache storage.

    Returns:
        Query response or async iterator.
    """
    if folder_id is None:
         raise ValueError("folder_id must be provided for query_with_keywords")

    # 1. Extract keywords (调用已修改的函数，传递 folder_id)
    hl_keywords, ll_keywords = await get_keywords_from_query(
        query=query,
        query_param=param,
        global_config=global_config,
        hashing_kv=hashing_kv,
        folder_id=folder_id, # <--- 传递 folder_id
    )

    # 2. Format the question/prompt (基于原始代码)
    # WARNING: 确认这个 formatted_question 的意图。通常 RAG 的 query 是用户输入，
    # prompt (尤其是 system_prompt) 是给 LLM 的指令+上下文。
    # 将 prompt 和 keywords 加到 query 前面可能会改变用户原始意图。
    # 如果 prompt 实际上是 system_prompt 的一部分，它应该在调用底层函数时
    # 作为 system_prompt 参数传递，或者在底层函数内部构建 system_prompt 时使用。
    # 这里的实现遵循了原始代码的格式化方式。
    ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else "None"
    hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else "None"
    formatted_question = f"{prompt}\n\n### Keywords:\nHigh-level: {hl_keywords_str}\nLow-level: {ll_keywords_str}\n\n### Query:\n{query}"
    logger.debug(f"Formatted question for tenant query (folder {folder_id}):\n{formatted_question}")


    # 3. Use appropriate query method based on mode (调用已修改的函数，传递 folder_id 和 formatted_question)
    if param.mode in ["local", "global", "hybrid"]:
        # 注意：这里传递的是 formatted_question，而不是原始 query
        return await kg_query_with_keywords( # 假设已修改
            formatted_question, # <--- 使用格式化后的问题
            knowledge_graph_inst,
            entities_vdb,
            relationships_vdb,
            text_chunks_db,
            param, # param 可能需要更新以反映 keywords，如果 kg_query_with_keywords 期望如此
            global_config,
            folder_id=folder_id, # <--- 传递 folder_id
            hashing_kv=hashing_kv,
            # system_prompt=None # 假设 system_prompt 在 kg_query_with_keywords 内部处理
        )
    elif param.mode == "naive":
         # 注意：这里传递的是 formatted_question
        return await naive_query( # 假设已修改
            formatted_question, # <--- 使用格式化后的问题
            chunks_vdb,
            text_chunks_db,
            param,
            global_config,
            folder_id=folder_id, # <--- 传递 folder_id
            hashing_kv=hashing_kv,
            # system_prompt=None # 假设 system_prompt 在 naive_query 内部处理
        )
    elif param.mode == "mix":
         # 注意：这里传递的是 formatted_question
        return await mix_kg_vector_query( # 假设已修改
            formatted_question, # <--- 使用格式化后的问题
            knowledge_graph_inst,
            entities_vdb,
            relationships_vdb,
            chunks_vdb,
            text_chunks_db,
            param,
            global_config,
            folder_id=folder_id, # <--- 传递 folder_id
            hashing_kv=hashing_kv,
             # system_prompt=None # 假设 system_prompt 在 mix_kg_vector_query 内部处理
        )
    else:
        raise ValueError(f"Unknown mode {param.mode}")
