from __future__ import annotations

import asyncio
import html
import io
import csv
import json
import logging
import logging.handlers
import os
import re
from dataclasses import dataclass
from functools import wraps
from hashlib import md5
from typing import Any, Callable
import xml.etree.ElementTree as ET
import numpy as np
import tiktoken
from lightrag.prompt import PROMPTS
from dotenv import load_dotenv
from .base import BaseKVStorage

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)

VERBOSE_DEBUG = os.getenv("VERBOSE", "false").lower() == "true"


def verbose_debug(msg: str, *args, **kwargs):
    """Function for outputting detailed debug information.
    When VERBOSE_DEBUG=True, outputs the complete message.
    When VERBOSE_DEBUG=False, outputs only the first 50 characters.

    Args:
        msg: The message format string
        *args: Arguments to be formatted into the message
        **kwargs: Keyword arguments passed to logger.debug()
    """
    if VERBOSE_DEBUG:
        logger.debug(msg, *args, **kwargs)
    else:
        # Format the message with args first
        if args:
            formatted_msg = msg % args
        else:
            formatted_msg = msg
        # Then truncate the formatted message
        truncated_msg = (
            formatted_msg[:100] + "..." if len(formatted_msg) > 100 else formatted_msg
        )
        logger.debug(truncated_msg, **kwargs)


def set_verbose_debug(enabled: bool):
    """Enable or disable verbose debug output"""
    global VERBOSE_DEBUG
    VERBOSE_DEBUG = enabled


statistic_data = {"llm_call": 0, "llm_cache": 0, "embed_call": 0}

# Initialize logger
logger = logging.getLogger("lightrag")
logger.propagate = False  # prevent log message send to root loggger
# Let the main application configure the handlers
logger.setLevel(logging.INFO)

# Set httpx logging level to WARNING
logging.getLogger("httpx").setLevel(logging.WARNING)


class LightragPathFilter(logging.Filter):
    """Filter for lightrag logger to filter out frequent path access logs"""

    def __init__(self):
        super().__init__()
        # Define paths to be filtered
        self.filtered_paths = [
            "/documents",
            "/health",
            "/webui/",
            "/documents/pipeline_status",
        ]
        # self.filtered_paths = ["/health", "/webui/"]

    def filter(self, record):
        try:
            # Check if record has the required attributes for an access log
            if not hasattr(record, "args") or not isinstance(record.args, tuple):
                return True
            if len(record.args) < 5:
                return True

            # Extract method, path and status from the record args
            method = record.args[1]
            path = record.args[2]
            status = record.args[4]

            # Filter out successful GET requests to filtered paths
            if (
                method == "GET"
                and (status == 200 or status == 304)
                and path in self.filtered_paths
            ):
                return False

            return True
        except Exception:
            # In case of any error, let the message through
            return True


def setup_logger(
    logger_name: str,
    level: str = "INFO",
    add_filter: bool = False,
    log_file_path: str | None = None,
    enable_file_logging: bool = True,
):
    """Set up a logger with console and optionally file handlers

    Args:
        logger_name: Name of the logger to set up
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        add_filter: Whether to add LightragPathFilter to the logger
        log_file_path: Path to the log file. If None and file logging is enabled, defaults to lightrag.log in LOG_DIR or cwd
        enable_file_logging: Whether to enable logging to a file (defaults to True)
    """
    # Configure formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    simple_formatter = logging.Formatter("%(levelname)s: %(message)s")

    logger_instance = logging.getLogger(logger_name)
    logger_instance.setLevel(level)
    logger_instance.handlers = []  # Clear existing handlers
    logger_instance.propagate = False

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(simple_formatter)
    console_handler.setLevel(level)
    logger_instance.addHandler(console_handler)

    # Add file handler by default unless explicitly disabled
    if enable_file_logging:
        # Get log file path
        if log_file_path is None:
            log_dir = os.getenv("LOG_DIR", os.getcwd())
            log_file_path = os.path.abspath(os.path.join(log_dir, "lightrag.log"))

        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        # Get log file max size and backup count from environment variables
        log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
        log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

        try:
            # Add file handler
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_file_path,
                maxBytes=log_max_bytes,
                backupCount=log_backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(detailed_formatter)
            file_handler.setLevel(level)
            logger_instance.addHandler(file_handler)
        except PermissionError as e:
            logger.warning(f"Could not create log file at {log_file_path}: {str(e)}")
            logger.warning("Continuing with console logging only")

    # Add path filter if requested
    if add_filter:
        path_filter = LightragPathFilter()
        logger_instance.addFilter(path_filter)


class UnlimitedSemaphore:
    """A context manager that allows unlimited access."""

    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc, tb):
        pass


ENCODER = None


@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable
    # concurrent_limit: int = 16

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)


def locate_json_string_body_from_string(content: str) -> str | None:
    """Locate the JSON string body from a string"""
    try:
        maybe_json_str = re.search(r"{.*}", content, re.DOTALL)
        if maybe_json_str is not None:
            maybe_json_str = maybe_json_str.group(0)
            maybe_json_str = maybe_json_str.replace("\\n", "")
            maybe_json_str = maybe_json_str.replace("\n", "")
            maybe_json_str = maybe_json_str.replace("'", '"')
            # json.loads(maybe_json_str) # don't check here, cannot validate schema after all
            return maybe_json_str
    except Exception:
        pass
        # try:
        #     content = (
        #         content.replace(kw_prompt[:-1], "")
        #         .replace("user", "")
        #         .replace("model", "")
        #         .strip()
        #     )
        #     maybe_json_str = "{" + content.split("{")[1].split("}")[0] + "}"
        #     json.loads(maybe_json_str)

        return None


def convert_response_to_json(response: str) -> dict[str, Any]:
    json_str = locate_json_string_body_from_string(response)
    assert json_str is not None, f"Unable to parse JSON from response: {response}"
    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {json_str}")
        raise e from None


def compute_args_hash(*args: Any, cache_type: str | None = None) -> str:
    """Compute a hash for the given arguments.
    Args:
        *args: Arguments to hash
        cache_type: Type of cache (e.g., 'keywords', 'query', 'extract')
    Returns:
        str: Hash string
    """
    import hashlib

    # Convert all arguments to strings and join them
    args_str = "".join([str(arg) for arg in args])
    if cache_type:
        args_str = f"{cache_type}:{args_str}"

    # Compute MD5 hash
    return hashlib.md5(args_str.encode()).hexdigest()


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute a unique ID for a given content string.

    The ID is a combination of the given prefix and the MD5 hash of the content string.
    """
    return prefix + md5(content.encode()).hexdigest()


def limit_async_func_call(max_size: int):
    """Add restriction of maximum concurrent async calls using asyncio.Semaphore"""

    def final_decro(func):
        sem = asyncio.Semaphore(max_size)

        @wraps(func)
        async def wait_func(*args, **kwargs):
            async with sem:
                result = await func(*args, **kwargs)
                return result

        return wait_func

    return final_decro


def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro


def load_json(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


def write_json(json_obj, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    tokens = ENCODER.encode(content)
    return tokens


def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    content = ENCODER.decode(tokens)
    return content


def pack_user_ass_to_openai_messages(*args: str):
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


# Refer the utils functions of the official GraphRAG implementation:
# https://github.com/microsoft/graphrag
def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


def is_float_regex(value: str) -> bool:
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def truncate_list_by_token_size(
    list_data: list[Any], key: Callable[[Any], str], max_token_size: int
) -> list[int]:
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data


def list_of_list_to_csv(data: list[list[str]]) -> str:
    output = io.StringIO()
    writer = csv.writer(
        output,
        quoting=csv.QUOTE_ALL,  # Quote all fields
        escapechar="\\",  # Use backslash as escape character
        quotechar='"',  # Use double quotes
        lineterminator="\n",  # Explicit line terminator
    )
    writer.writerows(data)
    return output.getvalue()


def csv_string_to_list(csv_string: str) -> list[list[str]]:
    # Clean the string by removing NUL characters
    cleaned_string = csv_string.replace("\0", "")

    output = io.StringIO(cleaned_string)
    reader = csv.reader(
        output,
        quoting=csv.QUOTE_ALL,  # Match the writer configuration
        escapechar="\\",  # Use backslash as escape character
        quotechar='"',  # Use double quotes
    )

    try:
        return [row for row in reader]
    except csv.Error as e:
        raise ValueError(f"Failed to parse CSV string: {str(e)}")
    finally:
        output.close()


def save_data_to_file(data, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def xml_to_json(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Print the root element's tag and attributes to confirm the file has been correctly loaded
        print(f"Root element: {root.tag}")
        print(f"Root attributes: {root.attrib}")

        data = {"nodes": [], "edges": []}

        # Use namespace
        namespace = {"": "http://graphml.graphdrawing.org/xmlns"}

        for node in root.findall(".//node", namespace):
            node_data = {
                "id": node.get("id").strip('"'),
                "entity_type": node.find("./data[@key='d0']", namespace).text.strip('"')
                if node.find("./data[@key='d0']", namespace) is not None
                else "",
                "description": node.find("./data[@key='d1']", namespace).text
                if node.find("./data[@key='d1']", namespace) is not None
                else "",
                "source_id": node.find("./data[@key='d2']", namespace).text
                if node.find("./data[@key='d2']", namespace) is not None
                else "",
            }
            data["nodes"].append(node_data)

        for edge in root.findall(".//edge", namespace):
            edge_data = {
                "source": edge.get("source").strip('"'),
                "target": edge.get("target").strip('"'),
                "weight": float(edge.find("./data[@key='d3']", namespace).text)
                if edge.find("./data[@key='d3']", namespace) is not None
                else 0.0,
                "description": edge.find("./data[@key='d4']", namespace).text
                if edge.find("./data[@key='d4']", namespace) is not None
                else "",
                "keywords": edge.find("./data[@key='d5']", namespace).text
                if edge.find("./data[@key='d5']", namespace) is not None
                else "",
                "source_id": edge.find("./data[@key='d6']", namespace).text
                if edge.find("./data[@key='d6']", namespace) is not None
                else "",
            }
            data["edges"].append(edge_data)

        # Print the number of nodes and edges found
        print(f"Found {len(data['nodes'])} nodes and {len(data['edges'])} edges")

        return data
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def process_combine_contexts(hl: str, ll: str):
    header = None
    list_hl = csv_string_to_list(hl.strip())
    list_ll = csv_string_to_list(ll.strip())

    if list_hl:
        header = list_hl[0]
        list_hl = list_hl[1:]
    if list_ll:
        header = list_ll[0]
        list_ll = list_ll[1:]
    if header is None:
        return ""

    if list_hl:
        list_hl = [",".join(item[1:]) for item in list_hl if item]
    if list_ll:
        list_ll = [",".join(item[1:]) for item in list_ll if item]

    combined_sources = []
    seen = set()

    for item in list_hl + list_ll:
        if item and item not in seen:
            combined_sources.append(item)
            seen.add(item)

    combined_sources_result = [",\t".join(header)]

    for i, item in enumerate(combined_sources, start=1):
        combined_sources_result.append(f"{i},\t{item}")

    combined_sources_result = "\n".join(combined_sources_result)

    return combined_sources_result


async def get_best_cached_response(
    hashing_kv: BaseKVStorage, # 类型提示
    current_embedding: np.ndarray,
    similarity_threshold: float = 0.95,
    mode: str = "default",
    folder_id: int | None = None, # <--- 新增参数
    use_llm_check: bool = False,
    llm_func: Callable | None = None,
    original_prompt: str | None = None,
    cache_type: str | None = None,
) -> str | None:
    """Finds the best cached response based on embedding similarity within a folder."""
    if folder_id is None:
        logger.warning("folder_id is required for get_best_cached_response. No cache retrieved.")
        return None

    logger.debug(
        f"get_best_cached_response: folder={folder_id} mode={mode} cache_type={cache_type} use_llm_check={use_llm_check}"
    )

    # 获取指定 folder 和 mode 的缓存数据
    mode_cache = await hashing_kv.get_by_id(mode, folder_id=folder_id)
    if not mode_cache:
        logger.debug(f"No cache found for mode '{mode}' in folder {folder_id}.")
        return None # 没有该模式的缓存

    best_similarity = -1.0
    best_response = None
    best_prompt = None
    best_cache_id = None

    # 仅迭代当前 folder+mode 的缓存条目
    for cache_id, cache_data in mode_cache.items():
        # 检查 cache_type 是否匹配
        if cache_type and cache_data.get("cache_type") != cache_type:
            continue
        # 检查是否存在 embedding 数据
        if cache_data.get("embedding") is None or cache_data.get("embedding_shape") is None:
             logger.warning(f"Cache entry {cache_id} in folder {folder_id} is missing embedding data.")
             continue

        try:
            # 反量化 embedding
            cached_quantized = np.frombuffer(
                bytes.fromhex(cache_data["embedding"]), dtype=np.uint8
            ).reshape(cache_data["embedding_shape"])
            cached_embedding = dequantize_embedding( # dequantize_embedding 需要导入或定义
                cached_quantized,
                cache_data["embedding_min"],
                cache_data["embedding_max"],
            )

            # 计算相似度
            similarity = cosine_similarity(current_embedding, cached_embedding) # cosine_similarity 需要导入或定义

            if similarity > best_similarity:
                best_similarity = similarity
                best_response = cache_data.get("return") # 使用 .get 以防万一
                best_prompt = cache_data.get("original_prompt")
                best_cache_id = cache_id
        except Exception as e:
             logger.error(f"Error processing cache entry {cache_id} in folder {folder_id}: {e}")
             continue # 跳过有问题的条目

    # 检查最佳相似度是否超过阈值
    if best_response is not None and best_similarity >= similarity_threshold:
        # --- LLM 检查逻辑 (如果启用) ---
        if use_llm_check and llm_func and original_prompt and best_prompt:
            compare_prompt = PROMPTS["similarity_check"].format(
                original_prompt=original_prompt, cached_prompt=best_prompt
            )
            try:
                llm_result = await llm_func(compare_prompt)
                llm_result = llm_result.strip()
                llm_similarity = float(llm_result)

                # 使用 LLM 的相似度进行最终判断
                if llm_similarity < similarity_threshold:
                    # LLM 判定不够相似，拒绝缓存
                    log_data = {
                        # ... (日志内容)
                        "folder_id": folder_id,
                    }
                    logger.debug(json.dumps(log_data, ensure_ascii=False))
                    logger.info(f"Cache rejected by LLM (folder:{folder_id} mode:{mode} type:{cache_type}) similarity: {llm_similarity:.4f}")
                    return None # 返回 None 表示缓存未命中
                else:
                     # LLM 判定相似，使用 LLM 的分数更新 best_similarity 用于日志记录
                     best_similarity = llm_similarity

            except Exception as e:
                logger.warning(f"LLM similarity check failed for folder {folder_id}: {e}")
                # LLM 检查失败时，是返回 None (拒绝缓存) 还是回退到向量相似度？
                # 为了安全起见，返回 None
                return None

        # --- 缓存命中 ---
        prompt_display = best_prompt[:50] + "..." if best_prompt and len(best_prompt) > 50 else best_prompt
        log_data = {
            "event": "cache_hit", "folder_id": folder_id, "type": cache_type, "mode": mode,
            "similarity": round(best_similarity, 4), "cache_id": best_cache_id,
            "original_prompt": prompt_display,
        }
        logger.debug(json.dumps(log_data, ensure_ascii=False))
        return best_response # 返回缓存内容

    # 相似度未达到阈值
    logger.debug(f"Highest similarity {best_similarity:.4f} below threshold {similarity_threshold} for folder {folder_id}")
    return None # 缓存未命中


def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot_product / (norm1 * norm2)


def quantize_embedding(embedding: np.ndarray | list[float], bits: int = 8) -> tuple:
    """Quantize embedding to specified bits"""
    # Convert list to numpy array if needed
    if isinstance(embedding, list):
        embedding = np.array(embedding)

    # Calculate min/max values for reconstruction
    min_val = embedding.min()
    max_val = embedding.max()

    # Quantize to 0-255 range
    scale = (2**bits - 1) / (max_val - min_val)
    quantized = np.round((embedding - min_val) * scale).astype(np.uint8)

    return quantized, min_val, max_val


def dequantize_embedding(
    quantized: np.ndarray, min_val: float, max_val: float, bits=8
) -> np.ndarray:
    """Restore quantized embedding"""
    scale = (max_val - min_val) / (2**bits - 1)
    return (quantized * scale + min_val).astype(np.float32)


async def handle_cache(
    hashing_kv: BaseKVStorage | None, # 类型提示用 BaseKVStorage
    args_hash: str,
    prompt: str,
    mode: str = "default",
    cache_type: str | None = None,
    folder_id: int | None = None, # <--- 新增参数
):
    """Generic cache handling function, now tenant-aware."""
    # 如果没有缓存存储或未提供 folder_id (对于需要隔离的模式)，则直接返回
    if hashing_kv is None:
        return None, None, None, None
    # 对于非 'default' 模式 (即各种查询模式)，folder_id 是必需的
    if mode != "default" and folder_id is None:
         logger.warning(f"folder_id is required for caching mode '{mode}'. Cache lookup skipped.")
         # 或者 raise ValueError("folder_id required for non-default cache mode")
         return None, None, None, None

    # 全局配置检查 (例如 enable_llm_cache)
    global_enable_cache = True # 默认启用
    embedding_cache_config = {"enabled": False} # 默认禁用 embedding cache

    if hasattr(hashing_kv, 'global_config'):
         global_config = hashing_kv.global_config
         # 检查特定模式的缓存是否启用
         if mode == "default": # 假设 'default' 对应 entity extraction
              global_enable_cache = global_config.get("enable_llm_cache_for_entity_extract", True)
         else: # 其他查询模式
              global_enable_cache = global_config.get("enable_llm_cache", True)

         embedding_cache_config = global_config.get(
              "embedding_cache_config",
              {"enabled": False, "similarity_threshold": 0.95, "use_llm_check": False}
         )
    else:
         logger.warning("hashing_kv does not have global_config attribute. Assuming cache is enabled.")


    if not global_enable_cache:
         return None, None, None, None


    # --- 基于 Embedding 的缓存查找 (需要 folder_id) ---
    is_embedding_cache_enabled = embedding_cache_config["enabled"]
    quantized = min_val = max_val = None

    # 仅当 embedding cache 启用且 folder_id 有效时执行
    if is_embedding_cache_enabled and mode != "default" and folder_id is not None:
        use_llm_check = embedding_cache_config.get("use_llm_check", False)
        llm_model_func = getattr(hashing_kv, 'global_config', {}).get("llm_model_func") if use_llm_check else None

        current_embedding = await hashing_kv.embedding_func([prompt])
        quantized, min_val, max_val = quantize_embedding(current_embedding[0]) # quantize_embedding 需要导入或定义

        # 调用修改后的 get_best_cached_response
        best_cached_response = await get_best_cached_response(
            hashing_kv,
            current_embedding[0],
            similarity_threshold=embedding_cache_config["similarity_threshold"],
            mode=mode,
            folder_id=folder_id, # <--- 传递 folder_id
            use_llm_check=use_llm_check,
            llm_func=llm_model_func,
            original_prompt=prompt,
            cache_type=cache_type,
        )
        if best_cached_response is not None:
            logger.debug(f"Embedding cache hit (folder:{folder_id} mode:{mode} type:{cache_type})")
            return best_cached_response, None, None, None
        else:
            logger.debug(f"Embedding cache missed (folder:{folder_id} mode:{mode} type:{cache_type})")
            # 返回 None 表示未命中，但返回量化信息以便后续保存
            return None, quantized, min_val, max_val

    # --- 基于 Hash 的缓存查找 (需要 folder_id) ---
    # 检查是否存在特定于租户的 get 方法
    if hasattr(hashing_kv, "get_by_mode_and_id") and folder_id is not None:
         # 这个方法可能不是我们 TenantAwareRedisKVStorage 的标准部分
         # 我们需要使用 get_by_id 并传入 mode 作为 key 的一部分
         # mode_cache = await hashing_kv.get_by_mode_and_id(mode, args_hash, folder_id=folder_id) or {} # 假设有此方法
         # 使用 get_by_id(key=mode) 更符合我们 TenantAwareRedisKVStorage 的设计
         mode_cache_content = await hashing_kv.get_by_id(mode, folder_id=folder_id)
         mode_cache = mode_cache_content if mode_cache_content else {}

    elif folder_id is not None: # 使用标准的 get_by_id
         mode_cache_content = await hashing_kv.get_by_id(mode, folder_id=folder_id)
         mode_cache = mode_cache_content if mode_cache_content else {}
    elif mode == "default": # 对于 default 模式，如果不需要隔离，可以考虑不传 folder_id
         # 但为了统一，最好总是传递 folder_id，如果 default 模式不需要隔离，
         # TenantAwareRedisKVStorage 的 _get_tenant_key 可以特殊处理 folder_id=None
         # 这里我们假设所有模式都需要 folder_id
         logger.warning(f"folder_id is None for handle_cache mode '{mode}', hash lookup skipped.")
         return None, None, None, None
    else: # folder_id is None for non-default mode - 已在前面处理
         return None, None, None, None


    # 检查 hash 是否在当前 folder 的 mode cache 中
    if args_hash in mode_cache:
         # 检查 cache_type 是否匹配 (如果提供了 cache_type)
         cache_entry = mode_cache[args_hash]
         if cache_type is None or cache_entry.get("cache_type") == cache_type:
              logger.debug(f"Hash cache hit (folder:{folder_id} mode:{mode} type:{cache_type})")
              return cache_entry.get("return"), None, None, None # 返回缓存内容

    logger.debug(f"Hash cache missed (folder:{folder_id} mode:{mode} type:{cache_type})")
    return None, None, None, None # 明确返回 None 表示未命中


@dataclass
class CacheData:
    args_hash: str
    content: str
    prompt: str
    quantized: np.ndarray | None = None
    min_val: float | None = None
    max_val: float | None = None
    mode: str = "default"
    cache_type: str = "query"


async def save_to_cache(
    hashing_kv: BaseKVStorage | None, # 类型提示用 BaseKVStorage
    cache_data: CacheData,
    folder_id: int | None = None, # <--- 新增参数
):
    """Save data to cache within a specific folder, with improved handling."""
    # --- 前置检查 ---
    if hashing_kv is None: return
    if not cache_data.content: return # 不缓存空内容
    if hasattr(cache_data.content, "__aiter__"): # 不缓存流式响应
        logger.debug("Streaming response detected, skipping cache save")
        return

    # 对于非 'default' 模式，folder_id 是必需的
    if cache_data.mode != "default" and folder_id is None:
        logger.warning(f"folder_id is required to save cache for mode '{cache_data.mode}'. Save skipped.")
        return

    # 全局配置检查 (逻辑同 handle_cache)
    global_enable_cache = True
    if hasattr(hashing_kv, 'global_config'):
         global_config = hashing_kv.global_config
         if cache_data.mode == "default":
              global_enable_cache = global_config.get("enable_llm_cache_for_entity_extract", True)
         else:
              global_enable_cache = global_config.get("enable_llm_cache", True)

    if not global_enable_cache:
         return

    # --- 获取当前 folder 的缓存 ---
    try:
        # 使用 tenant-aware get_by_id 获取当前 folder+mode 的数据
        mode_cache_content = await hashing_kv.get_by_id(cache_data.mode, folder_id=folder_id)
        mode_cache = mode_cache_content if mode_cache_content else {}
    except Exception as e:
        logger.error(f"Error retrieving cache for mode '{cache_data.mode}' in folder {folder_id}: {e}")
        mode_cache = {} # 出错时假定为空，尝试覆盖

    # --- 检查内容是否重复 ---
    if cache_data.args_hash in mode_cache:
        existing_entry = mode_cache[cache_data.args_hash]
        # 比较核心内容，忽略 embedding 等元数据
        if existing_entry.get("return") == cache_data.content:
            logger.debug(f"Cache content unchanged for hash {cache_data.args_hash} in folder {folder_id}, skipping update.")
            return # 内容未变，无需保存

    # --- 准备新的缓存条目 ---
    new_cache_entry = {
        "return": cache_data.content,
        "cache_type": cache_data.cache_type,
        "embedding": cache_data.quantized.tobytes().hex() if cache_data.quantized is not None else None,
        "embedding_shape": cache_data.quantized.shape if cache_data.quantized is not None else None,
        "embedding_min": cache_data.min_val,
        "embedding_max": cache_data.max_val,
        "original_prompt": cache_data.prompt, # 保存原始提示用于 LLM 检查 (如果启用)
    }

    # 更新 mode_cache 字典
    mode_cache[cache_data.args_hash] = new_cache_entry

    # --- 保存更新后的缓存到 Redis ---
    try:
        # 使用 tenant-aware upsert 保存整个 mode 的数据
        await hashing_kv.upsert({cache_data.mode: mode_cache}, folder_id=folder_id)
        logger.debug(f"Cache saved for hash {cache_data.args_hash} in folder {folder_id} mode {cache_data.mode}")
    except Exception as e:
        logger.error(f"Error saving cache for mode '{cache_data.mode}' in folder {folder_id}: {e}")


def safe_unicode_decode(content):
    # Regular expression to find all Unicode escape sequences of the form \uXXXX
    unicode_escape_pattern = re.compile(r"\\u([0-9a-fA-F]{4})")

    # Function to replace the Unicode escape with the actual character
    def replace_unicode_escape(match):
        # Convert the matched hexadecimal value into the actual Unicode character
        return chr(int(match.group(1), 16))

    # Perform the substitution
    decoded_content = unicode_escape_pattern.sub(
        replace_unicode_escape, content.decode("utf-8")
    )

    return decoded_content


def exists_func(obj, func_name: str) -> bool:
    """Check if a function exists in an object or not.
    :param obj:
    :param func_name:
    :return: True / False
    """
    if callable(getattr(obj, func_name, None)):
        return True
    else:
        return False


def get_conversation_turns(
    conversation_history: list[dict[str, Any]], num_turns: int
) -> str:
    """
    Process conversation history to get the specified number of complete turns.

    Args:
        conversation_history: List of conversation messages in chronological order
        num_turns: Number of complete turns to include

    Returns:
        Formatted string of the conversation history
    """
    # Check if num_turns is valid
    if num_turns <= 0:
        return ""

    # Group messages into turns
    turns: list[list[dict[str, Any]]] = []
    messages: list[dict[str, Any]] = []

    # First, filter out keyword extraction messages
    for msg in conversation_history:
        if msg["role"] == "assistant" and (
            msg["content"].startswith('{ "high_level_keywords"')
            or msg["content"].startswith("{'high_level_keywords'")
        ):
            continue
        messages.append(msg)

    # Then process messages in chronological order
    i = 0
    while i < len(messages) - 1:
        msg1 = messages[i]
        msg2 = messages[i + 1]

        # Check if we have a user-assistant or assistant-user pair
        if (msg1["role"] == "user" and msg2["role"] == "assistant") or (
            msg1["role"] == "assistant" and msg2["role"] == "user"
        ):
            # Always put user message first in the turn
            if msg1["role"] == "assistant":
                turn = [msg2, msg1]  # user, assistant
            else:
                turn = [msg1, msg2]  # user, assistant
            turns.append(turn)
        i += 2

    # Keep only the most recent num_turns
    if len(turns) > num_turns:
        turns = turns[-num_turns:]

    # Format the turns into a string
    formatted_turns: list[str] = []
    for turn in turns:
        formatted_turns.extend(
            [f"user: {turn[0]['content']}", f"assistant: {turn[1]['content']}"]
        )

    return "\n".join(formatted_turns)


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure that there is always an event loop available.

    This function tries to get the current event loop. If the current event loop is closed or does not exist,
    it creates a new event loop and sets it as the current event loop.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.
    """
    try:
        # Try to get the current event loop
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop

    except RuntimeError:
        # If no event loop exists or it is closed, create a new one
        logger.info("Creating a new event loop in main thread.")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop


def lazy_external_import(module_name: str, class_name: str) -> Callable[..., Any]:
    """Lazily import a class from an external module based on the package of the caller."""
    # Get the caller's module and package
    import inspect

    caller_frame = inspect.currentframe().f_back
    module = inspect.getmodule(caller_frame)
    package = module.__package__ if module else None

    def import_class(*args: Any, **kwargs: Any):
        import importlib

        module = importlib.import_module(module_name, package=package)
        cls = getattr(module, class_name)
        return cls(*args, **kwargs)

    return import_class


def get_content_summary(content: str, max_length: int = 250) -> str:
    """Get summary of document content

    Args:
        content: Original document content
        max_length: Maximum length of summary

    Returns:
        Truncated content with ellipsis if needed
    """
    content = content.strip()
    if len(content) <= max_length:
        return content
    return content[:max_length] + "..."


def clean_text(text: str) -> str:
    """Clean text by removing null bytes (0x00) and whitespace

    Args:
        text: Input text to clean

    Returns:
        Cleaned text
    """
    return text.strip().replace("\x00", "")


def check_storage_env_vars(storage_name: str) -> None:
    """Check if all required environment variables for storage implementation exist

    Args:
        storage_name: Storage implementation name

    Raises:
        ValueError: If required environment variables are missing
    """
    from lightrag.kg import STORAGE_ENV_REQUIREMENTS

    required_vars = STORAGE_ENV_REQUIREMENTS.get(storage_name, [])
    missing_vars = [var for var in required_vars if var not in os.environ]

    if missing_vars:
        raise ValueError(
            f"Storage implementation '{storage_name}' requires the following "
            f"environment variables: {', '.join(missing_vars)}"
        )


class TokenTracker:
    """Track token usage for LLM calls."""

    def __init__(self):
        self.reset()

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(self)

    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0

    def add_usage(self, token_counts):
        """Add token usage from one LLM call.

        Args:
            token_counts: A dictionary containing prompt_tokens, completion_tokens, total_tokens
        """
        self.prompt_tokens += token_counts.get("prompt_tokens", 0)
        self.completion_tokens += token_counts.get("completion_tokens", 0)

        # If total_tokens is provided, use it directly; otherwise calculate the sum
        if "total_tokens" in token_counts:
            self.total_tokens += token_counts["total_tokens"]
        else:
            self.total_tokens += token_counts.get(
                "prompt_tokens", 0
            ) + token_counts.get("completion_tokens", 0)

        self.call_count += 1

    def get_usage(self):
        """Get current usage statistics."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
        }

    def __str__(self):
        usage = self.get_usage()
        return (
            f"LLM call count: {usage['call_count']}, "
            f"Prompt tokens: {usage['prompt_tokens']}, "
            f"Completion tokens: {usage['completion_tokens']}, "
            f"Total tokens: {usage['total_tokens']}"
        )
