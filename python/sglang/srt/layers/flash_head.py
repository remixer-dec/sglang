# Copyright 2023-2024 SGLang Team
# Copyright (C) 2025 Embedl AB
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
FlashHead implementation for faster efficient language model head.

FlashHead speeds up token generation by up to 50% using a clustering-based
approach. Instead of computing logits for the entire vocabulary, FlashHead
activates only a small subset of clusters at each step, identifying the
desired token significantly faster.

The system achieves >99.9% accuracy compared to the original model through
a specific clustering scheme adapted to token logits.
"""

import json
import logging
import os
from typing import Iterable, Optional, Tuple, Union

import torch
from torch import nn

logger = logging.getLogger(__name__)


# Global FlashHead instance storage for inter-process communication
_GLOBAL_FLASH_HEAD: Optional["FlashHead"] = None


def get_global_flash_head() -> Optional["FlashHead"]:
    """Get the global FlashHead instance."""
    return _GLOBAL_FLASH_HEAD


def set_global_flash_head(flash_head: Optional["FlashHead"]) -> None:
    """Set the global FlashHead instance."""
    global _GLOBAL_FLASH_HEAD
    _GLOBAL_FLASH_HEAD = flash_head


def _resolve_asset(model_dir: str, relative_path: str) -> str:
    """Resolve the path to a FlashHead asset file.

    Args:
        model_dir: The model directory (local path).
        relative_path: The relative path to the asset within the model directory.

    Returns:
        The absolute path to the asset file.

    Raises:
        FileNotFoundError: If the asset file does not exist.
    """
    p = os.path.join(model_dir, relative_path)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing FlashHead asset: {p}")
    return p


def _load_centroids(
    vocab_size: int,
    hidden_size: int,
    model_dir: str,
    cache_dir: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load centroids and cluster assignments from cache files.

    Args:
        vocab_size: The vocabulary size of the model.
        hidden_size: The hidden dimension of the model.
        model_dir: The model directory.
        cache_dir: The relative path to the FlashHead cache directory.
        device: The device to load tensors on.
        dtype: The dtype for the centroids.

    Returns:
        A tuple of (centroids, cluster_assignments).

    Raises:
        ValueError: If the cache files are invalid or incompatible.
    """
    from safetensors.torch import load_file

    cache_file_rel = os.path.join(cache_dir, "clustering_cache.safetensors")
    meta_file_rel = os.path.join(cache_dir, "clustering_config.json")

    cache_file = _resolve_asset(model_dir, cache_file_rel)
    meta_file = _resolve_asset(model_dir, meta_file_rel)

    with open(meta_file, encoding="utf-8") as f:
        metadata = json.load(f)

    if metadata.get("format") not in (None, "safetensors"):
        raise ValueError(
            f"Expected safetensors format, found: {metadata.get('format')}"
        )

    if metadata.get("vocab_size") != vocab_size:
        raise ValueError(
            f"Cache vocab_size {metadata.get('vocab_size')} != expected {vocab_size}"
        )
    if metadata.get("hidden_size") != hidden_size:
        raise ValueError(
            f"Cache hidden_size {metadata.get('hidden_size')} != expected {hidden_size}"
        )

    tensors = load_file(cache_file)

    if "centroids" not in tensors or "cluster_assignments" not in tensors:
        raise KeyError(
            f"Cache missing required tensors. Found keys: {list(tensors.keys())}"
        )

    centroids = tensors["centroids"]
    cluster_assignments = tensors["cluster_assignments"]

    if cluster_assignments.ndim != 1 or cluster_assignments.shape[0] != vocab_size:
        raise ValueError(
            f"cluster_assignments shape {tuple(cluster_assignments.shape)}; "
            f"expected ({vocab_size},)"
        )

    centroids = centroids.to(device=device, dtype=dtype)
    cluster_assignments = cluster_assignments.to(device=device)

    return centroids, cluster_assignments


def get_flash_head_parameters(
    vocab_size: int,
    hidden_size: int,
    model_dir: str,
    cache_dir: str,
    device: torch.device,
    dtype: torch.dtype,
    n_clusters: Optional[int] = None,
) -> dict:
    """Get parameters for the FlashHead layer.

    Args:
        vocab_size: The vocabulary size of the model.
        hidden_size: The hidden dimension of the model.
        model_dir: The model directory.
        cache_dir: The relative path to the FlashHead cache directory.
        device: The device to load tensors on.
        dtype: The dtype for the centroids.
        n_clusters: The number of clusters (defaults to vocab_size / 16).

    Returns:
        A dictionary with 'centroids' and 'vocab_maps_tensor' keys.
    """
    if n_clusters is None:
        n_clusters = int(vocab_size / 16)

    centroids, cluster_assignments = _load_centroids(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        model_dir=model_dir,
        cache_dir=cache_dir,
        device=device,
        dtype=dtype,
    )

    total_clusters = n_clusters
    cluster_to_vocab_maps = [
        torch.where(cluster_assignments == i)[0] for i in range(total_clusters)
    ]

    combined_centroids = torch.zeros(
        (hidden_size, total_clusters),
        device=device,
        dtype=dtype,
    )
    centroids_reshaped = centroids.squeeze(0).squeeze(0)
    combined_centroids[:, :n_clusters] = centroids_reshaped

    max_len = max(m.shape[0] for m in cluster_to_vocab_maps)
    vocab_maps_tensor = torch.full(
        (len(cluster_to_vocab_maps), max_len), -1, device=device
    )
    for i, m in enumerate(cluster_to_vocab_maps):
        length = m.shape[0]
        vocab_maps_tensor[i, :length] = m
        vocab_maps_tensor[i, length:] = m[0]

    return {
        "centroids": combined_centroids,
        "vocab_maps_tensor": vocab_maps_tensor,
    }


class FlashHead(nn.Module):
    """
    FlashHead: A drop-in module for the standard lm_head layer that speeds up
    token generation by up to 50%.

    Instead of heavy dense matrix multiplication, a two-step retrieval process
    is used:
    1. Find the top-k clusters based on hidden state similarity to centroids
    2. Compute logits only for tokens in those clusters

    This approach identifies the desired token significantly faster by activating
    only a small subset of the vocabulary at each step.

    Args:
        lm_head_weight: The weight tensor from the original lm_head layer.
        centroids: The cluster centroids tensor.
        vocab_maps_tensor: A mapping between cluster centroid index and token index.
        n_probes: Number of clusters to probe (defaults to n_clusters / 16).
        special_token_ids: Tokens to process independently of clusters.
    """

    def __init__(
        self,
        lm_head_weight: torch.Tensor,
        centroids: torch.Tensor,
        vocab_maps_tensor: torch.Tensor,
        n_probes: Optional[int] = None,
        special_token_ids: Optional[Union[int, Iterable[int]]] = None,
    ):
        super().__init__()

        self.vocab_size = lm_head_weight.shape[0]
        self.hidden_size = lm_head_weight.shape[1]

        # Store the original lm_head weight for computing logits on selected tokens
        self.register_buffer("lm_head_weight", lm_head_weight)

        self.register_buffer("vocab_maps_tensor", vocab_maps_tensor)
        self.register_buffer("centroids", centroids.contiguous())

        if n_probes is None:
            n_probes = int(centroids.shape[1] / 16)
        self.n_probes = n_probes

        pre_norm = centroids / centroids.norm(dim=0, keepdim=True)
        pre_norm = pre_norm.t().contiguous()
        self.register_buffer("pre_normalized_centroids", pre_norm)

        self.cluster_linear = nn.Linear(
            pre_norm.shape[1],
            pre_norm.shape[0],
            bias=False,
        )
        self.cluster_linear.weight = nn.Parameter(pre_norm)

        self.register_buffer(
            "vocab_maps_lengths", (vocab_maps_tensor != -1).sum(dim=1)
        )
        self.register_buffer(
            "row_indices", torch.arange(vocab_maps_tensor.shape[1])[None, :]
        )
        self.register_buffer("output_buffer", torch.zeros((1, 1), dtype=torch.int64))

        special_token_list = []
        if special_token_ids is None:
            special_token_list = []
        elif isinstance(special_token_ids, int):
            special_token_list = [special_token_ids]
        else:
            special_token_list = list(special_token_ids)

        special_token_list = [
            int(t) for t in special_token_list if 0 <= int(t) < self.vocab_size
        ]

        self.register_buffer(
            "special_token_ids_tensor",
            torch.tensor(special_token_list, dtype=torch.int64),
            persistent=False,
        )

    def _get_cluster_probs(
        self, hidden_states: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        """Compute probabilities over clusters."""
        similarities = torch.nn.functional.linear(
            hidden_states, self.centroids.t(), bias=None
        )
        probs = torch.softmax(similarities / temperature, dim=-1)
        return probs

    def _get_top_clusters(
        self,
        hidden_states: torch.Tensor,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Get the top-k clusters for the given hidden states."""
        if do_sample:
            probs = self._get_cluster_probs(
                hidden_states=hidden_states, temperature=temperature
            )
            B, T, num_clusters = probs.shape
            probs_flat = probs.view(-1, num_clusters)
            sampled_indices = torch.multinomial(
                probs_flat, self.n_probes, replacement=False
            )
            top_clusters = sampled_indices.view(B, T, self.n_probes)
        else:
            similarities = self.cluster_linear(hidden_states)
            _, top_clusters = torch.topk(similarities, k=self.n_probes, dim=-1)
        return top_clusters

    def _get_cluster_logits(
        self,
        hidden_states: torch.Tensor,
        top_clusters: torch.Tensor,
        use_identical_tiebreak: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute logits only for tokens in the selected clusters."""
        if top_clusters.shape[1] > 1 or top_clusters.shape[0] > 1:
            raise NotImplementedError(
                "FlashHead only supports seq-len=1 and batch_size=1"
            )

        cluster_indices = top_clusters[0, 0]
        maps = self.vocab_maps_tensor.index_select(0, cluster_indices)
        indices = maps.flatten()

        if self.special_token_ids_tensor.numel() > 0:
            special_ids = self.special_token_ids_tensor.to(device=indices.device)
            indices = torch.unique(torch.cat([indices, special_ids], dim=0))

        mapping = None
        if use_identical_tiebreak:
            sorted_result = indices.sort()
            indices = sorted_result.values
            mapping = sorted_result.indices

        result = self.lm_head_weight.index_select(0, indices)
        final_result = (
            torch.nn.functional.linear(hidden_states, result, bias=None),
            mapping,
        )
        return final_result

    def forward(self, _hidden_states: torch.Tensor):
        """Guard for forward method."""
        raise ValueError(
            "Forward method not supported, please use `get_next_token`."
        )

    def get_next_token(
        self,
        hidden_states: torch.Tensor,
        do_sample: bool = False,
        temperature: float = 1.0,
        use_identical_tiebreak: bool = False,
    ) -> torch.Tensor:
        """
        Return the next token, given `hidden_states`.

        Args:
            hidden_states: The output of the model body with shape [1, 1, hidden_size].
            do_sample: Whether to sample the next token according to probabilities,
                or simply return the most probable.
            temperature: The temperature to use in the softmax
                (both for cluster probabilities and token probabilities).
                Only relevant when `do_sample` is True.
            use_identical_tiebreak: Whether to reorder the logits so that when two
                logits are the same, the new head will use the same tiebreak as
                the original.

        Returns:
            The next predicted token ID as a tensor with shape [1, 1].
        """
        top_clusters = self._get_top_clusters(
            hidden_states,
            do_sample=do_sample,
            temperature=temperature,
        )
        cluster_logits, mapping = self._get_cluster_logits(
            hidden_states, top_clusters, use_identical_tiebreak
        )

        if do_sample:
            probs = (cluster_logits[:, -1, :] / temperature).softmax(dim=-1)
            cluster_token_idx = torch.multinomial(probs, num_samples=1)
        else:
            cluster_token_idx = cluster_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if use_identical_tiebreak and mapping is not None:
                cluster_token_idx = mapping[cluster_token_idx]

        cluster_indices = top_clusters[0, 0]
        maps = self.vocab_maps_tensor.index_select(0, cluster_indices)
        indices = maps.flatten().to(torch.int64)
        if self.special_token_ids_tensor.numel() > 0:
            special_ids = self.special_token_ids_tensor.to(device=indices.device)
            indices = torch.unique(torch.cat([indices, special_ids], dim=0))
        if use_identical_tiebreak:
            indices = indices.sort().values

        vocab_index = indices[cluster_token_idx]
        self.output_buffer[0][0] = vocab_index.item()
        return self.output_buffer


def load_flash_head_from_config(
    model_dir: str,
    flash_head_cache_dir: str,
    vocab_size: int,
    hidden_size: int,
    lm_head_weight: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    special_token_ids: Optional[Union[int, Iterable[int]]] = None,
) -> FlashHead:
    """
    Load a FlashHead module from the model configuration.

    Args:
        model_dir: The model directory path.
        flash_head_cache_dir: The relative path to FlashHead cache within model_dir.
        vocab_size: The vocabulary size of the model.
        hidden_size: The hidden dimension of the model.
        lm_head_weight: The weight tensor from the original lm_head layer.
        device: The device to load the module on.
        dtype: The dtype for the module.
        special_token_ids: Tokens to process independently of clusters.

    Returns:
        A FlashHead module initialized with the cache weights.
    """
    params = get_flash_head_parameters(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        model_dir=model_dir,
        cache_dir=flash_head_cache_dir,
        device=device,
        dtype=dtype,
    )

    flash_head = FlashHead(
        lm_head_weight=lm_head_weight,
        centroids=params["centroids"],
        vocab_maps_tensor=params["vocab_maps_tensor"],
        special_token_ids=special_token_ids,
    ).to(device=device, dtype=dtype)

    logger.info(
        f"[FlashHead] Initialized with cache from {flash_head_cache_dir}, "
        f"vocab_size={vocab_size}, hidden_size={hidden_size}"
    )

    return flash_head


def detect_flash_head_config(
    model_path: str, model_name: str = ""
) -> Tuple[Optional[str], Optional[str]]:
    """
    Detect if a model has FlashHead support.

    Args:
        model_path: The model path (directory for safetensors, file for GGUF).
        model_name: The model name (for GGUF detection by name).

    Returns:
        A tuple of (model_dir, flash_head_cache_dir) if found, (None, None) otherwise.
        - model_dir: The directory containing both model and FlashHead assets
        - flash_head_cache_dir: The relative path to FlashHead cache within model_dir
    """
    # Determine if this is a file (GGUF) or directory (safetensors)
    if os.path.isfile(model_path):
        # GGUF model - model_path is a file, use parent directory
        model_dir = os.path.dirname(model_path)
        model_filename = os.path.basename(model_path)
        is_gguf = model_path.lower().endswith(".gguf")
    else:
        model_dir = model_path
        model_filename = ""
        is_gguf = False

    # Check for flash_head_cache_dir in config.json (safetensors models)
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            if "flash_head_cache_dir" in config:
                cache_dir = config["flash_head_cache_dir"]
                # Verify the cache directory exists
                full_cache_path = os.path.join(model_dir, cache_dir)
                if os.path.isdir(full_cache_path):
                    logger.info(f"[FlashHead] Detected from config.json: {cache_dir}")
                    return model_dir, cache_dir
                else:
                    logger.warning(
                        f"[FlashHead] flash_head_cache_dir specified but not found: {full_cache_path}"
                    )
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"[FlashHead] Error reading config.json: {e}")

    # Check for GGUF models with "FlashHead" in the name
    # For GGUF files, check both the filename and model_name parameter
    check_names = [model_name, model_filename] if is_gguf else [model_name]
    for name in check_names:
        if "FlashHead" in name or "flashhead" in name.lower():
            default_cache_dir = "flash_head_assets"
            full_cache_path = os.path.join(model_dir, default_cache_dir)
            if os.path.isdir(full_cache_path):
                # Verify required files exist
                cache_file = os.path.join(
                    full_cache_path, "clustering_cache.safetensors"
                )
                config_file = os.path.join(full_cache_path, "clustering_config.json")
                if os.path.exists(cache_file) and os.path.exists(config_file):
                    logger.info(
                        f"[FlashHead] Detected from model name (GGUF): {default_cache_dir}"
                    )
                    return model_dir, default_cache_dir
                else:
                    logger.warning(
                        f"[FlashHead] flash_head_assets found but missing required files: "
                        f"clustering_cache.safetensors and/or clustering_config.json"
                    )
            else:
                logger.warning(
                    f"[FlashHead] Model name contains 'FlashHead' but default cache dir "
                    f"not found: {full_cache_path}"
                )
            break  # Only check once if we found a matching name

    return None, None


def sanitize_flash_head_config(model_dir: str) -> bool:
    """
    Sanitize the config.json to remove FlashHead-specific fields.

    This allows loading FlashHead models as standard models (e.g., Llama, Qwen)
    without requiring the embedl package. The FlashHead acceleration is applied
    separately after model loading.

    Args:
        model_dir: The model directory path.

    Returns:
        True if config was modified, False otherwise.
    """
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return False

    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        modified = False

        # Remove auto_map which references embedl package
        if "auto_map" in config:
            logger.info("[FlashHead] Removing auto_map from config.json")
            del config["auto_map"]
            modified = True

        # Update architectures - remove "FlashHead" from architecture names
        if "architectures" in config:
            new_architectures = []
            for arch in config["architectures"]:
                if "FlashHead" in arch:
                    # e.g., "FlashHeadLlamaForCausalLM" -> "LlamaForCausalLM"
                    new_arch = arch.replace("FlashHead", "")
                    logger.info(
                        f"[FlashHead] Updating architecture: {arch} -> {new_arch}"
                    )
                    new_architectures.append(new_arch)
                    modified = True
                else:
                    new_architectures.append(arch)
            if modified:
                config["architectures"] = new_architectures

        # Update model_type - remove "flash_head_" prefix
        if "model_type" in config:
            model_type = config["model_type"]
            if "flash_head_" in model_type:
                new_model_type = model_type.replace("flash_head_", "")
                logger.info(
                    f"[FlashHead] Updating model_type: {model_type} -> {new_model_type}"
                )
                config["model_type"] = new_model_type
                modified = True

        if modified:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            logger.info(f"[FlashHead] Config sanitized at {config_path}")

        return modified

    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"[FlashHead] Error sanitizing config: {e}")
        return False


def get_flash_head_special_token_ids(model_dir: str) -> Optional[list]:
    """
    Get special token IDs from the model config for FlashHead.

    Args:
        model_dir: The model directory path.

    Returns:
        List of special token IDs if found in config, None otherwise.
    """
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            return config.get("flash_head_special_token_ids")
        except (json.JSONDecodeError, IOError):
            pass
    return None
