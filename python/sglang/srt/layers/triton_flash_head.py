# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""
Triton-accelerated FlashHead implementation.

This module provides fused Triton kernels for FlashHead that avoid the
memory allocation overhead of index_select operations.

Key optimizations:
1. Fused cluster similarity computation
2. Direct dot product computation without weight copying
3. Fused argmax to avoid materializing all logits
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


@triton.jit
def _cluster_similarities_kernel(
    hidden_states_ptr,      # [hidden_size]
    centroids_ptr,          # [hidden_size, n_clusters] (column-major for coalesced access)
    output_ptr,             # [n_clusters]
    hidden_size,
    n_clusters,
    BLOCK_H: tl.constexpr,
):
    """
    Compute dot product between hidden_states and each cluster centroid.

    Each program handles one cluster.
    """
    cluster_idx = tl.program_id(0)

    if cluster_idx >= n_clusters:
        return

    # Accumulate dot product
    acc = tl.zeros([1], dtype=tl.float32)

    for h_start in range(0, hidden_size, BLOCK_H):
        h_offs = h_start + tl.arange(0, BLOCK_H)
        mask = h_offs < hidden_size

        # Load hidden states
        h = tl.load(hidden_states_ptr + h_offs, mask=mask, other=0.0).to(tl.float32)

        # Load centroid values (column-major: centroids[h, cluster] = centroids_ptr[h * n_clusters + cluster])
        c = tl.load(centroids_ptr + h_offs * n_clusters + cluster_idx, mask=mask, other=0.0).to(tl.float32)

        acc += tl.sum(h * c)

    tl.store(output_ptr + cluster_idx, acc)


@triton.jit
def _compute_selected_logits_kernel(
    hidden_states_ptr,      # [hidden_size]
    lm_head_ptr,            # [vocab_size, hidden_size] (row-major)
    token_indices_ptr,      # [num_tokens] - token IDs to compute logits for
    output_logits_ptr,      # [num_tokens]
    hidden_size,
    lm_head_stride,         # hidden_size (row stride for lm_head)
    num_tokens,
    BLOCK_H: tl.constexpr,
):
    """
    Compute logits for selected tokens without copying weights.

    This is the key optimization: instead of index_select (which allocates memory),
    we directly compute dot products.

    Each program handles one token.
    """
    token_idx = tl.program_id(0)

    if token_idx >= num_tokens:
        return

    # Get vocab index for this token
    vocab_idx = tl.load(token_indices_ptr + token_idx)

    # Handle padding (-1 indices)
    if vocab_idx < 0:
        tl.store(output_logits_ptr + token_idx, float('-inf'))
        return

    # Compute dot product: hidden_states @ lm_head[vocab_idx]
    acc = tl.zeros([1], dtype=tl.float32)

    for h_start in range(0, hidden_size, BLOCK_H):
        h_offs = h_start + tl.arange(0, BLOCK_H)
        mask = h_offs < hidden_size

        # Load hidden states
        h = tl.load(hidden_states_ptr + h_offs, mask=mask, other=0.0).to(tl.float32)

        # Load lm_head weight row (row-major: lm_head[vocab, h] = lm_head_ptr[vocab * stride + h])
        w = tl.load(lm_head_ptr + vocab_idx * lm_head_stride + h_offs, mask=mask, other=0.0).to(tl.float32)

        acc += tl.sum(h * w)

    tl.store(output_logits_ptr + token_idx, acc)


@triton.jit
def _fused_flash_head_kernel(
    # Inputs
    hidden_states_ptr,      # [hidden_size]
    lm_head_ptr,            # [vocab_size, hidden_size]
    centroids_ptr,          # [hidden_size, n_clusters]
    vocab_maps_ptr,         # [n_clusters, max_cluster_size]
    # Outputs
    output_token_ptr,       # [num_programs] - local best token per program
    output_logit_ptr,       # [num_programs] - local best logit per program
    # Sizes
    hidden_size,
    n_clusters,
    max_cluster_size,
    n_probes,
    lm_head_stride,
    # Scratch space
    cluster_sims_ptr,       # [n_clusters] - pre-computed cluster similarities
    top_clusters_ptr,       # [n_probes] - top cluster indices
    # Constants
    BLOCK_H: tl.constexpr,
    TOKENS_PER_PROGRAM: tl.constexpr,
):
    """
    Fully fused FlashHead kernel.

    This kernel is called AFTER cluster similarities are computed and top-k clusters
    are selected (those operations are small and fast in PyTorch).

    Each program:
    1. Processes TOKENS_PER_PROGRAM tokens from the selected clusters
    2. Computes logits for those tokens
    3. Tracks local max logit and token ID

    A final reduction step in Python finds the global max.
    """
    pid = tl.program_id(0)

    # Each program handles a range of the total tokens
    total_tokens = n_probes * max_cluster_size
    start_idx = pid * TOKENS_PER_PROGRAM

    # Track local max
    local_max_logit = float('-inf')
    local_max_token = -1

    for i in range(TOKENS_PER_PROGRAM):
        global_idx = start_idx + i

        if global_idx >= total_tokens:
            break

        # Determine which cluster and position within cluster
        cluster_local_idx = global_idx // max_cluster_size
        pos_in_cluster = global_idx % max_cluster_size

        if cluster_local_idx >= n_probes:
            break

        # Get actual cluster index from top_clusters
        cluster_idx = tl.load(top_clusters_ptr + cluster_local_idx)

        # Get token ID from vocab_maps
        token_id = tl.load(vocab_maps_ptr + cluster_idx * max_cluster_size + pos_in_cluster)

        # Skip padding tokens
        if token_id < 0:
            continue

        # Compute logit: dot(hidden_states, lm_head[token_id])
        logit = tl.zeros([1], dtype=tl.float32)

        for h_start in range(0, hidden_size, BLOCK_H):
            h_offs = h_start + tl.arange(0, BLOCK_H)
            mask = h_offs < hidden_size

            h = tl.load(hidden_states_ptr + h_offs, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(lm_head_ptr + token_id * lm_head_stride + h_offs, mask=mask, other=0.0).to(tl.float32)

            logit += tl.sum(h * w)

        # Update local max
        logit_scalar = logit  # tl.sum returns scalar
        if logit_scalar > local_max_logit:
            local_max_logit = logit_scalar
            local_max_token = token_id

    # Store local results for reduction
    tl.store(output_logit_ptr + pid, local_max_logit)
    tl.store(output_token_ptr + pid, local_max_token)


class TritonFlashHead(torch.nn.Module):
    """
    Triton-accelerated FlashHead module.

    This implementation uses fused Triton kernels to avoid the memory allocation
    overhead of the pure PyTorch implementation.
    """

    def __init__(
        self,
        lm_head_weight: torch.Tensor,
        centroids: torch.Tensor,
        vocab_maps_tensor: torch.Tensor,
        n_probes: Optional[int] = None,
    ):
        super().__init__()

        self.vocab_size = lm_head_weight.shape[0]
        self.hidden_size = lm_head_weight.shape[1]
        self.n_clusters = centroids.shape[1]
        self.max_cluster_size = vocab_maps_tensor.shape[1]

        if n_probes is None:
            n_probes = max(1, self.n_clusters // 16)
        self.n_probes = n_probes

        # Store lm_head weight (row-major, vocab_size x hidden_size)
        self.register_buffer("lm_head_weight", lm_head_weight.contiguous())

        # Store centroids (hidden_size x n_clusters for column access)
        # Normalize centroids
        centroids_normed = centroids / (centroids.norm(dim=0, keepdim=True) + 1e-8)
        self.register_buffer("centroids", centroids_normed.contiguous())

        # Store vocab maps (n_clusters x max_cluster_size)
        self.register_buffer("vocab_maps", vocab_maps_tensor.contiguous())

        # Pre-allocate buffers
        self.register_buffer(
            "cluster_similarities",
            torch.zeros(self.n_clusters, dtype=torch.float32, device=lm_head_weight.device)
        )

        # Determine optimal block sizes
        self.BLOCK_H = min(128, triton.next_power_of_2(self.hidden_size))
        self.TOKENS_PER_PROGRAM = 32

        # Calculate number of programs needed for fused kernel
        total_tokens = self.n_probes * self.max_cluster_size
        self.num_programs = triton.cdiv(total_tokens, self.TOKENS_PER_PROGRAM)

        # Pre-allocate output buffers for fused kernel
        self.register_buffer(
            "program_logits",
            torch.zeros(self.num_programs, dtype=torch.float32, device=lm_head_weight.device)
        )
        self.register_buffer(
            "program_tokens",
            torch.zeros(self.num_programs, dtype=torch.int64, device=lm_head_weight.device)
        )

    def _compute_cluster_similarities(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute similarities between hidden_states and all cluster centroids."""
        # hidden_states: [hidden_size] or [1, hidden_size]
        hidden_flat = hidden_states.view(-1).contiguous()

        grid = (self.n_clusters,)

        _cluster_similarities_kernel[grid](
            hidden_flat,
            self.centroids,
            self.cluster_similarities,
            self.hidden_size,
            self.n_clusters,
            BLOCK_H=self.BLOCK_H,
        )

        return self.cluster_similarities

    def _get_top_clusters(self, similarities: torch.Tensor) -> torch.Tensor:
        """Get top-k cluster indices based on similarities."""
        _, top_indices = torch.topk(similarities, k=self.n_probes, dim=-1)
        return top_indices.to(torch.int64)

    def get_next_token(
        self,
        hidden_states: torch.Tensor,
        return_logprob: bool = False,
    ) -> torch.Tensor:
        """
        Get the next token using FlashHead.

        Args:
            hidden_states: [1, 1, hidden_size] or [1, hidden_size]
            return_logprob: If True, also return approximate logprob

        Returns:
            Token ID tensor of shape [1, 1]
            If return_logprob=True, returns (token_id, logprob) tuple
        """
        # Flatten hidden states
        hidden_flat = hidden_states.view(-1).contiguous()

        # Step 1: Compute cluster similarities using Triton
        self._compute_cluster_similarities(hidden_flat)

        # Step 2: Get top-k clusters (small operation, PyTorch is fine)
        top_clusters = self._get_top_clusters(self.cluster_similarities)

        # Step 3: Compute logits for tokens in selected clusters using fused kernel
        grid = (self.num_programs,)

        _fused_flash_head_kernel[grid](
            hidden_flat,
            self.lm_head_weight,
            self.centroids,
            self.vocab_maps,
            self.program_tokens,
            self.program_logits,
            self.hidden_size,
            self.n_clusters,
            self.max_cluster_size,
            self.n_probes,
            self.hidden_size,  # lm_head_stride
            self.cluster_similarities,
            top_clusters,
            BLOCK_H=self.BLOCK_H,
            TOKENS_PER_PROGRAM=self.TOKENS_PER_PROGRAM,
        )

        # Step 4: Final reduction to find global max
        best_idx = self.program_logits.argmax()
        best_token = self.program_tokens[best_idx]
        best_logit = self.program_logits[best_idx]

        result = best_token.view(1, 1)

        if return_logprob:
            # Approximate logprob using cluster probability
            cluster_probs = torch.softmax(self.cluster_similarities, dim=-1)
            top_cluster_prob = cluster_probs[top_clusters].sum()

            # Compute log probability of selected token within its cluster
            # This is approximate - we'd need all cluster logits for exact
            approx_logprob = best_logit - torch.logsumexp(self.program_logits, dim=0)
            approx_logprob = approx_logprob + torch.log(top_cluster_prob + 1e-10)

            return result, approx_logprob.view(1)

        return result


class TritonFlashHeadV2(torch.nn.Module):
    """
    Alternative Triton FlashHead using separate kernels for better occupancy.

    This version:
    1. Uses Triton for cluster similarities
    2. Uses PyTorch topk (very fast for small k)
    3. Uses Triton for logit computation (main bottleneck)
    4. Uses PyTorch argmax (fast for small tensor)
    """

    def __init__(
        self,
        lm_head_weight: torch.Tensor,
        centroids: torch.Tensor,
        vocab_maps_tensor: torch.Tensor,
        n_probes: Optional[int] = None,
    ):
        super().__init__()

        self.vocab_size = lm_head_weight.shape[0]
        self.hidden_size = lm_head_weight.shape[1]
        self.n_clusters = centroids.shape[1]
        self.max_cluster_size = vocab_maps_tensor.shape[1]

        if n_probes is None:
            n_probes = max(1, self.n_clusters // 16)
        self.n_probes = n_probes

        # Store tensors
        self.register_buffer("lm_head_weight", lm_head_weight.contiguous())

        centroids_normed = centroids / (centroids.norm(dim=0, keepdim=True) + 1e-8)
        self.register_buffer("centroids", centroids_normed.contiguous())
        self.register_buffer("vocab_maps", vocab_maps_tensor.to(torch.int64).contiguous())

        # Pre-allocate buffers
        self.num_tokens_per_call = self.n_probes * self.max_cluster_size
        self.register_buffer(
            "token_logits",
            torch.zeros(self.num_tokens_per_call, dtype=torch.float32, device=lm_head_weight.device)
        )
        self.register_buffer(
            "token_indices",
            torch.zeros(self.num_tokens_per_call, dtype=torch.int64, device=lm_head_weight.device)
        )

        self.BLOCK_H = min(128, triton.next_power_of_2(self.hidden_size))

    def get_next_token(
        self,
        hidden_states: torch.Tensor,
        return_logprob: bool = False,
    ) -> torch.Tensor:
        """Get next token using optimized Triton kernels."""
        hidden_flat = hidden_states.view(-1).contiguous()

        # Step 1: Cluster similarities (Triton)
        cluster_sims = torch.empty(
            self.n_clusters, dtype=torch.float32, device=hidden_flat.device
        )

        grid = (self.n_clusters,)
        _cluster_similarities_kernel[grid](
            hidden_flat,
            self.centroids,
            cluster_sims,
            self.hidden_size,
            self.n_clusters,
            BLOCK_H=self.BLOCK_H,
        )

        # Step 2: Top-k clusters (PyTorch - fast for small k)
        _, top_clusters = torch.topk(cluster_sims, k=self.n_probes)

        # Step 3: Gather token indices from selected clusters
        # This creates a flattened list of all token IDs in selected clusters
        selected_maps = self.vocab_maps[top_clusters]  # [n_probes, max_cluster_size]
        token_indices = selected_maps.view(-1)  # [n_probes * max_cluster_size]
        num_tokens = token_indices.shape[0]

        # Step 4: Compute logits for selected tokens (Triton)
        grid = (num_tokens,)
        _compute_selected_logits_kernel[grid](
            hidden_flat,
            self.lm_head_weight,
            token_indices,
            self.token_logits,
            self.hidden_size,
            self.hidden_size,  # lm_head_stride
            num_tokens,
            BLOCK_H=self.BLOCK_H,
        )

        # Step 5: Find argmax (PyTorch - fast for small tensor)
        logits_view = self.token_logits[:num_tokens]
        best_local_idx = logits_view.argmax()
        best_token_id = token_indices[best_local_idx]
        best_logit = logits_view[best_local_idx]

        result = best_token_id.view(1, 1)

        if return_logprob:
            cluster_probs = torch.softmax(cluster_sims, dim=-1)
            top_cluster_prob = cluster_probs[top_clusters].sum()
            approx_logprob = best_logit - torch.logsumexp(logits_view, dim=0)
            approx_logprob = approx_logprob + torch.log(top_cluster_prob + 1e-10)
            return result, approx_logprob.view(1)

        return result


def create_triton_flash_head(
    lm_head_weight: torch.Tensor,
    centroids: torch.Tensor,
    vocab_maps_tensor: torch.Tensor,
    n_probes: Optional[int] = None,
    version: str = "v2",
) -> torch.nn.Module:
    """
    Create a Triton-accelerated FlashHead module.

    Args:
        lm_head_weight: [vocab_size, hidden_size] weight tensor
        centroids: [hidden_size, n_clusters] cluster centroids
        vocab_maps_tensor: [n_clusters, max_cluster_size] token indices per cluster
        n_probes: Number of clusters to probe (default: n_clusters // 16)
        version: "v1" for fully fused, "v2" for separate kernels (recommended)

    Returns:
        TritonFlashHead module
    """
    if version == "v1":
        return TritonFlashHead(lm_head_weight, centroids, vocab_maps_tensor, n_probes)
    else:
        return TritonFlashHeadV2(lm_head_weight, centroids, vocab_maps_tensor, n_probes)


