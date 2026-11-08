"""Small, deterministic RQ-VAE for item Semantic IDs.

The implementation deliberately keeps item encoding separate from the LLM GPU
process.  Hashing features are built on CPU and the RQ-VAE checkpoint is tiny,
so this stage does not compete with QLoRA/vLLM memory on a 24GB card.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class RQConfig:
    """Conservative defaults for a CPU-friendly item encoder."""

    feature_dim: int = 256
    hidden_dim: int = 128
    latent_dim: int = 32
    num_levels: int = 3
    codebook_size: int = 16
    epochs: int = 80
    batch_size: int = 32
    learning_rate: float = 1e-3
    commitment_weight: float = 0.25
    codebook_weight: float = 1.0
    reconstruction_weight: float = 1.0
    collision_refresh_attempts: int = 4
    collision_suffix_bytes: int = 4
    seed: int = 42
    device: str = "cpu"
    max_items: int | None = None


def compact_base_code(codes: Iterable[int]) -> str:
    """Format one residual-code path without the public ``S`` prefix."""

    return "".join(f"{int(code):02d}" for code in codes)


def codebook_diagnostics(indices: torch.Tensor, codebook_size: int) -> dict[str, Any]:
    """Return JSON-serializable utilization and raw path-collision metrics."""

    if indices.ndim != 2:
        raise ValueError("indices must have shape [items, levels]")
    rows = indices.detach().cpu().to(torch.long)
    num_items, num_levels = rows.shape
    used_codes: list[int] = []
    utilization: list[float] = []
    perplexity: list[float] = []
    for level in range(num_levels):
        counts = torch.bincount(rows[:, level], minlength=int(codebook_size)).to(torch.float64)
        used = int((counts > 0).sum().item())
        probabilities = counts[counts > 0] / max(1.0, float(counts.sum().item()))
        entropy = float((-(probabilities * probabilities.log()).sum()).item()) if len(probabilities) else 0.0
        used_codes.append(used)
        utilization.append(used / max(1, int(codebook_size)))
        perplexity.append(float(math.exp(entropy)))

    path_counts: dict[tuple[int, ...], int] = {}
    for row in rows.tolist():
        path = tuple(int(code) for code in row)
        path_counts[path] = path_counts.get(path, 0) + 1
    collided_sizes = [count for count in path_counts.values() if count > 1]
    collision_items = sum(count - 1 for count in collided_sizes)
    return {
        "num_items": int(num_items),
        "unique_base_codes": len(path_counts),
        "collision_groups": len(collided_sizes),
        "collision_items": int(collision_items),
        "collision_rate": float(collision_items / max(1, num_items)),
        "max_collision_group_size": max(collided_sizes, default=1),
        "used_codes_per_level": used_codes,
        "codebook_utilization_per_level": utilization,
        "codebook_perplexity_per_level": perplexity,
    }


def resolve_base_code_collisions(
    base_codes: dict[str, str],
    suffix_bytes: int = 4,
) -> tuple[dict[str, str], dict[str, str]]:
    """Resolve raw RQ path collisions with stable, item-local hash suffixes.

    The suffix depends only on the original item ID, unlike an ordinal X00/X01
    suffix whose assignment can change when another colliding item is inserted.
    Hash collisions are handled by deterministically increasing the digest size.
    """

    groups: dict[str, list[str]] = {}
    for item_id, code in base_codes.items():
        groups.setdefault(str(code), []).append(str(item_id))
    item_to_sid: dict[str, str] = {}
    sid_to_item: dict[str, str] = {}
    minimum_bytes = max(2, int(suffix_bytes))
    for code, group in sorted(groups.items()):
        collided = len(group) > 1
        for item_id in sorted(group):
            if not collided:
                sid = f"S{code}"
            else:
                digest_bytes = minimum_bytes
                while True:
                    suffix = hashlib.blake2b(item_id.encode("utf-8"), digest_size=digest_bytes).hexdigest().upper()
                    sid = f"S{code}X{suffix}"
                    if sid not in sid_to_item or sid_to_item[sid] == item_id:
                        break
                    digest_bytes += 1
            item_to_sid[item_id] = sid
            sid_to_item[sid] = item_id
    return item_to_sid, sid_to_item


def _stable_hash(token: str, seed: int = 17) -> int:
    digest = hashlib.blake2b(f"{seed}:{token}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    # Character ngrams make short Amazon titles less brittle while retaining a
    # fixed-size, dependency-free representation.
    ngrams: list[str] = []
    for word in words:
        padded = f"^{word}$"
        ngrams.extend(padded[i : i + 3] for i in range(max(0, len(padded) - 2)))
    return words + ngrams


def build_hashed_text_features(
    item_meta: dict[str, dict[str, Any]],
    feature_dim: int = 256,
    seed: int = 17,
) -> tuple[list[str], np.ndarray]:
    """Build L2-normalized TF-IDF-like hashing features entirely on CPU."""

    item_ids = sorted(str(item_id) for item_id in item_meta)
    matrix = np.zeros((len(item_ids), int(feature_dim)), dtype=np.float32)
    document_frequency = np.zeros(int(feature_dim), dtype=np.float32)
    for row, item_id in enumerate(item_ids):
        meta = item_meta[item_id]
        text = " ".join(
            [str(meta.get("title", "")), str(meta.get("category", meta.get("categories", "")))]
        )
        bucket_counts: dict[int, int] = {}
        for token in _tokens(text):
            bucket = _stable_hash(token, seed) % int(feature_dim)
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        for bucket, count in bucket_counts.items():
            matrix[row, bucket] = 1.0 + math.log(float(count))
            document_frequency[bucket] += 1.0
    if len(item_ids):
        idf = np.log((1.0 + len(item_ids)) / (1.0 + document_frequency)) + 1.0
        matrix *= idf[None, :]
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix /= np.maximum(norms, 1e-8)
    return item_ids, matrix


class RQEmbeddingModel(nn.Module):
    """Residual vector quantizer with a compact MLP encoder/decoder."""

    def __init__(self, cfg: RQConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Sequential(
            nn.Linear(cfg.feature_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.feature_dim),
        )
        self.codebooks = nn.Parameter(
            torch.randn(cfg.num_levels, cfg.codebook_size, cfg.latent_dim) * 0.05
        )

    def quantize(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = z
        quantized = torch.zeros_like(z)
        indices: list[torch.Tensor] = []
        codebook_loss = z.new_zeros(())
        commitment_loss = z.new_zeros(())
        for level in range(self.cfg.num_levels):
            codebook = self.codebooks[level]
            distances = (
                residual.square().sum(dim=-1, keepdim=True)
                - 2.0 * residual @ codebook.t()
                + codebook.square().sum(dim=-1).unsqueeze(0)
            )
            index = distances.argmin(dim=-1)
            selected = codebook[index]
            quantized = quantized + selected
            codebook_loss = codebook_loss + (selected - residual.detach()).square().mean()
            commitment_loss = commitment_loss + (residual - selected.detach()).square().mean()
            residual = residual - selected.detach()
            indices.append(index)
        return quantized, torch.stack(indices, dim=1), codebook_loss, commitment_loss

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(features)
        quantized, indices, codebook_loss, commitment_loss = self.quantize(z)
        quantized_st = z + (quantized - z).detach()
        reconstruction = self.decoder(quantized_st)
        reconstruction_loss = (reconstruction - features).square().mean()
        loss = (
            self.cfg.reconstruction_weight * reconstruction_loss
            + self.cfg.codebook_weight * codebook_loss
            + self.cfg.commitment_weight * commitment_loss
        )
        return {
            "loss": loss,
            "reconstruction": reconstruction,
            "indices": indices,
            "reconstruction_loss": reconstruction_loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
        }

    @torch.no_grad()
    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.quantize(self.encoder(features))[1]


@torch.no_grad()
def _kmeans(points: torch.Tensor, clusters: int, seed: int, iterations: int = 30) -> torch.Tensor:
    """Small farthest-first k-means used to refresh collapsed codebooks."""

    if points.ndim != 2 or not len(points):
        raise ValueError("k-means expects a non-empty [items, dim] tensor")
    generator = torch.Generator(device=points.device)
    generator.manual_seed(int(seed))
    effective = min(int(clusters), len(points))
    first = int(torch.randint(len(points), (1,), generator=generator, device=points.device).item())
    centers = [points[first].clone()]
    minimum_distance = (points - centers[0]).square().sum(dim=1)
    for _ in range(1, effective):
        index = int(minimum_distance.argmax().item())
        centers.append(points[index].clone())
        minimum_distance = torch.minimum(
            minimum_distance,
            (points - centers[-1]).square().sum(dim=1),
        )
    center_tensor = torch.stack(centers)
    for _ in range(max(1, int(iterations))):
        distances = torch.cdist(points, center_tensor).square()
        assignments = distances.argmin(dim=1)
        updated = center_tensor.clone()
        for cluster in range(effective):
            members = points[assignments == cluster]
            if len(members):
                updated[cluster] = members.mean(dim=0)
        if torch.allclose(updated, center_tensor, atol=1e-6, rtol=0.0):
            center_tensor = updated
            break
        center_tensor = updated
    if effective < int(clusters):
        repeats = center_tensor[torch.arange(int(clusters) - effective, device=points.device) % effective]
        center_tensor = torch.cat([center_tensor, repeats], dim=0)
    return center_tensor


@torch.no_grad()
def refresh_residual_codebooks(
    model: RQEmbeddingModel,
    features: np.ndarray,
    seed: int,
) -> None:
    """Fit residual k-means codebooks after SGD to prevent codebook collapse.

    RQ-VAE datasets can contain far fewer items than a language-model dataset.
    An SGD-only codebook may collapse to one code while the decoder still lowers
    reconstruction loss. A deterministic residual k-means refresh restores code
    utilization without increasing GPU memory or adding dependencies.
    """

    device = next(model.parameters()).device
    tensor = torch.from_numpy(features.astype(np.float32, copy=False)).to(device)
    residual = model.encoder(tensor)
    for level in range(model.cfg.num_levels):
        centers = _kmeans(residual, model.cfg.codebook_size, seed + level)
        model.codebooks[level].copy_(centers)
        assignments = torch.cdist(residual, centers).square().argmin(dim=1)
        residual = residual - centers[assignments]


def train_rqvae(
    features: np.ndarray,
    cfg: RQConfig,
) -> tuple[RQEmbeddingModel, dict[str, Any]]:
    """Train with bounded CPU/GPU batches and return loss statistics."""

    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device_name = str(cfg.device).lower()
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    model = RQEmbeddingModel(cfg).to(device)
    # Cap batches even if a user accidentally supplies a huge value.
    batch_size = max(1, min(int(cfg.batch_size), 128))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features.astype(np.float32, copy=False))),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate), weight_decay=1e-4)
    last: dict[str, float] = {}
    model.train()
    for _epoch in range(max(1, int(cfg.epochs))):
        totals = {"loss": 0.0, "reconstruction_loss": 0.0, "codebook_loss": 0.0, "commitment_loss": 0.0}
        count = 0
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=False)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            outputs["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            size = int(batch.shape[0])
            count += size
            for key in totals:
                totals[key] += float(outputs[key].detach().cpu()) * size
        last = {key: value / max(1, count) for key, value in totals.items()}
    model.eval()
    feature_tensor = torch.from_numpy(features.astype(np.float32, copy=False)).to(device)
    attempts: list[dict[str, Any]] = []
    best_codebooks: torch.Tensor | None = None
    best_score: tuple[int, float] | None = None
    best_attempt = 0
    refresh_attempts = max(1, int(cfg.collision_refresh_attempts))
    for attempt in range(refresh_attempts):
        attempt_seed = int(cfg.seed) + attempt * 1009
        refresh_residual_codebooks(model, features, attempt_seed)
        with torch.no_grad():
            attempt_output = model(feature_tensor)
        diagnostic = codebook_diagnostics(attempt_output["indices"], cfg.codebook_size)
        reconstruction = float(attempt_output["reconstruction_loss"].detach().cpu())
        attempts.append(
            {
                "attempt": attempt,
                "seed": attempt_seed,
                "collision_items": diagnostic["collision_items"],
                "unique_base_codes": diagnostic["unique_base_codes"],
                "reconstruction_loss": reconstruction,
            }
        )
        score = (int(diagnostic["collision_items"]), reconstruction)
        if best_score is None or score < best_score:
            best_score = score
            best_attempt = attempt
            best_codebooks = model.codebooks.detach().clone()
        # Collision avoidance is the primary objective. Preserve the first
        # deterministic collision-free assignment instead of changing every
        # public Semantic ID merely for a tiny reconstruction improvement.
        if int(diagnostic["collision_items"]) == 0:
            break
    if best_codebooks is None:
        raise RuntimeError("collision-aware codebook refresh produced no candidate")
    with torch.no_grad():
        model.codebooks.copy_(best_codebooks)
        refreshed = model(feature_tensor)
    diagnostic = codebook_diagnostics(refreshed["indices"], cfg.codebook_size)
    last.update(
        {
            "refreshed_loss": float(refreshed["loss"].detach().cpu()),
            "refreshed_reconstruction_loss": float(refreshed["reconstruction_loss"].detach().cpu()),
            **diagnostic,
            "collision_refresh_attempts_requested": refresh_attempts,
            "collision_refresh_attempts_executed": len(attempts),
            "selected_collision_refresh_attempt": best_attempt,
            "collision_refresh_trials": attempts,
        }
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model.cpu(), last


def encode_semantic_ids(
    model: RQEmbeddingModel,
    item_ids: list[str],
    features: np.ndarray,
) -> tuple[dict[str, str], dict[str, str]]:
    """Encode unique, human-readable Semantic IDs and resolve collisions."""

    with torch.no_grad():
        indices = model.encode(torch.from_numpy(features.astype(np.float32, copy=False))).cpu().tolist()
    base_codes: dict[str, str] = {}
    for item_id, row in zip(item_ids, indices):
        # Compact codes substantially reduce SFT target and vLLM KV-cache
        # length compared with verbose strings such as SID_L0_C03_L1_C07.
        base_codes[item_id] = compact_base_code(row)
    return resolve_base_code_collisions(base_codes, suffix_bytes=model.cfg.collision_suffix_bytes)


def save_rqvae_checkpoint(
    path: str | Path,
    model: RQEmbeddingModel,
    cfg: RQConfig,
    item_to_sid: dict[str, str],
    sid_to_item: dict[str, str],
    stats: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(cfg),
            "state_dict": model.state_dict(),
            "item_to_sid": item_to_sid,
            "sid_to_item": sid_to_item,
            "stats": stats,
        },
        path,
    )


def load_semantic_mapping(path: str | Path) -> dict[str, Any]:
    """Load JSON mapping generated by train_rqvae."""

    with Path(path).open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    if not isinstance(mapping, dict) or "item_to_sid" not in mapping or "sid_to_item" not in mapping:
        raise ValueError(f"Invalid Semantic ID mapping: {path}")
    return mapping
