#!/usr/bin/env python3
"""
expert_pager.py — ExpertPager : pagination des experts sur disque (mémoire virtuelle MoE)
==========================================================================================

La MMU du MoE : les experts sont des PAGES, le hash est la TABLE DES PAGES.
  - E experts totaux (pages sur disque, safetensors, illimités = SSD).
  - S slots résidents en RAM (modules FusedSwiGLU time-sharés, S << E).
  - ensure() = défaut de page : charge si absent. prefetch() = readahead exact.
  - LRU + hot-pinning (experts chauds épinglés) + writeback des pages dirty.
  - Chaque page contient {expert.*, tc_slice} (sa tranche to_channels) → le
    forward paginé est mathématiquement IDENTIQUE au dense (jamais de D→E×D).

Masquage de la latence I/O (réponse à la question du page-fault) :
  1. HASH-AHEAD PREFETCH (exact, mode token) : le hash du batch N+1 est calculé
     pendant le compute du batch N → les pages arrivent AVANT le défaut.
     Condition de masquage total : T_load(pages N+1) ≤ T_compute(batch N).
     Double buffering : compute(N) ∥ load(N+1).
  2. HOT-PINNING : top-N usage-EMA épinglés → les fautes ne touchent que la queue froide.
  3. PRÉDICTEUR WORKING-SET (mode LSH / couches profondes) : union des pages des
     derniers batches + hint token-hash → précharge un sur-ensemble (recall mesuré).
  4. PAGES QUANTIFIÉES int8 sur disque (÷4 octets) → ÷4 temps I/O, déquant au chargement.
  5. JAMAIS DE STALL DUR : vagues (waves) si pages_uniques > slots ; fallback joker
     compté (page non chargée à temps → contribution 0 sur ses tokens, STAT, pas crash).
  Ironie CPU : le compute est si lent que depth-1 suffit toujours ; la sim à latence
  artificielle prouve que le mécanisme transfère à un compute plus rapide.

Usage :
    PYTHONPATH=source:. python3 expert_pager.py --self-test   # parité + anti-stall (CPU)
"""

import os
import sys
import time
import json
import threading
from pathlib import Path
from dataclasses import dataclass, field
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "source"))
sys.path.insert(0, str(HERE))

from cognet_1b_optimized import RMSNorm, FusedSwiGLU  # noqa: E402
from cognet_moe import CogNetMoE1B  # noqa: E402
from hash_moe import LSHHasher, token_hash_experts  # noqa: E402

try:
    from safetensors.torch import save_file as _st_save, load_file as _st_load
    _HAS_ST = True
except ImportError:
    _HAS_ST = False


# ═══════════════════════════════════════════════════════════════════════
# Store : 1 fichier par page (+ quant int8 optionnelle)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PagerConfig:
    resident_pages: int = 8    # capacité RAM totale (pages), tous blocs confondus
    n_hot_pinned: int = 0      # top-N usage-EMA épinglés (jamais évincés)
    num_loaders: int = 2       # threads de chargement async
    quantize: Optional[str] = None  # None | 'int8'
    io_delay_ms: float = 0.0   # SIMULATION latence SSD par faute (valide le mécanisme)
    prefetch_depth: int = 1    # batches d'avance (hash-ahead)
    store_dir: str = "./pager_store"


def _quantize_state(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Seules les MATRICES (dim>=2, le gros du volume) passent en int8 ;
    # vecteurs/scalaires (biais, normes, compteurs Hebbiens) restent exacts.
    out = {}
    for k, v in sd.items():
        if v.dim() < 2 or not v.is_floating_point():
            out[k] = v
            continue
        v = v.float()
        s = v.abs().max().clamp_min(1e-8) / 127.0
        out[k] = (v / s).round().clamp(-128, 127).to(torch.int8)
        out[k + ".scale"] = torch.tensor(float(s))
    return out


def _dequantize_state(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        if k.endswith(".scale"):
            continue
        if k + ".scale" in sd:
            out[k] = v.float() * float(sd[k + ".scale"].item())
        else:
            out[k] = v  # stocké exact (vecteur/scalaire)
    return out


class ExpertPageStore:
    """Stockage disque : 1 fichier par page (bloc, expert). Backend safetensors."""

    def __init__(self, store_dir: str, quantize: Optional[str] = None):
        assert _HAS_ST, "safetensors requis (pip install safetensors)"
        assert quantize in (None, "int8")
        self.dir = Path(store_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.quantize = quantize
        self.lock = threading.Lock()

    def path(self, block: int, expert: int) -> Path:
        return self.dir / f"blk{block:02d}_exp{expert:03d}.safetensors"

    def save_page(self, block: int, expert: int, sd: Dict[str, torch.Tensor]):
        payload = {k: v.detach().cpu().contiguous() for k, v in sd.items()}
        if self.quantize == "int8":
            payload = _quantize_state(payload)
        with self.lock:
            _st_save(payload, str(self.path(block, expert)))

    def load_page(self, block: int, expert: int) -> Dict[str, torch.Tensor]:
        with self.lock:
            sd = _st_load(str(self.path(block, expert)), device="cpu")
        if self.quantize == "int8":
            sd = _dequantize_state(sd)
        return sd

    def disk_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.dir.glob("*.safetensors"))


# ═══════════════════════════════════════════════════════════════════════
# Pager : LRU + pin + async + writeback
# ═══════════════════════════════════════════════════════════════════════

PageId = Tuple[int, int]  # (bloc, expert)


class ExpertPager:
    """
    Gère les slots résidents de tous les routers paginés du modèle.
    Les tenseurs des slots RESTENT les mêmes objets (copie in-place) →
    l'optimizer survit au paging sans chirurgie.
    """

    def __init__(self, cfg: PagerConfig):
        self.cfg = cfg
        self.store = ExpertPageStore(cfg.store_dir, cfg.quantize)
        self.pool = ThreadPoolExecutor(max_workers=cfg.num_loaders,
                                       thread_name_prefix="pager-load")
        self.routers: Dict[int, "PagedHashExpertRouter"] = {}
        self.resident: "OrderedDict[PageId, int]" = OrderedDict()  # page -> slot, LRU (fin=MRU)
        self.page_of_slot: Dict[Tuple[int, int], PageId] = {}      # (bloc, slot) -> page
        self.pinned: Set[PageId] = set()
        self.dirty: Set[PageId] = set()
        self.in_flight: Dict[PageId, Future] = {}
        self.usage_ema: Dict[PageId, float] = {}
        self.lock = threading.RLock()
        self.stats = {"hits": 0, "faults_sync": 0, "faults_covered": 0,
                      "faults_waited": 0, "evictions": 0, "writebacks": 0,
                      "joker_fallbacks": 0, "stall_ms": 0.0}

    # ── Enregistrement ─────────────────────────────────────────────
    def register_router(self, block_idx: int, router: "PagedHashExpertRouter"):
        self.routers[block_idx] = router

    # ── Snapshot initial : tout le modèle → disque ─────────────────
    def snapshot_router(self, block_idx: int, router):
        router.snapshot_pages(self)

    # ── Chemin chaud ───────────────────────────────────────────────
    def _sim_io(self):
        if self.cfg.io_delay_ms > 0:
            time.sleep(self.cfg.io_delay_ms / 1000.0)

    def _load_sync(self, page: PageId) -> Dict[str, torch.Tensor]:
        self._sim_io()  # latence SSD simulée (mécanisme, pas la mesure brute)
        return self.store.load_page(*page)

    def _install(self, page: PageId, sd: Dict[str, torch.Tensor], slot: int):
        """Copie page → slot (délégué au router : dense, associatif, ...)."""
        self.routers[page[0]].install_page(page, sd, slot)

    def _free_slot(self, block: int, n_slots: int) -> Optional[int]:
        taken = {s for (bb, s), _ in self.page_of_slot.items() if bb == block}
        for s in range(n_slots):
            if s not in taken:
                return s
        return None

    def _evict_lru(self, block: int):
        """Évince la page LRU non-pinnée du bloc (writeback si dirty)."""
        for page in list(self.resident.keys()):
            if page[0] != block or page in self.pinned:
                continue
            slot = self.resident.pop(page)
            del self.page_of_slot[(block, slot)]
            if page in self.dirty:
                self._writeback(page, block, slot)
            self.stats["evictions"] += 1
            return slot
        return None

    def _writeback(self, page: PageId, block: int, slot: int):
        sd = self.routers[block].read_slot(slot)
        self.store.save_page(*page, sd)
        self.dirty.discard(page)
        self.stats["writebacks"] += 1

    def ensure(self, block: int, pages: List[PageId]) -> Dict[PageId, int]:
        """
        Garantit la résidence (faute synchrone si besoin) → {page: slot}.
        Appelé AVANT le dispatch de chaque vague.
        """
        router = self.routers[block]
        S = router.n_slots
        slot_map: Dict[PageId, int] = {}
        # Résidentes d'abord : évite d'évincer (LRU) les pages qu'on va toucher
        # juste après — pathologie du balayage séquentiel (hits=0 sinon).
        pages = sorted(pages, key=lambda p: p not in self.resident)
        with self.lock:
            for p in pages:
                self.usage_ema[p] = 0.9 * self.usage_ema.get(p, 0.0) + 0.1
                if p in self.resident:
                    self.stats["hits"] += 1
                    self.resident.move_to_end(p)
                    slot_map[p] = self.resident[p]
                    continue
                # Faute.
                t0 = time.time()
                if p in self.in_flight:
                    fut = self.in_flight.pop(p)
                    # release lock pendant l'attente (évite deadlock avec loader)
                    self.lock.release()
                    try:
                        sd = fut.result()
                    finally:
                        self.lock.acquire()
                    self.stats["faults_covered"] += 1
                    if (time.time() - t0) * 1000 > 1.0:
                        self.stats["faults_waited"] += 1
                else:
                    self.lock.release()
                    try:
                        sd = self._load_sync(p)
                    finally:
                        self.lock.acquire()
                    self.stats["faults_sync"] += 1
                self.stats["stall_ms"] += (time.time() - t0) * 1000
                slot = self._free_slot(block, S)
                if slot is None:
                    slot = self._evict_lru(block)
                if slot is None:  # tout est pinné → joker (pas de stall dur)
                    self.stats["joker_fallbacks"] += 1
                    continue
                self._install(p, sd, slot)
                self.resident[p] = slot
                self.page_of_slot[(block, slot)] = p
                slot_map[p] = slot
        return slot_map

    def prefetch(self, pages: List[PageId]):
        """Soumet les pages manquantes au pool async (jamais bloquant)."""
        with self.lock:
            for p in pages:
                if p in self.resident:
                    self.resident.move_to_end(p)  # hint-hit = usage prédit → MRU
                    continue
                if p in self.in_flight:
                    continue
                fut = self.pool.submit(self._load_sync, p)
                self.in_flight[p] = fut

    def _drain(self, pages: Optional[List[PageId]] = None, timeout: float = 30.0):
        """Attend les chargements en vol (liste donnée, sinon TOUS) et les installe.
        Faute COUVERTE : le thread compute attend un transfert déjà lancé, jamais
        un aller-retour disque complet depuis zéro. En production, drainer n'est
        presque jamais bloquant (prefetch soumis un batch à l'avance)."""
        if pages is None:
            items = list(self.in_flight.items())
            excl: FrozenSet[PageId] = frozenset()
        else:
            items = [(p, self.in_flight[p]) for p in pages if p in self.in_flight]
            excl = frozenset(pages)
        for p, fut in items:
            try:
                _t0 = time.perf_counter()
                data = fut.result(timeout=timeout)
                self.stats["stall_ms"] += (time.perf_counter() - _t0) * 1000.0
            except Exception:
                self.in_flight.pop(p, None)
                continue
            self.in_flight.pop(p, None)
            self.stats["faults_covered"] += 1
            b, e = p
            if p in self.resident:
                self.resident.move_to_end(p)
                continue
            router = self.routers.get(b)
            if router is None:
                continue
            slot = self._free_slot(b, router.n_slots)
            if slot is None:
                # Évince le plus vieux (les premiers du hint partent en premier ;
                # les derniers installés = prioritaires = survivent).
                slot = self._evict_lru(b)
                if slot is None:
                    self.stats["jokers"] += 1
                    continue
            self._install(p, data, slot)
            self.resident[p] = slot
            self.page_of_slot[(b, slot)] = p

    def _install_prefetched(self):
        """Installe les prefetches terminés (appel opportuniste, non bloquant)."""
        with self.lock:
            done = [p for p, f in self.in_flight.items() if f.done()]
            for p in done:
                try:
                    sd = self.in_flight.pop(p).result()
                except Exception:
                    self.in_flight.pop(p, None)
                    continue
                b = p[0]
                router = self.routers.get(b)
                if router is None or p in self.resident:
                    continue
                slot = self._free_slot(b, router.n_slots)
                if slot is None:
                    slot = self._evict_lru(b)
                if slot is None:
                    continue  # pas de place : le prefetch est jeté (sera re-demandé)
                self._install(p, sd, slot)
                self.resident[p] = slot
                self.page_of_slot[(b, slot)] = p
                self.stats["faults_covered"] += 1

    def mark_dirty(self, pages: List[PageId]):
        with self.lock:
            for p in pages:
                if p in self.resident:
                    self.dirty.add(p)

    def mark_all_resident_dirty(self):
        with self.lock:
            for p in list(self.resident.keys()):
                b = p[0]
                router = self.routers[b]
                frozen = router.frozen_pages is not None and p[1] in router.frozen_pages
                if not frozen:
                    self.dirty.add(p)

    def update_hot_pinning(self):
        """Épingle le top-N usage-EMA (mécanisme 2 : les fautes → queue froide)."""
        n = self.cfg.n_hot_pinned
        if n <= 0:
            return
        with self.lock:
            top = sorted(self.usage_ema.items(), key=lambda kv: -kv[1])[:n]
            self.pinned = {p for p, _ in top}
            # Installe immédiatement les pinnées manquantes.
            for p in list(self.pinned):
                if p not in self.resident:
                    b = p[0]
                    router = self.routers[b]
                    slot = self._free_slot(b, router.n_slots)
                    if slot is None:
                        slot = self._evict_lru(b)
                    if slot is not None:
                        self._install(p, self.store.load_page(*p), slot)
                        self.resident[p] = slot
                        self.page_of_slot[(b, slot)] = p

    def flush(self):
        """Attend les prefetches + writeback tous les dirty."""
        while True:
            with self.lock:
                pending = list(self.in_flight.items())
            if not pending:
                break
            for _, f in pending:
                f.result()
            self._install_prefetched()
        with self.lock:
            for p in list(self.dirty):
                if p in self.resident:
                    b = p[0]
                    self._writeback(p, b, self.resident[p])

    def summary(self) -> Dict:
        s = dict(self.stats)
        tot = s["hits"] + s["faults_sync"] + s["faults_covered"]
        s["hit_rate"] = s["hits"] / max(1, tot)
        s["covered_rate"] = s["faults_covered"] / max(1, tot)
        s["resident"] = len(self.resident)
        s["disk_bytes"] = self.store.disk_bytes()
        return s


# ═══════════════════════════════════════════════════════════════════════
# Dispatch paginé avec backward par recompute (style gradient-checkpointing)
# ═══════════════════════════════════════════════════════════════════════
# Problème : les slots sont time-sharés entre vagues ; un swap in-place pendant
# le forward invaliderait le graphe autograd des vagues précédentes.
# Solution : le forward ne trace rien ; au backward, on ré-installe les pages de
# chaque vague (via ensure, avec writeback si dirty) et on rejoue la vague AVEC
# grad. Coût = 1 forward supplémentaire (standard checkpointing). Correct.

# Registre du router actif (apply() n'accepte que des tenseurs : le forward
# synchrone snapshotte dans ctx, le backward relit ctx — thread-safe ici).
_ACTIVE_ROUTER = None


class _PagedDispatchFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_flat, w_full, assign_flat):
        # NOTE : forward d'une Function tourne SANS autograd (pas de graphe).
        global _ACTIVE_ROUTER
        router = _ACTIVE_ROUTER
        assert router is not None
        B_N, D = x_flat.shape
        K = w_full.shape[1]
        E = router.n_experts
        uniq = sorted(set(assign_flat.reshape(-1).tolist()))
        pinned_here = sum(1 for p in router.pager.pinned if p[0] == router.block_idx)
        wave = max(1, router.n_slots - pinned_here)
        waves = [uniq[i:i + wave] for i in range(0, len(uniq), wave)]
        combined = torch.zeros(B_N, D, device=x_flat.device, dtype=x_flat.dtype)
        for chunk in waves:
            slot_map = router.pager.ensure(router.block_idx,
                                           [(router.block_idx, e) for e in chunk])
            for e in chunk:
                page = (router.block_idx, e)
                if page not in slot_map:
                    continue
                s = slot_map[page]
                m = (assign_flat == e)
                tok = m.any(-1)
                if not bool(tok.any().item()):
                    continue
                wt = (w_full * m.float()).sum(-1)[tok]
                proj = x_flat[tok] @ router.slot_tc[s].T
                combined[tok] += wt.unsqueeze(-1).to(proj.dtype) * router.slots[s](proj)
        ctx.router = router
        ctx.waves = waves
        ctx.save_for_backward(x_flat, w_full, assign_flat)
        return combined

    @staticmethod
    def backward(ctx, grad_combined):
        router = ctx.router
        x_flat, w_full, assign_flat = ctx.saved_tensors
        # Rejoue chaque vague (ordre inverse) avec grad, pages ré-installées.
        with torch.enable_grad():
            for chunk in reversed(ctx.waves):
                slot_map = router.pager.ensure(
                    router.block_idx, [(router.block_idx, e) for e in chunk])
                for e in chunk:
                    page = (router.block_idx, e)
                    if page not in slot_map:
                        continue
                    s = slot_map[page]
                    m = (assign_flat == e)
                    tok = m.any(-1)
                    if not bool(tok.any().item()):
                        continue
                    wt = (w_full * m.float()).sum(-1)[tok]
                    # x_flat n'a pas requires_grad ici (saved) → recrée le lien.
                    xf = x_flat.detach().requires_grad_(x_flat.requires_grad)
                    proj = xf[tok] @ router.slot_tc[s].T
                    out_p = router.slots[s](proj)
                    g = (grad_combined[tok].to(out_p.dtype)
                         * wt.unsqueeze(-1).to(out_p.dtype))
                    torch.autograd.backward(out_p, g)
                    if x_flat.requires_grad and xf.grad is not None:
                        if getattr(ctx, "gx", None) is None:
                            ctx.gx = torch.zeros_like(x_flat)
                        ctx.gx += xf.grad
        gx = getattr(ctx, "gx", None)
        if gx is None:
            gx = torch.zeros_like(x_flat) if x_flat.requires_grad else None
        return gx, None, None


# ═══════════════════════════════════════════════════════════════════════
# Router paginé (S slots, E pages)
# ═══════════════════════════════════════════════════════════════════════

class PagedHashExpertRouter(nn.Module):
    """
    HashExpertRouter avec E experts sur disque et S slots RAM (S << E).
    forward → assign (hash, sans poids) → vagues de S pages → dispatch par slot.
    Mathématiquement identique au dense (page = expert + tranche to_channels).
    """

    def __init__(self, hidden_dim: int, ff_dim: int, n_experts: int, n_slots: int,
                 top_k: int = 2, dropout: float = 0.0, mode: str = "token",
                 n_bits: int = 24, seed: int = 1234,
                 block_idx: int = 0, pager: Optional[ExpertPager] = None,
                 frozen_pages: Optional[Set[int]] = None, salt_per_block: bool = False):
        super().__init__()
        assert mode in ("token", "lsh") and top_k <= n_slots
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.n_experts = n_experts
        self.num_channels = n_experts
        self.n_slots = n_slots
        self.top_k = top_k
        self.mode = mode
        self.block_idx = block_idx
        self.pager = pager
        self.frozen_pages = frozen = None  # résolu après (set via freeze_pages)
        self._frozen_set: Set[int] = set(frozen_pages or ())

        self.slots = nn.ModuleList([FusedSwiGLU(hidden_dim, ff_dim, dropout)
                                    for _ in range(n_slots)])
        self.slot_tc = nn.ParameterList([nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.02)
                                         for _ in range(n_slots)])
        self.norm = RMSNorm(hidden_dim)
        _jb = block_idx if salt_per_block else 0  # diversité inter-blocs (défaut OFF = parité dense)
        self.lsh = LSHHasher(hidden_dim, n_bits, seed + _jb) if mode == "lsh" else None
        self.salt = 0x9E3779B9 + _jb * 7919
        self._batch_token_ids: Optional[torch.Tensor] = None
        # Références denses (remplies par convert_to_paged, puis snapshotées).
        self.dense_experts: Optional[nn.ModuleList] = None
        self.dense_tc_weight: Optional[torch.Tensor] = None

    @property
    def frozen_pages(self):
        return self._frozen_set

    @frozen_pages.setter
    def frozen_pages(self, v):
        self._frozen_set = set(v or ())

    # ── Codec de page dense (substituable : cf. PagedAssocExpertRouter) ──
    def snapshot_pages(self, pager):
        """Vide le contenu dense initial → disque (source de vérité initiale)."""
        dense_tc = self.dense_tc_weight  # (E*D, D)
        D = self.hidden_dim
        for e in range(self.n_experts):
            sd = {f"expert.{k}": v.detach().cpu().clone()
                  for k, v in self.dense_experts[e].state_dict().items()}
            sd["tc_slice"] = dense_tc[e * D:(e + 1) * D].detach().cpu().clone()
            pager.store.save_page(self.block_idx, e, sd)

    def install_page(self, page, sd, slot: int):
        """Copie page → slot (in-place, REQUIRES_GRAD selon frozen)."""
        frozen = self.frozen_pages is not None and page[1] in self.frozen_pages
        self.slots[slot].load_state_dict(
            {k.split("expert.", 1)[1]: v for k, v in sd.items() if k.startswith("expert.")},
            strict=True)
        self.slot_tc[slot].data.copy_(sd["tc_slice"])
        for p in self.slots[slot].parameters():
            p.requires_grad = not frozen
        self.slot_tc[slot].requires_grad = not frozen

    def read_slot(self, slot: int):
        """Clone slot → dict (writeback)."""
        sd = {f"expert.{k}": v.detach().cpu().clone()
              for k, v in self.slots[slot].state_dict().items()}
        sd["tc_slice"] = self.slot_tc[slot].detach().cpu().clone()
        return sd

    def set_batch_token_ids(self, ids):
        self._batch_token_ids = ids

    @property
    def supports_id_prefetch(self) -> bool:
        return self.mode == "token"

    def assign_pages(self, x: torch.Tensor, token_ids=None) -> torch.Tensor:
        if self.mode == "token":
            ids = self._batch_token_ids if token_ids is None else token_ids
            assert ids is not None
            return token_hash_experts(ids.to(x.device), self.n_experts, self.top_k, self.salt)
        return self.lsh.experts(x.detach(), self.n_experts, self.top_k)

    def forward(self, x: torch.Tensor, token_ids=None):
        B, T, D = x.shape
        E, K = self.n_experts, self.top_k
        N = B * T
        assert self.pager is not None, "router paginé sans pager"
        with torch.no_grad():
            assign = self.assign_pages(x, token_ids)  # (B,T,K) pages
            w = torch.full(assign.shape, 1.0 / K, device=x.device, dtype=x.dtype)
        flat_idx = assign.reshape(N, K)
        flat_w = w.reshape(N, K)
        # Dispatch paginé (vagues + recompute au backward, cf. _PagedDispatchFn).
        global _ACTIVE_ROUTER
        _ACTIVE_ROUTER = self
        try:
            combined = _PagedDispatchFn.apply(x.reshape(N, D).contiguous(), flat_w, flat_idx)
        finally:
            _ACTIVE_ROUTER = None
        out = self.norm(combined.view(B, T, D))
        out = x + out
        f = torch.bincount(flat_idx.reshape(-1), minlength=E).float() / max(1, flat_idx.numel())
        with torch.no_grad():
            Pp = f.clamp_min(1e-8) / f.sum().clamp_min(1e-8)
            ent = -(Pp * Pp.log()).sum()
        stats = {
            "moe_aux_loss": torch.tensor(0.0, device=x.device),
            "moe_z_loss": torch.tensor(0.0, device=x.device),
            "moe_max_load": f.max().detach(), "moe_min_load": f.min().detach(),
            "moe_routing_entropy": ent.detach(), "moe_expert_usage": f.detach(),
            "routing_entropy": ent.detach(),
        }
        return out, stats


def convert_to_paged(model: CogNetMoE1B, n_slots: int, pager: ExpertPager,
                     mode: str = "token", seed: int = 1234,
                     frozen_pages: Optional[Set[int]] = None,
                     salt_per_block: bool = False) -> CogNetMoE1B:
    """Swap routers → paginés + snapshot disque. Le dense source doit être un
    HashExpertRouter (ou legacy : to_channels/experts copiés)."""
    for b, blk in enumerate(model.blocks):
        legacy = blk.cognitive_expert_router
        E = len(legacy.experts)
        ff = legacy.experts[0].w_down.weight.shape[1]
        D = legacy.hidden_dim
        r = PagedHashExpertRouter(D, ff, E, n_slots, legacy.top_k,
                                  legacy.experts[0].dropout.p, mode,
                                  block_idx=b, pager=pager,
                                  frozen_pages=set(frozen_pages or ()),
                                  salt_per_block=salt_per_block)
        # Références denses (copie, puis snapshot disque).
        r.dense_experts = nn.ModuleList([FusedSwiGLU(D, ff, 0.0) for _ in range(E)])
        for i in range(E):
            r.dense_experts[i].load_state_dict(legacy.experts[i].state_dict())
        if hasattr(legacy, "to_channels"):
            W = legacy.to_channels.weight.detach().cpu().clone()
            assert W.shape == (E * D, D), f"to_channels dense attendu {(E*D, D)}, vu {tuple(W.shape)}"
        else:  # déjà paginé (re-conversion) : reconstruit depuis le store
            rows = [pager.store.load_page(b, e)["tc_slice"] for e in range(E)]
            W = torch.cat(rows, dim=0)
        r.dense_tc_weight = W
        r.norm.load_state_dict(legacy.norm.state_dict())
        blk.cognitive_expert_router = r
        pager.register_router(b, r)
        pager.snapshot_router(b, r)
        # Libère les refs denses : le store disque est désormais autoritaire.
        # Sans ça, la RAM contiendrait les E experts → pas de paging réel !
        r.dense_experts = None
        r.dense_tc_weight = None
    import gc
    gc.collect()
    return model


# ───────────────────────────────────────────────────────────────────
# Driver hash-ahead : prefetch du batch N+1 pendant compute(N)
# ───────────────────────────────────────────────────────────────────

def hash_ahead_pages(next_token_ids: torch.Tensor, model: CogNetMoE1B) -> List[PageId]:
    """Pages EXACTES du prochain batch (mode token, tous blocs) — sans aucun forward."""
    pages: Set[PageId] = set()
    for b, blk in enumerate(model.blocks):
        r = blk.cognitive_expert_router
        # Polymorphe : dense-token et associatif exposent assign_pages(ids).
        # (Dense-LSH : pas de prefetch exact par ids → working_set_hint.)
        if not getattr(r, "supports_id_prefetch", False):
            continue
        a = r.assign_pages(next_token_ids, next_token_ids)
        for e in set(a.reshape(-1).tolist()):
            pages.add((b, e))
    return sorted(pages)


def working_set_hint(recent_pages: List[PageId], next_token_ids: torch.Tensor,
                     model: CogNetMoE1B) -> List[PageId]:
    """Prefetch LSH (ordre = priorité d'installation : les DERNIERS survivent) :
    1. Localité temporelle (pages récentes, bornées par l'appelant).
    2. EXACT couche 0 : hidden = encoder(ids) ne dépend d'AUCUN expert → LSH
       exact, soumis EN DERNIER (installé en dernier = MRU = survit).
    NOTE : pas de hint token-hash ici — c'est un espace d'adressage différent,
    donc du bruit pur pour LSH (mesuré : dilue le recall)."""
    pages: List[PageId] = [p for p in recent_pages]
    r0 = model.blocks[0].cognitive_expert_router
    if isinstance(r0, PagedHashExpertRouter) and r0.mode == "lsh":
        with torch.no_grad():
            h0 = model.encoder(next_token_ids)
            a0 = r0.lsh.experts(h0, r0.n_experts, r0.top_k)
        for e in set(a0.reshape(-1).tolist()):
            if (0, e) not in pages:
                pages.append((0, e))
    # Déduplique en gardant la dernière occurrence (priorité).
    seen, ordered = set(), []
    for p in reversed(pages):
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return list(reversed(ordered))


# ═══════════════════════════════════════════════════════════════════════
# Self-test : parité + anti-stall + writeback + quant
# ═══════════════════════════════════════════════════════════════════════

def _tiny_dense(E=8, seed=0):
    torch.manual_seed(seed)
    m = CogNetMoE1B(vocab_size=256, hidden_dim=32, num_blocks=2, num_channels=E,
                    channel_dim=16, ff_dim=64, max_seq_len=32, working_slots=2,
                    episodic_slots=4, semantic_slots=8, key_dim=16,
                    n_experts=E, top_k=2, use_gradient_checkpointing=False)
    from hash_moe import convert_to_hash
    return convert_to_hash(m, mode="token")


def self_test():
    print("=" * 70)
    print("ExpertPager — Self-test (CPU, latence SSD simulée)")
    print("=" * 70)
    import tempfile
    torch.manual_seed(0)
    tmp = tempfile.mkdtemp()

    # [1] Parité paginé == dense (bit à bit).
    print("\n[1] Parité paginé/dense...")
    dense = _tiny_dense(E=8)
    dense.eval()
    x = torch.randint(0, 256, (4, 16))
    for r in [b.cognitive_expert_router for b in dense.blocks]:
        r.set_batch_token_ids(x)
    with torch.no_grad():
        ref = dense(x)["logits"]
    paged = _tiny_dense(E=8)
    paged.load_state_dict(dense.state_dict())
    pager = ExpertPager(PagerConfig(resident_pages=6, store_dir=os.path.join(tmp, "s1")))
    paged = convert_to_paged(paged, n_slots=3, pager=pager, mode="token")
    paged.eval()
    for r in [b.cognitive_expert_router for b in paged.blocks]:
        r.set_batch_token_ids(x)
    with torch.no_grad():
        got = paged(x)["logits"]
    d = (ref - got).abs().max().item()
    print(f"  écart max dense↔paginé : {d:.2e} (S=3 slots, E=8 pages/bloc, vagues forcées)")
    assert d == 0.0, f"le paging doit être transparent! ({d})"
    print("  ✓ TRANSPARENCE BIT-À-BIT")

    # [2] Anti-stall : hash-ahead vs sync, latence 15ms simulée.
    print("\n[2] Anti-stall (io_delay=15ms, E=16, S=4)...")
    for tag, use_prefetch in [("SYNC (sans prefetch)", False), ("HASH-AHEAD", True)]:
        m = _tiny_dense(E=16, seed=1)
        m.train()
        pg = ExpertPager(PagerConfig(resident_pages=8, store_dir=os.path.join(tmp, f"s2_{tag[:4]}"),
                                     io_delay_ms=15.0, num_loaders=8))
        m = convert_to_paged(m, n_slots=4, pager=pg, mode="token")
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
        g = torch.Generator().manual_seed(7)
        batches = [torch.randint(0, 256, (8, 16), generator=g) for _ in range(6)]
        if use_prefetch:
            pg.prefetch(hash_ahead_pages(batches[0], m))  # warmup N+1
        t0 = time.time()
        for i, ids in enumerate(batches):
            if use_prefetch and i + 1 < len(batches):
                pg.prefetch(hash_ahead_pages(batches[i + 1], m))  # compute(i) ∥ load(i+1)
                pg._install_prefetched()
            for r in [b.cognitive_expert_router for b in m.blocks]:
                r.set_batch_token_ids(ids)
            opt.zero_grad()
            out = m(ids)["logits"].sum()
            if use_prefetch:
                # Le backward rejoue les mêmes pages → précharge le batch courant
                # (convertit les fautes sync du recompute en fautes couvertes).
                pg.prefetch(hash_ahead_pages(ids, m))
            out.backward()
            opt.step()
            pg.mark_all_resident_dirty()
        pg.flush()
        dt = time.time() - t0
        s = pg.summary()
        print(f"  {tag:22s} wall={dt:.2f}s hits={s['hits']} sync={s['faults_sync']} "
              f"couverts={s['faults_covered']} stall={s['stall_ms']:.0f}ms "
              f"writebacks={s['writebacks']} jokers={s['joker_fallbacks']}")
        if not use_prefetch:
            sync_wall, sync_stall = dt, s["stall_ms"]
        else:
            ah_wall, ah_stall = dt, s["stall_ms"]
            assert s["faults_covered"] > 0, "le prefetch devrait couvrir des fautes!"
    print(f"  → hash-ahead : {sync_wall/ah_wall:.2f}× plus rapide, "
          f"stall {sync_stall:.0f}ms → {ah_stall:.0f}ms")
    assert ah_wall < sync_wall, "le prefetch doit masquer la latence!"

    # [3] Writeback dirty : le store reflète l'entraînement.
    print("\n[3] Writeback des pages dirty...")
    sd_before = dict(pager.store.load_page(0, 0))
    m3 = _tiny_dense(E=8, seed=2)
    pg3 = ExpertPager(PagerConfig(resident_pages=4, store_dir=os.path.join(tmp, "s3")))
    m3 = convert_to_paged(m3, n_slots=2, pager=pg3, mode="token")
    m3.train()
    opt3 = torch.optim.AdamW([p for p in m3.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(4):
        ids = torch.randint(0, 256, (8, 16))
        for r in [b.cognitive_expert_router for b in m3.blocks]:
            r.set_batch_token_ids(ids)
        opt3.zero_grad()
        m3(ids)["logits"].sum().backward()
        opt3.step()
        pg3.mark_all_resident_dirty()
    pg3.flush()
    assert pg3.summary()["writebacks"] > 0, "aucun writeback?!"
    # Recharge frais depuis le store → forward identique au modèle entraîné.
    # (m4 reçoit l'état COMPLET de m3 ; son pager vide réinstalle tout depuis
    # le disque → tout écart vient d'un writeback manquant/faussé.)
    m3.eval()
    ids = torch.randint(0, 256, (4, 16))
    for r in [b.cognitive_expert_router for b in m3.blocks]:
        r.set_batch_token_ids(ids)
    with torch.no_grad():
        ref3 = m3(ids)["logits"]
    m4 = _tiny_dense(E=8, seed=99)
    pg4 = ExpertPager(PagerConfig(resident_pages=16, store_dir=os.path.join(tmp, "s4")))
    m4 = convert_to_paged(m4, n_slots=2, pager=pg4, mode="token")
    for b in range(2):
        for e in range(8):
            pg4.store.save_page(b, e, pg3.store.load_page(b, e))
    m4.load_state_dict(m3.state_dict())
    m4.eval()
    for r in [b.cognitive_expert_router for b in m4.blocks]:
        r.set_batch_token_ids(ids)
    with torch.no_grad():
        got4 = m4(ids)["logits"]
    d4 = (ref3 - got4).abs().max().item()
    print(f"  writebacks={pg3.summary()['writebacks']}, écart reload entraîné : {d4:.2e}")
    assert d4 == 0.0
    print("  ✓ WRITEBACK EXACT (le disque = l'état entraîné)")

    # [4] Quant int8 : ÷4 disque, fidélité.
    print("\n[4] Pages int8...")
    pgq = ExpertPager(PagerConfig(resident_pages=16, store_dir=os.path.join(tmp, "sq"),
                                  quantize="int8"))
    mq = convert_to_paged(_tiny_dense(E=8, seed=3), n_slots=8, pager=pgq, mode="token")
    mq.eval()
    dense8 = _tiny_dense(E=8, seed=3)
    dense8.eval()
    for r in [b.cognitive_expert_router for b in dense8.blocks]:
        r.set_batch_token_ids(x)
    for r in [b.cognitive_expert_router for b in mq.blocks]:
        r.set_batch_token_ids(x)
    with torch.no_grad():
        a = dense8(x)["logits"]
        b = mq(x)["logits"]
    cos = F.cosine_similarity(a.reshape(-1).float(), b.reshape(-1).float(), dim=0).item()
    raw = ExpertPageStore(os.path.join(tmp, "sraw")).disk_bytes() if False else None
    # taille comparée : snapshot fp32 vs int8
    pgs = ExpertPager(PagerConfig(resident_pages=16, store_dir=os.path.join(tmp, "sfp")))
    convert_to_paged(_tiny_dense(E=8, seed=3), n_slots=8, pager=pgs, mode="token")
    ratio = pgs.summary()["disk_bytes"] / max(1, pgq.summary()["disk_bytes"])
    print(f"  cos-sim fp32↔int8 : {cos:.6f}, ratio disque fp32/int8 : {ratio:.2f}× (attendu ~4×)")
    assert cos > 0.999 and ratio > 3.0
    print("  ✓ INT8 FIDÈLE (cos>0.999) ET COMPACT (÷4)")

    # [5] Recall du hint LSH (working-set + token-hash).
    print("\n[5] Prefetch LSH (heuristique)...")
    ml = _tiny_dense(E=16, seed=4)
    from hash_moe import convert_to_hash as _cth
    ml = _cth(ml, mode="lsh")
    pgl = ExpertPager(PagerConfig(resident_pages=8, store_dir=os.path.join(tmp, "sl")))
    ml = convert_to_paged(ml, n_slots=4, pager=pgl, mode="lsh")
    ml.eval()
    # Embeddings corrélés par groupe (simule des représentations ENTRAÎNÉES :
    # à embeddings purement aléatoires, LSH disperse uniformément et tout test
    # de localité est vide de sens). Groupe de 16 tokens = même base + bruit.
    with torch.no_grad():
        E_emb, D_emb = ml.encoder.token_emb.weight.shape
        _g = torch.Generator().manual_seed(5)
        bases = torch.randn(E_emb // 16 + 1, D_emb, generator=_g) * 2.0
        noise = torch.randn(E_emb, D_emb, generator=_g) * 0.15
        ml.encoder.token_emb.weight.copy_(
            bases[torch.arange(E_emb) // 16] + noise)
    g = torch.Generator().manual_seed(11)
    # Batches à localité topique + drift LENT (réaliste : burstiness + continuité
    # topique ; à i.i.d. pur, ~toutes les pages sont touchées et même un oracle
    # serait plafonné à S/touchées).
    batches = []
    for i in range(6):
        base = (i * 8) % 200
        batches.append(torch.randint(base, base + 56, (8, 16), generator=g) % 256)
    recent: List[PageId] = []
    recalls, coverages, needs = [], [], []
    with torch.no_grad():
        for i, ids in enumerate(batches):
            hint = working_set_hint(recent, ids, ml)
            pgl.prefetch(hint)
            pgl._drain()  # collecte (en prod : chevauché avec le compute)
            # pages réellement demandées (LSH exact, sans prefetch préalable)
            need = set()
            h = ml.encoder(ids)
            for b, blk in enumerate(ml.blocks):
                r = blk.cognitive_expert_router
                a = r.lsh.experts(h, r.n_experts, r.top_k)
                for e in set(a.reshape(-1).tolist()):
                    need.add((b, e))
                h, _ = blk.memory(h)
                h = blk.composer(h)
                h = blk.norm(h)
            # Couverture du HINT (qualité de la prédiction, indépendante des slots)
            coverages.append(len(set(hint) & need) / max(1, len(need)))
            # Recall RÉSIDENT (bout-en-bout : hint + pression des slots)
            hit = sum(1 for p in need if p in pgl.resident)
            recalls.append(hit / max(1, len(need)))
            needs.append(len(need))
            recent = sorted(need)[-8:]  # borné (~slots) : pas de thrashing
            # forward réel (remplit le résident pour la suite)
            ml(ids)
    import statistics as _st
    print(f"  pages touchées/batch : {needs} (LSH disperse : RoPE rend les hiddens position-dépendants)")
    print(f"  couverture du hint   : {[f'{r:.2f}' for r in coverages]} "
          f"(moy {sum(coverages)/len(coverages):.2f} — qualité de la prédiction)")
    print(f"  recall résident      : {[f'{r:.2f}' for r in recalls]} "
          f"(moy {sum(recalls)/len(recalls):.2f}, plafond slots ≈ "
          f"{8.0 / _st.mean(needs):.2f} — le reste est couvert en sync)")
    assert sum(coverages) / len(coverages) > 0.5, "le hint doit couvrir >50% du besoin"
    print("  ✓ HEURISTIQUE LSH MESURÉE (couche-0 exacte + localité temporelle)")

    print("\n" + "=" * 70)
    print("✓ Self-test ExpertPager passé : transparence + anti-stall + writeback + int8.")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.parse_args()
    self_test()
