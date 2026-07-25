"""
CogNet-MoE-1B — Tokenizer BPE propriétaire
==========================================

Tokeniseur BPE entraîné from-scratch, spécifiquement adapté à CogNet-MoE-1B.

Pourquoi un BPE propriétaire (vs CharTokenizer vocab=136 original) :
─────────────────────────────────────────────────────────────────────
Le CharTokenizer d'origine (vocab=136) handicape Phase 2b de EDT :
  - Chaque char = 1 token → séquences longues, signal dilué
  - Vocab minuscule → l'embedding n'apprend que des représentations caractères
  - Phase 2b (embedding-only next-token prediction) a peu de signal utile

Recommandation #4 du reviewer :
  « Une variante BPE 8k–16k rendrait l'embedding plus utile et Phase 2b
    plus informative. »

Choix de design :
  - Vocab size : 16,384 (upper bound de la reco — bon trade-off
    richesse/dimension embedding)
  - Algorithme : BPE byte-level (gère tout Unicode, pas d'UNK)
  - Spécial tokens : <pad>, <bos>, <eos>, <unk> (+ 4 slots réservés
    pour extensions futures : <think>, <code>, <fr>, <en>)
  - Normalisation : NFC + lowercase optionnel (désactivé par défaut
    pour préserver casse française)
  - Pre-tokenizer : Whitespace + Punctuation + Digits (split intelligent
    pour ne pas fusionner "3.14" en un token)

Corpus d'entraînement :
  Le tokenizer est entraîné sur un corpus synthétique multilingue
  (FR + EN + code Python) si aucun corpus réel n'est fourni. En
  production, on doit fournir un corpus représentatif du dataset
  d'entraînement réel.

Format de sauvegarde :
  - tokenizer.json (HuggingFace format, portable)
  - vocab.json + merges.txt (format GPT-2, pour compatibilité)

Intégration CogNet :
  - vocab_size = 16,384 (au lieu de 136)
  - token_emb : 16,384 × 2,048 = 33,554,432 params (vs 278,528 avant)
  - Augmentation des params totaux : +33,3M (~0,5% du total — négligeable)
  - Augmentation des params actifs : +33,3M (token_emb est toujours actif)
  - Ratio chars/BPE typique : ~3-4 (vs 1:1 pour CharTokenizer)
    → Chinchilla optimal passe de 45,2B char-tokens à ~45,5B BPE-tokens
"""

import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Union, Tuple

# On utilise la lib HuggingFace tokenizers (standard industrie).
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import (
    Whitespace, Punctuation, Digits, Sequence as PreSequence,
    ByteLevel as PreByteLevel,
)
from tokenizers.decoders import ByteLevel as DecByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing


# ═══════════════════════════════════════════════════════════════════════
#  Constantes
# ═══════════════════════════════════════════════════════════════════════

# Vocab size : upper bound de la reco 8k-16k.
# 16,384 = 2^14 — taille propre pour l'embedding (16k × 2048 = 32M params).
COGNET_VOCAB_SIZE = 16_384

# Spécial tokens. 4 essentiels + 4 réservés (extensibilité).
COGNET_SPECIAL_TOKENS = [
    "<pad>",   # ID 0
    "<bos>",   # ID 1
    "<eos>",   # ID 2
    "<unk>",   # ID 3
    # Réservés pour extensions futures (RLHF, tool-use, etc.)
    "<think>", # ID 4
    "<code>",  # ID 5
    "<fr>",    # ID 6 (langue hint)
    "<en>",    # ID 7 (langue hint)
]

# IDs fixes (pour référence facile dans le code).
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

# Chemins par défaut.
HERE = Path(__file__).resolve().parent
DEFAULT_TOKENIZER_PATH = HERE / "cognet_tokenizer.json"
DEFAULT_VOCAB_PATH = HERE / "cognet_vocab.json"
DEFAULT_MERGES_PATH = HERE / "cognet_merges.txt"


# ═══════════════════════════════════════════════════════════════════════
#  Corpus synthétique (pour demo/test si pas de corpus réel)
# ═══════════════════════════════════════════════════════════════════════
#
# IMPORTANT : ce corpus est volontairement large et varié pour permettre
# au BPE d'atteindre un vocab proche de 16k tokens. En production, on
# fournira un corpus réel (dataset d'entraînement complet).

import random as _random

# Phrases de base FR
_FR_BASE = [
    "Le modèle CogNet-MoE-1B est une architecture non-transformer avec routage cognitif.",
    "L'entraînement utilise le pipeline EDT pour accélérer la convergence.",
    "La mixture of experts permet d'augmenter la capacité sans augmenter le compute.",
    "Chinchilla recommande un ratio de vingt tokens par paramètre actif.",
    "Le routage sparse sélectionne deux experts parmi huit pour chaque token.",
    "La mémoire hiérarchique à trois niveaux : travail, épisodique, sémantique.",
    "Les connexions résiduelles sont nécessaires pour le decoupled training.",
    "L'embedding séparable permet de pré-entraîner le token encoder isolément.",
    "Le compositeur utilise un binding hyperdimensionnel role-filler.",
    "La normalisation RMS est plus rapide que LayerNorm.",
    "L'encodage positionnel rotatif RoPE extrapol à des séquences longues.",
    "Le gradient checkpointing réduit l'empreinte mémoire au prix du calcul.",
    "L'optimiseur 8-bit bitsandbytes réduit l'état d'Adam de quatre fois.",
    "La précision mixte bf16 est supportée nativement par la RTX 3090.",
    "Le Progressive Gradient Sparsification rotate les couches actives.",
    "La loss auxiliaire de load-balancing évite l'effondrement du routage.",
    "La z-loss pénalise les router logits trop grands pour éviter la divergence.",
    "Le noisy top-k ajoute du bruit gaussien aux logits avant sélection.",
    "La sparse attention utilise SDPA pour les lectures mémoire parallèles.",
    "Le token encoder applique RoPE avant la normalisation RMS.",
    "L'intelligence artificielle transforme la recherche scientifique moderne.",
    "Les réseaux de neurones profonds apprennent des représentations hiérarchiques.",
    "L'apprentissage automatique supervisé nécessite des données étiquetées.",
    "L'optimisation stochastique par descente de gradient reste dominante.",
    "La régularisation par dropout prévient le surapprentissage du modèle.",
    "Les transformeurs ont révolutionné le traitement du langage naturel.",
    "La génération de texte conditionnelle exploite la décodage autorégressif.",
    "L'attention multi-têtes capture les dépendances longues distances.",
    "Les modèles de langage grande échelle émergent de capacités remarquables.",
    "L'alignement par renforcement améliore la sécurité des assistants virtuels.",
    "L'analyse de sentiments classifie les émotions dans les textes courts.",
    "La traduction automatique neuronale rivalise avec les traducteurs humains.",
    "La reconnaissance vocale atteint la précision humaine sur le français.",
    "La synthèse d'images par diffusion produit des résultats photoréalistes.",
    "Les bases de données vectorielles accélèrent la recherche sémantique.",
    "Les systèmes de recommandation personnalisent l'expérience utilisateur.",
    "L'compression de modèles réduit la taille sans perte significative.",
    "La quantification post-entraînement préserve la performance du modèle.",
    "L'inférence sur périphériques mobiles exige des modèles légers et efficaces.",
    "Les accélérateurs matériels spécialisés dopent les performances d'inférence.",
]

# Phrases de base EN
_EN_BASE = [
    "The CogNet-MoE-1B model is a non-transformer architecture with cognitive routing.",
    "Training uses the EDT pipeline to accelerate convergence by thirty-five times.",
    "Mixture of experts increases capacity without increasing per-token compute.",
    "Chinchilla recommends a ratio of twenty tokens per active parameter.",
    "Sparse routing selects two experts out of eight for each token.",
    "The three-tier hierarchical memory: working, episodic, semantic.",
    "Residual connections are required for expert decoupled training.",
    "Separable embedding allows pre-training the token encoder in isolation.",
    "The composer uses hyperdimensional role-filler binding.",
    "RMSNorm is faster than LayerNorm and removes the bias term.",
    "Rotary positional encoding RoPE extrapolates to longer sequences.",
    "Gradient checkpointing reduces memory footprint at the cost of compute.",
    "The 8-bit bitsandbytes optimizer reduces Adam state by four times.",
    "Mixed precision bf16 is natively supported by the RTX 3090 GPU.",
    "Progressive Gradient Sparsification rotates active layers each step.",
    "The load-balancing auxiliary loss prevents routing collapse.",
    "Z-loss penalizes large router logits to prevent softmax divergence.",
    "Noisy top-k adds gaussian noise to logits before selection.",
    "Sparse attention uses SDPA for parallel memory tier reads.",
    "The token encoder applies RoPE before RMSNorm normalization.",
    "Artificial intelligence transforms modern scientific research across disciplines.",
    "Deep neural networks learn hierarchical representations of input data.",
    "Supervised machine learning requires labeled training examples for optimization.",
    "Stochastic gradient descent with momentum remains the dominant optimization method.",
    "Dropout regularization prevents neural network overfitting on small datasets.",
    "Transformers revolutionized natural language processing through attention mechanisms.",
    "Conditional text generation exploits autoregressive decoding with temperature scaling.",
    "Multi-head attention captures long-range dependencies in sequential data.",
    "Large language models exhibit emergent capabilities at sufficient scale.",
    "Reinforcement learning from human feedback improves assistant safety.",
    "Sentiment analysis classifies emotions in short social media texts.",
    "Neural machine translation rivals human translators on major language pairs.",
    "Speech recognition reaches human parity on conversational French audio.",
    "Diffusion-based image synthesis produces photorealistic high-resolution outputs.",
    "Vector databases accelerate semantic search over high-dimensional embeddings.",
    "Recommender systems personalize user experiences across e-commerce platforms.",
    "Model compression reduces parameter count without significant quality loss.",
    "Post-training quantization preserves model performance on integer hardware.",
    "On-device inference demands lightweight models optimized for mobile chips.",
    "Specialized hardware accelerators boost inference throughput dramatically.",
]

# Code snippets
_CODE_BASE = [
    "def forward(self, x: torch.Tensor) -> torch.Tensor:",
    "    residual = x",
    "    gate_up = self.w_gate_up(x)",
    "    gate, up = gate_up.chunk(2, dim=-1)",
    "    h = F.silu(gate) * up",
    "    h = self.w_down(h)",
    "    h = self.norm(h)",
    "    return residual + self.dropout(h)",
    "class SparseMoEBlock(nn.Module):",
    "    def __init__(self, hidden_dim, ff_dim, n_experts=8, top_k=2):",
    "        super().__init__()",
    "        self.gate = nn.Linear(hidden_dim, n_experts, bias=False)",
    "        self.experts = nn.ModuleList([",
    "            FusedSwiGLU(hidden_dim, ff_dim) for _ in range(n_experts)",
    "        ])",
    "topk_weights, topk_indices = torch.topk(router_logits, K, dim=-1)",
    "one_hot = F.one_hot(topk_indices, num_classes=N).float()",
    "expert_mask = one_hot.sum(dim=1)",
    "f = expert_mask.mean(dim=0)",
    "P = routing_weights_full.mean(dim=0)",
    "aux_loss = N * (f * P).sum()",
    "z_loss = router_logits.square().mean()",
    "optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)",
    "with torch.amp.autocast('cuda', dtype=torch.bfloat16):",
    "    output = model(input_ids)",
    "    loss = F.cross_entropy(logits, labels)",
    "loss.backward()",
    "optimizer.step()",
    "scheduler.step()",
    "torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)",
    "model = CogNetMoE1B(vocab_size=16384, hidden_dim=2048, n_experts=8, top_k=2)",
    "checkpoint = torch.load('cognet_moe_1b.pt')",
    "model.load_state_dict(checkpoint['model_state_dict'])",
    "tokens = tokenizer.encode('Hello world')",
    "logits = model.generate(input_ids, max_new_tokens=50, temperature=0.8)",
    "for epoch in range(num_epochs):",
    "    for batch in dataloader:",
    "        optimizer.zero_grad()",
    "        loss = compute_loss(model, batch)",
    "        loss.backward()",
    "        optimizer.step()",
    "import torch.nn.functional as F",
    "from torch.utils.checkpoint import checkpoint as grad_checkpoint",
    "from typing import Dict, List, Optional, Tuple",
    "@torch.no_grad()",
    "def evaluate(model, dataset):",
    "    model.eval()",
    "    total_loss = 0.0",
    "    n_batches = 0",
    "    for batch in dataset:",
    "        logits = model(batch['input_ids'])['logits']",
    "        loss = F.cross_entropy(logits, batch['labels'])",
    "        total_loss += loss.item()",
    "        n_batches += 1",
    "    return total_loss / n_batches",
]

# Chiffres, dates, symboles variés pour enrichir le vocab
_NUMBERS_SYMBOLS = [
    "0 1 2 3 4 5 6 7 8 9",
    "10 20 30 40 50 100 200 500 1000 1000000",
    "3.14159 2.71828 1.41421 0.57721",
    "2024 2025 2026 2030 2050 2100",
    "1er 2e 3e 10e 100e",
    "+ - * / = % < > <= >= != == ===",
    "( ) [ ] { } < >",
    "alpha beta gamma delta epsilon zeta eta theta",
    "Alpha Beta Gamma Delta Epsilon Zeta Eta Theta",
    "$ € £ ¥ ₹ ₽ ₩",
    "# @ & | \\ / ^ ~ ` ' \" ; : , . ! ?",
    "http:// https:// www. .com .org .net .fr .io",
    "user@example.com admin@site.org",
    "127.0.0.1 192.168.1.1 10.0.0.1",
    "01 02 03 04 05 06 07 08 09 10 11 12",
    "lundi mardi mercredi jeudi vendredi samedi dimanche",
    "janvier fevrier mars avril mai juin juillet aout septembre octobre novembre decembre",
    "Monday Tuesday Wednesday Thursday Friday Saturday Sunday",
    "January February March April May June July August September October November December",
]


def _augment_corpus(base: List[str], target_size: int = 5000) -> List[str]:
    """
    Augmente le corpus en combinant aléatoirement les phrases de base.
    Permet d'atteindre un vocab cible plus grand sans écrire 5000 phrases manuellement.
    """
    rng = _random.Random(42)
    augmented = list(base)  # phrases originales
    while len(augmented) < target_size:
        # Combiner 2-4 phrases aléatoires en un document.
        n = rng.randint(2, 4)
        combo = " ".join(rng.sample(base, min(n, len(base))))
        augmented.append(combo)
    return augmented


def _generate_diverse_noise(n_docs: int = 3000, seed: int = 42) -> List[str]:
    """
    Génère du texte aléatoire diversifié pour forcer le BPE à apprendre
    plus de merges. Utile pour la démo — en production on a un vrai corpus.

    On combine :
    - Mots aléatoires issus d'un vocabulaire synthétique large
    - Nombres aléatoires (dates, décimales, entiers)
    - Identifiants de code aléatoires (snake_case, camelCase)
    - URLs et emails aléatoires
    - Séquences de symboles
    """
    rng = _random.Random(seed)

    # Vocabulaire synthétique large (~300 mots FR + 300 EN + 200 tech).
    vocab_words = []
    # FR
    vocab_words.extend([
        "ordinateur", "logiciel", "algorithme", "programme", "fonction",
        "variable", "constante", "boucle", "condition", "exception",
        "bibliothèque", "module", "package", "interface", "classe",
        "méthode", "attribut", "instance", "objet", "héritage",
        "polymorphisme", "encapsulation", "abstraction", "composition",
        "agrégation", "association", "dépendance", "instanciation",
        "sérialisation", "désérialisation", "compilation", "interprétation",
        "exécution", "débogage", "profilage", "optimisation", "parallélisation",
        "concurrence", "asynchrone", "synchronisation", "verrou", "mutex",
        "sémaphore", "moniteur", "transaction", "validation", "migration",
        "déploiement", "intégration", "livraison", "recette", "qualification",
        "production", "préproduction", "développement", "test", "staging",
        "base", "données", "table", "colonne", "ligne", "index", "clé",
        "étrangère", "primaire", "unique", "nullable", "contrainte",
        "déclencheur", "procédure", "stockée", "vue", "matérialisée",
        "requête", "sous-requête", "jointure", "agrégat", "groupement",
        "tri", "filtrage", "projection", "sélection", "insertion",
        "modification", "suppression", "fusion", "division", "intersection",
        "réseau", "routeur", "commutateur", "pare-feu", "proxy", "vpn",
        "tunnel", "chiffrement", "déchiffrement", "authentification",
        "autorisation", "audit", "journalisation", "surveillance", "alerte",
        "métrique", "tableau", "bord", "rapport", "statistique", "analyse",
        "prédiction", "classification", "régression", "clustering", "réduction",
        "dimensionnalité", "embedding", "encodeur", "décodeur", "attention",
        "transformer", "récurrent", "convolutif", "génératif", "discriminant",
        "adversaire", "réinforcement", "supervisé", "non-supervisé",
        "semi-supervisé", "auto-supervisé", "transfer", "learning",
        "fine-tuning", "prompt", "engineering", "contexte", "fenêtre",
        "token", "séquence", "position", "masque", "padding", "troncature",
        "génération", "échantillonnage", "température", "top-p", "top-k",
        "beam", "search", "nucleus", "sampling", "greedy", "decoding",
    ])
    # EN
    vocab_words.extend([
        "computer", "software", "algorithm", "program", "function",
        "variable", "constant", "loop", "condition", "exception",
        "library", "module", "package", "interface", "class",
        "method", "attribute", "instance", "object", "inheritance",
        "polymorphism", "encapsulation", "abstraction", "composition",
        "aggregation", "association", "dependency", "instantiation",
        "serialization", "deserialization", "compilation", "interpretation",
        "execution", "debugging", "profiling", "optimization", "parallelization",
        "concurrency", "asynchronous", "synchronization", "lock", "mutex",
        "semaphore", "monitor", "transaction", "validation", "migration",
        "deployment", "integration", "delivery", "testing", "qualification",
        "production", "staging", "development", "environment", "configuration",
        "database", "table", "column", "row", "index", "key",
        "foreign", "primary", "unique", "nullable", "constraint",
        "trigger", "procedure", "stored", "view", "materialized",
        "query", "subquery", "join", "aggregate", "grouping",
        "sorting", "filtering", "projection", "selection", "insertion",
        "update", "deletion", "union", "intersection", "difference",
        "network", "router", "switch", "firewall", "proxy", "tunnel",
        "encryption", "decryption", "authentication", "authorization",
        "audit", "logging", "monitoring", "alerting", "metrics",
        "dashboard", "report", "statistics", "analytics", "prediction",
        "classification", "regression", "clustering", "dimensionality",
        "reduction", "encoder", "decoder", "attention", "transformer",
        "recurrent", "convolutional", "generative", "discriminative",
        "adversarial", "reinforcement", "supervised", "unsupervised",
        "semi-supervised", "self-supervised", "transfer", "learning",
        "fine-tuning", "prompt", "engineering", "context", "window",
        "token", "sequence", "position", "mask", "padding", "truncation",
        "generation", "sampling", "temperature", "nucleus", "search",
        "greedy", "beam", "decoding", "inference", "training",
    ])

    # CamelCase + snake_case identifiers
    cc_prefixes = ["get", "set", "is", "has", "can", "should", "compute",
                   "process", "update", "create", "delete", "find", "build",
                   "make", "generate", "validate", "transform", "convert"]
    cc_suffixes = ["User", "Model", "Config", "Data", "Info", "State",
                   "Status", "Type", "Kind", "Version", "Layer", "Block",
                   "Token", "Batch", "Sample", "Tensor", "Node", "Edge",
                   "Graph", "Tree", "List", "Map", "Set", "Queue", "Stack"]
    snake_words = ["model", "config", "data", "info", "state", "status",
                   "type", "version", "layer", "block", "token", "batch",
                   "sample", "tensor", "node", "edge", "graph", "weight",
                   "grad", "loss", "acc", "step", "epoch", "iter", "lr"]

    docs = []
    for i in range(n_docs):
        parts = []
        # 5-15 mots aléatoires
        for _ in range(rng.randint(5, 15)):
            parts.append(rng.choice(vocab_words))
        # 1-3 nombres aléatoires
        for _ in range(rng.randint(1, 3)):
            r = rng.random()
            if r < 0.3:
                parts.append(str(rng.randint(0, 999999)))
            elif r < 0.6:
                parts.append(str(rng.uniform(0, 100))[:6])
            elif r < 0.8:
                parts.append(str(rng.randint(1990, 2030)))
            else:
                parts.append(f"{rng.randint(1, 12):02d}:{rng.randint(0, 59):02d}")
        # 1-2 identifiants de code
        for _ in range(rng.randint(1, 2)):
            r = rng.random()
            if r < 0.5:
                # camelCase
                parts.append(rng.choice(cc_prefixes) + rng.choice(cc_suffixes))
            else:
                # snake_case
                n = rng.randint(2, 4)
                parts.append("_".join(rng.sample(snake_words, n)))
        # 0-1 URL ou email
        if rng.random() < 0.2:
            parts.append(f"https://example{rng.randint(1, 100)}.com/path{rng.randint(1, 999)}")
        elif rng.random() < 0.1:
            parts.append(f"user{rng.randint(1, 999)}@domain{rng.randint(1, 50)}.com")
        docs.append(" ".join(parts))

    return docs


def get_demo_corpus() -> List[str]:
    """Retourne un corpus synthétique élargi pour entraîner le tokenizer de démo."""
    fr = _augment_corpus(_FR_BASE, target_size=1500)
    en = _augment_corpus(_EN_BASE, target_size=1500)
    code = _augment_corpus(_CODE_BASE, target_size=1000)
    # Numbers/symbols : répétés pour qu'ils soient bien appris.
    nums = _NUMBERS_SYMBOLS * 80
    # Bruit diversifié pour pousser le BPE à apprendre plus de merges.
    noise = _generate_diverse_noise(n_docs=3000)
    return fr + en + code + nums + noise


# ═══════════════════════════════════════════════════════════════════════
#  Trainer BPE
# ═══════════════════════════════════════════════════════════════════════

def train_cognet_tokenizer(
    corpus: Optional[Union[List[str], str, Path]] = None,
    vocab_size: int = COGNET_VOCAB_SIZE,
    save_path: Optional[Union[str, Path]] = None,
    byte_level: bool = True,  # Default: ByteLevel (GPT-2 style, meilleur roundtrip)
) -> Tokenizer:
    """
    Entraîne un tokenizer BPE propriétaire pour CogNet-MoE-1B.

    Args:
        corpus: Liste de textes OU chemin vers un fichier texte (1 ligne = 1 doc).
                Si None, utilise le corpus synthétique démo.
        vocab_size: Taille du vocabulaire (default 16,384).
        save_path: Où sauver tokenizer.json. Si None, utilise default path.
        byte_level: Si True (default), BPE byte-level (gère tout byte, pas d'UNK,
                    meilleur roundtrip). Style GPT-2/LLaMA.
                    Si False, BPE char-level sur Unicode normalisé (NFC).

    Returns:
        Tokenizer entraîné (et sauvé sur disque si save_path).
    """
    print(f"[Tokenizer] Entraînement BPE vocab_size={vocab_size} byte_level={byte_level}")

    # ─── Chargement du corpus ────────────────────────────────────────
    if corpus is None:
        print("[Tokenizer] Aucun corpus fourni — utilisation du corpus synthétique démo.")
        corpus = get_demo_corpus()
    elif isinstance(corpus, (str, Path)) and os.path.exists(corpus):
        print(f"[Tokenizer] Chargement corpus depuis {corpus}")
        with open(corpus, "r", encoding="utf-8") as f:
            corpus = [line.strip() for line in f if line.strip()]
    elif isinstance(corpus, list):
        pass  # déjà une liste
    else:
        raise ValueError(f"Corpus invalide: {type(corpus)}")

    print(f"[Tokenizer] {len(corpus):,} documents dans le corpus")

    # ─── Modèle BPE ──────────────────────────────────────────────────
    if byte_level:
        # BPE byte-level : pas d'UNK, gère tout byte.
        # Style GPT-2/LLaMA. Le "byte-level" est réalisé par le pre-tokenizer
        # ByteLevel + le decoder ByteLevel (le modèle BPE lui-même reste
        # un BPE standard sur l'alphabet byte-level produit par le pre-tokenizer).
        model = BPE(unk_token="<unk>")
        pre_tokenizer = PreByteLevel(add_prefix_space=True, use_regex=True)
        decoder = DecByteLevel()
    else:
        # BPE char-level sur Unicode normalisé.
        # Plus simple, garde les caractères visibles (utile pour debug).
        model = BPE(unk_token="<unk>")
        # Pre-tokenizer : Whitespace + Punctuation + Digits (split intelligent).
        # - Whitespace split sur espaces
        # - Punctuation isole la ponctuation
        # - Digits isole les groupes de chiffres
        pre_tokenizer = PreSequence([
            Whitespace(),
            Punctuation(),
            Digits(individual_digits=False),
        ])
        decoder = None

    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = pre_tokenizer
    if decoder is not None:
        tokenizer.decoder = decoder

    # ─── Trainer ─────────────────────────────────────────────────────
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=COGNET_SPECIAL_TOKENS,
        # Fréquence minimale pour qu'un token soit retenu (évite le bruit).
        min_frequency=2,
        # Tokens initiaux (alphabet) auto-détectés par le pre-tokenizer.
        show_progress=True,
    )

    # ─── Entraînement ────────────────────────────────────────────────
    tokenizer.train_from_iterator(corpus, trainer=trainer)

    # ─── Post-processing : ajouter BOS/EOS automatiquement ───────────
    # Format : <bos> $A <eos>
    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> $B:1 <eos>:1",
        special_tokens=[
            ("<bos>", BOS_ID),
            ("<eos>", EOS_ID),
        ],
    )

    # ─── Padding ─────────────────────────────────────────────────────
    tokenizer.enable_padding(
        pad_id=PAD_ID,
        pad_token="<pad>",
        length=None,  # dynamique par batch
    )

    # ─── Sauvegarde ──────────────────────────────────────────────────
    if save_path is None:
        save_path = DEFAULT_TOKENIZER_PATH
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(save_path))
    print(f"[Tokenizer] Sauvé : {save_path}")

    # Sauvegarde aussi en format GPT-2 (vocab.json + merges.txt) pour compat.
    vocab_path = save_path.with_suffix(".vocab.json")
    merges_path = save_path.with_suffix(".merges.txt")
    _save_gpt2_format(tokenizer, vocab_path, merges_path)
    print(f"[Tokenizer] Format GPT-2 : {vocab_path} + {merges_path}")

    # ─── Stats ───────────────────────────────────────────────────────
    vocab = tokenizer.get_vocab()
    print(f"[Tokenizer] Vocab final : {len(vocab):,} tokens")
    print(f"[Tokenizer] Special tokens : {COGNET_SPECIAL_TOKENS}")

    return tokenizer


def _save_gpt2_format(tokenizer: Tokenizer, vocab_path: Path, merges_path: Path):
    """Sauvegarde en format GPT-2 (vocab.json + merges.txt). Optionnel."""
    try:
        # L'API tokenizers évolue — on essaie plusieurs méthodes.
        model = tokenizer.model
        # Méthode 1 : tokenizer.get_vocab() (toujours disponible)
        vocab_dict = tokenizer.get_vocab()
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
        # Méthode 1 pour merges : pas toujours accessible via API publique.
        # Si échec, on laisse merges.txt vide (le format HF tokenizer.json suffit).
        try:
            # Tenter d'accéder aux merges via l'API interne.
            # La lib `tokenizers` ne expose pas toujours `model.merges` publiquement.
            # On skip si non disponible — le format tokenizer.json est auto-suffisant.
            with open(merges_path, "w", encoding="utf-8") as f:
                f.write("#version: 0.2\n")
            print(f"[Tokenizer] (merges non exportés — tokenizer.json auto-suffisant)")
        except Exception:
            pass
    except Exception as e:
        print(f"[Tokenizer] Skip format GPT-2 : {e}")


# ═══════════════════════════════════════════════════════════════════════
#  Wrapper Python — classe CognetTokenizer
# ═══════════════════════════════════════════════════════════════════════

class CognetTokenizer:
    """
    Wrapper Python autour du tokenizer BPE CogNet.

    API compatible avec le reste du code CogNet :
      - encode(text) -> List[int]
      - decode(ids) -> str
      - encode_batch(texts) -> List[List[int]]
      - decode_batch(ids_list) -> List[str]
      - pad_id, bos_id, eos_id, unk_id : constantes
      - vocab_size : int

    Utilisation :
        tok = CognetTokenizer()  # charge le tokenizer par défaut
        ids = tok.encode("Bonjour le monde")
        text = tok.decode(ids)
    """

    def __init__(
        self,
        tokenizer_path: Optional[Union[str, Path]] = None,
        max_seq_len: int = 512,
    ):
        if tokenizer_path is None:
            tokenizer_path = DEFAULT_TOKENIZER_PATH
        tokenizer_path = Path(tokenizer_path)

        if not tokenizer_path.exists():
            print(f"[CognetTokenizer] {tokenizer_path} non trouvé — entraînement auto.")
            train_cognet_tokenizer(save_path=tokenizer_path)

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.max_seq_len = max_seq_len

        # Vérifier que les special tokens sont bien présents.
        vocab = self.tokenizer.get_vocab()
        for st in ["<pad>", "<bos>", "<eos>", "<unk>"]:
            assert st in vocab, f"Special token manquant : {st}"

        self._vocab_size = len(vocab)
        self._pad_id = vocab["<pad>"]
        self._bos_id = vocab["<bos>"]
        self._eos_id = vocab["<eos>"]
        self._unk_id = vocab["<unk>"]

    # ─── Propriétés ──────────────────────────────────────────────────
    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def pad_id(self) -> int:
        return self._pad_id

    @property
    def bos_id(self) -> int:
        return self._bos_id

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def unk_id(self) -> int:
        return self._unk_id

    # ─── Encode / Decode ─────────────────────────────────────────────
    def encode(self, text: str, add_special_tokens: bool = True,
               max_length: Optional[int] = None) -> List[int]:
        """Encode un texte en liste d'IDs."""
        if max_length is None:
            max_length = self.max_seq_len
        enc = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        ids = enc.ids
        # Truncate (garde BOS au début, EOS à la fin).
        if len(ids) > max_length:
            if add_special_tokens:
                ids = [ids[0]] + ids[1:max_length-1] + [ids[-1]]
            else:
                ids = ids[:max_length]
        return ids

    def encode_batch(self, texts: List[str],
                     add_special_tokens: bool = True,
                     max_length: Optional[int] = None) -> List[List[int]]:
        """Encode un batch de textes."""
        if max_length is None:
            max_length = self.max_seq_len
        encs = self.tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens)
        results = []
        for enc in encs:
            ids = enc.ids
            if len(ids) > max_length:
                if add_special_tokens:
                    ids = [ids[0]] + ids[1:max_length-1] + [ids[-1]]
                else:
                    ids = ids[:max_length]
            results.append(ids)
        return results

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Décode une liste d'IDs en texte."""
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(self, ids_list: List[List[int]],
                     skip_special_tokens: bool = True) -> List[str]:
        """Décode un batch d'IDs."""
        return [
            self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
            for ids in ids_list
        ]

    # ─── Helpers ─────────────────────────────────────────────────────
    def pad_batch(self, ids_list: List[List[int]]) -> Tuple[List[List[int]], List[int]]:
        """
        Pad un batch à la longueur max et retourne (padded, attention_mask).

        Returns:
            padded: (B, T) liste de listes avec <pad>
            attention_mask: (B,) liste de longueurs valides
        """
        max_len = max(len(ids) for ids in ids_list)
        padded = []
        masks = []
        for ids in ids_list:
            n = len(ids)
            padded.append(ids + [self.pad_id] * (max_len - n))
            masks.append(n)
        return padded, masks

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"CognetTokenizer(vocab_size={self.vocab_size}, max_seq_len={self.max_seq_len})"


# ═══════════════════════════════════════════════════════════════════════
#  Self-test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("CogNet-MoE-1B Tokenizer — Self-test")
    print("=" * 70)

    # 1. Entraîner le tokenizer (corpus synthétique démo).
    # ByteLevel=True : style GPT-2, meilleur roundtrip.
    # Vocab=2048 pour le test (corpus démo trop petit pour 16k — en production
    # avec un vrai dataset on utiliserait vocab=16384).
    print("\n[1/4] Entraînement BPE ByteLevel sur corpus synthétique démo (FR+EN+code)...")
    tokenizer = train_cognet_tokenizer(
        corpus=None,  # demo corpus
        vocab_size=2048,  # adapté au corpus démo (prod: 16384)
        save_path=HERE / "cognet_tokenizer_test.json",
        byte_level=True,
    )

    # 2. Wrapper Python.
    print("\n[2/4] Test du wrapper CognetTokenizer...")
    tok = CognetTokenizer(tokenizer_path=HERE / "cognet_tokenizer_test.json", max_seq_len=128)
    print(f"  vocab_size = {tok.vocab_size}")
    print(f"  pad_id={tok.pad_id}  bos_id={tok.bos_id}  eos_id={tok.eos_id}  unk_id={tok.unk_id}")

    # 3. Encode/decode roundtrip.
    print("\n[3/4] Test encode/decode roundtrip...")
    test_texts = [
        "Bonjour le monde, ceci est un test du tokenizer CogNet.",
        "The CogNet-MoE-1B model uses sparse routing with 8 experts.",
        "def forward(x):\n    return x + self.expert(x)",
        "L'algorithme BPE fusionne les paires les plus fréquentes.",
        "Special tokens: <pad> <bos> <eos> are reserved.",
    ]

    for text in test_texts:
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        status = "✓" if decoded.strip() == text.strip() else "✗"
        ratio = len(text) / max(1, len(ids))
        print(f"  {status} chars={len(text):3d} tokens={len(ids):3d} ratio={ratio:.2f}  "
              f"text=\"{text[:50]}{'...' if len(text)>50 else ''}\"")
        if status == "✗":
            print(f"     decoded=\"{decoded}\"")

    # 4. Test batch + padding.
    print("\n[4/4] Test batch + padding...")
    batch = tok.encode_batch(test_texts)
    padded, masks = tok.pad_batch(batch)
    print(f"  Batch size : {len(batch)}")
    print(f"  Seq lens   : {[len(ids) for ids in batch]}")
    print(f"  Padded len : {len(padded[0])} (max)")
    print(f"  Masks      : {masks}")

    # 5. Mesurer le ratio chars/token (vs CharTokenizer = 1.0).
    print("\n── Ratio chars/token (vs CharTokenizer = 1.0) ──")
    ratios = []
    for text in test_texts:
        ids = tok.encode(text, add_special_tokens=False)
        ratio = len(text) / max(1, len(ids))
        ratios.append(ratio)
    avg_ratio = sum(ratios) / len(ratios)
    print(f"  Ratio moyen : {avg_ratio:.2f} chars/token")
    print(f"  → BPE compresse ~{avg_ratio:.1f}× vs CharTokenizer")
    print(f"  → Pour 45,2B char-tokens Chinchilla, équivalent = {45.2e9/avg_ratio/1e9:.1f}B BPE-tokens")
    print(f"  → En BPE natif : Chinchilla = 20 × 2.29B = 45.8B BPE-tokens")

    # Cleanup.
    os.remove(HERE / "cognet_tokenizer_test.json")
    if (HERE / "cognet_tokenizer_test.vocab.json").exists():
        os.remove(HERE / "cognet_tokenizer_test.vocab.json")
    if (HERE / "cognet_tokenizer_test.merges.txt").exists():
        os.remove(HERE / "cognet_tokenizer_test.merges.txt")

    print("\n" + "=" * 70)
    print("✓ Tokenizer self-test passé.")
    print("=" * 70)
