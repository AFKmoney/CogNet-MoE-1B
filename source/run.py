#!/usr/bin/env python3
"""
CogNet-1B — Lanceur d'entraînement Python pur
===============================================
Remplace acil_submit.sh — tout est en Python !
Détecte les GPUs automatiquement, prépare les données,
lance l'entraînement multi-GPU avec torchrun si nécessaire.

Usage:
    # Simple — tout automatique
    python run.py

    # Avec options
    python run.py --max-steps 100000 --batch-size 4 --hf-token hf_xxxx

    # Reprendre un checkpoint
    python run.py --resume ./checkpoints_1b/cognet_1b_latest.pt

    # Seulement préparer les données
    python run.py --prep-only

    # Sur un cluster avec SLURM (soumission auto)
    python run.py --slurm --time 72:00:00 --gpus 4
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import json
import shutil
from datetime import datetime
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
#  Configuration par défaut
# ═══════════════════════════════════════════════════════════════════

DEFAULTS = {
    'model_size': '1b',
    'batch_size': 4,
    'grad_accum': 8,
    'seq_len': 512,
    'max_lr': 1e-4,
    'min_lr': 1e-5,
    'warmup_steps': 2000,
    'max_steps': 100000,
    'ckpt_dir': './checkpoints_1b',
    'data_dir': './data_1b',
    'save_every': 2000,
    'eval_every': 500,
    'log_every': 50,
    'weight_decay': 0.1,
    'grad_clip': 1.0,
}

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(WORKSPACE, 'train_ultra.py')


# ═══════════════════════════════════════════════════════════════════
#  Détection GPU
# ═══════════════════════════════════════════════════════════════════

def detect_gpus():
    """Détecte le nombre de GPUs disponibles."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return 0, []
        lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
        gpus = []
        for line in lines:
            parts = line.split(',')
            name = parts[0].strip()
            vram = float(parts[1].strip()) if len(parts) > 1 else 0
            gpus.append({'name': name, 'vram_mb': vram})
        return len(gpus), gpus
    except Exception:
        # Fallback: torch
        try:
            import torch
            count = torch.cuda.device_count()
            gpus = []
            for i in range(count):
                name = torch.cuda.get_device_name(i)
                vram = torch.cuda.get_device_properties(i).total_mem / 1e6  # MB
                gpus.append({'name': name, 'vram_mb': vram})
            return count, gpus
        except Exception:
            return 0, []


def get_gpu_type(gpus):
    """Retourne le type de GPU (A100, H100, etc.)."""
    if not gpus:
        return 'CPU'
    name = gpus[0]['name'].upper()
    if 'H100' in name:
        return 'H100'
    elif 'A100' in name:
        return 'A100'
    elif 'A6000' in name:
        return 'A6000'
    elif '4090' in name:
        return 'RTX4090'
    elif '3090' in name:
        return 'RTX3090'
    elif 'V100' in name:
        return 'V100'
    return gpus[0]['name']


# NOTE: Les estimations de temps seront calculées dynamiquement
# par le vrai benchmark au début du training dans train_ultra.py.
# Plus aucune estimation fabriquée ici.


# ═══════════════════════════════════════════════════════════════════
#  Préparation des données (Python)
# ═══════════════════════════════════════════════════════════════════

def prepare_data_python(data_dir, hf_token='', skip=False):
    """Lance la préparation des données via train_ultra.py."""
    if skip:
        print('[DATA] Skip (--skip-data-prep)')
        return True

    merged = os.path.join(data_dir, 'train_merged.pt')
    if os.path.exists(merged):
        size_mb = os.path.getsize(merged) / 1e6
        print(f'[DATA] Déjà préparé: {merged} ({size_mb:.0f} MB)')
        return True

    print('[DATA] Préparation des datasets (HF + AICL + synthetic)...')
    env = os.environ.copy()
    if hf_token:
        env['HF_TOKEN'] = hf_token

    cmd = [sys.executable, TRAIN_SCRIPT, '--max-steps', '0', '--skip-data-prep']
    # Note: --max-steps 0 avec --skip-data-prep ne fait rien
    # On doit lancer sans --skip-data-prep pour que la data prep se fasse
    cmd = [sys.executable, TRAIN_SCRIPT, '--max-steps', '0']

    try:
        result = subprocess.run(cmd, env=env, cwd=WORKSPACE, timeout=7200)  # 2h max
        if result.returncode != 0:
            print(f'[DATA] ERREUR: data prep a échoué (code {result.returncode})')
            return False
    except subprocess.TimeoutExpired:
        print('[DATA] ERREUR: data prep a timeout (2h)')
        return False
    except Exception as e:
        print(f'[DATA] ERREUR: {e}')
        return False

    if os.path.exists(merged):
        size_mb = os.path.getsize(merged) / 1e6
        print(f'[DATA] Préparation terminée: {merged} ({size_mb:.0f} MB)')
        return True

    print('[DATA] ERREUR: fichier merged non trouvé après préparation')
    return False


# ═══════════════════════════════════════════════════════════════════
#  Vérification des dépendances
# ═══════════════════════════════════════════════════════════════════

def check_dependencies():
    """Vérifie que les dépendances Python sont installées."""
    required = ['torch', 'datasets', 'huggingface_hub', 'tokenizers']
    missing = []

    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    # Vérification optionnelle
    optional_missing = []
    try:
        import bitsandbytes
    except ImportError:
        optional_missing.append('bitsandbytes (optionnel: 8-bit optimizer)')

    return missing, optional_missing


def install_dependencies(packages):
    """Installe les packages manquants."""
    for pkg in packages:
        print(f'[INSTALL] Installation de {pkg}...')
        subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '-q'], check=False)


# ═══════════════════════════════════════════════════════════════════
#  Lancement de l'entraînement
# ═══════════════════════════════════════════════════════════════════

def launch_training(args, num_gpus):
    """Lance l'entraînement — torchrun si multi-GPU, sinon python direct."""

    # Construction des arguments communs
    common_args = [
        '--model-size', str(args.model_size),
        '--batch-size', str(args.batch_size),
        '--grad-accum', str(args.grad_accum),
        '--seq-len', str(args.seq_len),
        '--max-lr', str(args.max_lr),
        '--min-lr', str(args.min_lr),
        '--warmup-steps', str(args.warmup_steps),
        '--max-steps', str(args.max_steps),
        '--ckpt-dir', str(args.ckpt_dir),
        '--save-every', str(args.save_every),
        '--eval-every', str(args.eval_every),
        '--log-every', str(args.log_every),
        '--weight-decay', str(args.weight_decay),
        '--grad-clip', str(args.grad_clip),
    ]

    # Optimisations V2 — toutes activées par défaut
    if args.bf16:
        common_args.append('--bf16')
    if args.compile:
        common_args.append('--compile')
    if args.cuda_prefetch:
        common_args.append('--cuda-prefetch')
    if args.seq_warmup:
        common_args.append('--seq-warmup')
    if args.async_ckpt:
        common_args.append('--async-ckpt')
    if args.use_8bit:
        common_args.append('--8bit-optim')

    # Resume
    if args.resume:
        common_args.extend(['--resume', args.resume])

    # Skip data prep (déjà fait)
    common_args.append('--skip-data-prep')

    # Environnement
    env = os.environ.copy()
    if args.hf_token:
        env['HF_TOKEN'] = args.hf_token
    env['COGNET_WORKSPACE'] = WORKSPACE
    env['AICL_REPEAT'] = str(args.aicl_repeat)

    # CUDA optimizations
    env['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
    env['TORCH_NCCL_AVOID_RECORD_STREAMS'] = '1'
    if 'NCCL_P2P_LEVEL' not in env:
        env['NCCL_P2P_LEVEL'] = 'NVL'

    # Multi-GPU → torchrun
    if num_gpus > 1 and args.use_fsdp:
        common_args.append('--use-fsdp')

        cmd = [
            sys.executable, '-m', 'torch.distributed.run',
            '--standalone',
            f'--nproc_per_node={num_gpus}',
            TRAIN_SCRIPT,
        ] + common_args

        print(f'\n[TRAIN] Lancement FSDP avec {num_gpus} GPUs via torchrun...')
        print(f'[TRAIN] Commande: {" ".join(cmd[:8])}... ({" ".join(common_args[:6])}...)')

    # Single GPU → python direct
    else:
        if args.compile_step:
            common_args.append('--compile-step')

        cmd = [sys.executable, TRAIN_SCRIPT] + common_args

        print(f'\n[TRAIN] Lancement single GPU...')
        print(f'[TRAIN] Commande: {" ".join(cmd[:4])}... ({" ".join(common_args[:6])}...)')

    # Lancement
    start_time = time.time()
    try:
        process = subprocess.Popen(
            cmd, env=env, cwd=WORKSPACE,
            stdout=sys.stdout, stderr=sys.stderr,
        )

        # Gestion des signaux pour propager au sous-processus
        def forward_signal(signum, frame):
            process.send_signal(signum)

        signal.signal(signal.SIGTERM, forward_signal)
        signal.signal(signal.SIGINT, forward_signal)

        # Attendre la fin
        return_code = process.wait()
        elapsed = time.time() - start_time

        if return_code == 0:
            print(f'\n[TRAIN] Entraînement terminé avec succès! ({elapsed/3600:.1f}h)')
        else:
            print(f'\n[TRAIN] Entraînement terminé avec code {return_code} ({elapsed/3600:.1f}h)')

        return return_code == 0

    except KeyboardInterrupt:
        print('\n[TRAIN] Interruption clavier — checkpoint sauvegardé par train_ultra.py')
        return True
    except Exception as e:
        print(f'\n[TRAIN] ERREUR: {e}')
        return False


# ═══════════════════════════════════════════════════════════════════
#  Soumission SLURM (optionnel)
# ═══════════════════════════════════════════════════════════════════

def submit_slurm(args, num_gpus):
    """Soumet le job via SLURM — mais le script reste en Python!"""
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=cognet-1b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={num_gpus}
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --time={args.time}
#SBATCH --output=logs/cognet-%j.out
#SBATCH --error=logs/cognet-%j.err

cd {WORKSPACE}
{sys.executable} run.py {" ".join(get_run_args_for_slurm(args))}
"""
    script_path = os.path.join(WORKSPACE, '_slurm_submit.sh')
    os.makedirs(os.path.join(WORKSPACE, 'logs'), exist_ok=True)

    with open(script_path, 'w') as f:
        f.write(slurm_script)

    print(f'[SLURM] Soumission du job...')
    result = subprocess.run(['sbatch', script_path], capture_output=True, text=True)
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f'[SLURM] Job soumis: {job_id}')
        print(f'[SLURM] Logs: logs/cognet-{job_id}.out')
    else:
        print(f'[SLURM] ERREUR: {result.stderr}')
    os.remove(script_path)


def get_run_args_for_slurm(args):
    """Retourne les arguments Python pour la soumission SLURM."""
    arg_list = []
    if args.hf_token:
        arg_list.extend(['--hf-token', args.hf_token])
    arg_list.extend(['--max-steps', str(args.max_steps)])
    arg_list.extend(['--batch-size', str(args.batch_size)])
    arg_list.extend(['--grad-accum', str(args.grad_accum)])
    arg_list.extend(['--seq-len', str(args.seq_len)])
    if args.no_compile:
        arg_list.append('--no-compile')
    if args.no_fsdp:
        arg_list.append('--no-fsdp')
    return arg_list


# ═══════════════════════════════════════════════════════════════════
#  Vérification des checkpoints
# ═══════════════════════════════════════════════════════════════════

def check_existing_checkpoints(ckpt_dir):
    """Affiche les checkpoints existants."""
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        return None

    latest = ckpt_path / 'cognet_1b_latest.pt'
    best = ckpt_path / 'cognet_1b_best.pt'
    final = ckpt_path / 'cognet_1b_final.pt'

    info = {}
    if latest.exists():
        try:
            data = torch.load(str(latest), map_location='cpu', weights_only=False)
            info['latest_step'] = data.get('step', 0)
            info['latest_loss'] = data.get('loss', float('inf'))
            info['latest_path'] = str(latest)
        except Exception:
            pass
    if best.exists():
        try:
            data = torch.load(str(best), map_location='cpu', weights_only=False)
            info['best_step'] = data.get('step', 0)
            info['best_loss'] = data.get('best_loss', float('inf'))
            info['best_path'] = str(best)
        except Exception:
            pass
    if final.exists():
        info['final_path'] = str(final)

    return info


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='CogNet-1B — Lanceur Python (remplace acil_submit.sh)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python run.py                                    # Tout automatique
  python run.py --max-steps 50000                  # 50k steps
  python run.py --hf-token hf_xxx                  # Avec token HF
  python run.py --resume ./checkpoints_1b/cognet_1b_latest.pt  # Reprendre
  python run.py --prep-only                        # Seulement data prep
  python run.py --slurm --gpus 4 --time 72:00:00   # SLURM auto
  python run.py --no-fsdp                          # Single GPU
        """
    )

    # Config
    parser.add_argument('--model-size', type=str, default=DEFAULTS['model_size'], choices=['1b', '350m'])
    parser.add_argument('--batch-size', type=int, default=DEFAULTS['batch_size'])
    parser.add_argument('--grad-accum', type=int, default=DEFAULTS['grad_accum'])
    parser.add_argument('--seq-len', type=int, default=DEFAULTS['seq_len'])
    parser.add_argument('--max-lr', type=float, default=DEFAULTS['max_lr'])
    parser.add_argument('--min-lr', type=float, default=DEFAULTS['min_lr'])
    parser.add_argument('--warmup-steps', type=int, default=DEFAULTS['warmup_steps'])
    parser.add_argument('--max-steps', type=int, default=DEFAULTS['max_steps'])
    parser.add_argument('--ckpt-dir', type=str, default=DEFAULTS['ckpt_dir'])
    parser.add_argument('--data-dir', type=str, default=DEFAULTS['data_dir'])
    parser.add_argument('--save-every', type=int, default=DEFAULTS['save_every'])
    parser.add_argument('--eval-every', type=int, default=DEFAULTS['eval_every'])
    parser.add_argument('--log-every', type=int, default=DEFAULTS['log_every'])
    parser.add_argument('--weight-decay', type=float, default=DEFAULTS['weight_decay'])
    parser.add_argument('--grad-clip', type=float, default=DEFAULTS['grad_clip'])

    # Token & repos
    parser.add_argument('--hf-token', type=str, default=os.environ.get('HF_TOKEN', ''),
                        help='HuggingFace API token')
    parser.add_argument('--aicl-repeat', type=int, default=10,
                        help='Nombre de répétitions des données AICL')

    # Optimizations (activées par défaut)
    parser.add_argument('--no-compile', action='store_true', help='Désactiver torch.compile')
    parser.add_argument('--no-fsdp', action='store_true', help='Désactiver FSDP (single GPU)')
    parser.add_argument('--no-cuda-prefetch', action='store_true', help='Désactiver CUDA prefetch')
    parser.add_argument('--no-seq-warmup', action='store_true', help='Désactiver seq length warmup')
    parser.add_argument('--no-async-ckpt', action='store_true', help='Désactiver async checkpointing')
    parser.add_argument('--no-bf16', action='store_true', help='Désactiver BF16 (utiliser FP16)')
    parser.add_argument('--8bit', action='store_true', help='Activer 8-bit optimizer (bitsandbytes)')
    parser.add_argument('--compile-step', action='store_true', help='Compiler forward+backward ensemble')

    # Resume
    parser.add_argument('--resume', type=str, default=None, help='Chemin du checkpoint à reprendre')

    # Modes spéciaux
    parser.add_argument('--prep-only', action='store_true', help='Seulement préparer les données')
    parser.add_argument('--skip-data-prep', action='store_true', help='Sauter la préparation des données')
    parser.add_argument('--check-only', action='store_true', help='Seulement vérifier le setup')

    # SLURM
    parser.add_argument('--slurm', action='store_true', help='Soumettre via SLURM')
    parser.add_argument('--gpus', type=int, default=None, help='Nombre de GPUs pour SLURM')
    parser.add_argument('--time', type=str, default='72:00:00', help='Temps SLURM')

    args = parser.parse_args()

    # Dériver les flags booléens (inversés car les flags sont "no-*")
    args.bf16 = not args.no_bf16
    args.compile = not args.no_compile
    args.use_fsdp = not args.no_fsdp
    args.cuda_prefetch = not args.no_cuda_prefetch
    args.seq_warmup = not args.no_seq_warmup
    args.async_ckpt = not args.no_async_ckpt
    args.use_8bit = getattr(args, '8bit', False)

    # ═══ Bannière ═══
    print()
    print('╔══════════════════════════════════════════════════════════╗')
    print('║       CogNet-1B — Lanceur Python V2                     ║')
    print('║       Les performances seront mesurées par benchmark     ║')
    print('╚══════════════════════════════════════════════════════════╝')
    print()

    # ═══ Détection GPU ═══
    num_gpus, gpus = detect_gpus()
    gpu_type = get_gpu_type(gpus)

    print(f'[GPU] {num_gpus} GPU(s) détecté(s):')
    for i, gpu in enumerate(gpus):
        print(f'  GPU {i}: {gpu["name"]} ({gpu["vram_mb"]:.0f} MB VRAM)')
    print(f'  Type: {gpu_type}')

    if num_gpus == 0:
        print('[GPU] ATTENTION: Aucun GPU détecté — entraînement sur CPU (très lent!)')
        print('[GPU] Vérifiez que nvidia-smi fonctionne et que CUDA est installé')

    # ═══ Vérification dépendances ═══
    missing, optional = check_dependencies()
    if missing:
        print(f'\n[DEPS] Packages manquants: {", ".join(missing)}')
        response = input('[DEPS] Installer automatiquement? (o/n) [o] ').strip().lower()
        if response in ('', 'o', 'oui', 'y', 'yes'):
            install_dependencies(missing)
        else:
            print('[DEPS] Installation annulée. Installez manuellement:')
            print(f'  pip install {" ".join(missing)}')
            sys.exit(1)

    if optional:
        print(f'[DEPS] Optionnels non installés: {", ".join(optional)}')

    # ═══ Vérification du script d'entraînement ═══
    if not os.path.exists(TRAIN_SCRIPT):
        print(f'[ERREUR] Script d\'entraînement introuvable: {TRAIN_SCRIPT}')
        sys.exit(1)

    if not os.path.exists(os.path.join(WORKSPACE, 'cognet_1b_optimized.py')):
        print(f'[ERREUR] Modèle optimisé introuvable: cognet_1b_optimized.py')
        sys.exit(1)

    # ═══ Checkpoints existants ═══
    ckpt_info = check_existing_checkpoints(args.ckpt_dir)
    if ckpt_info:
        print(f'\n[CKPT] Checkpoints existants dans {args.ckpt_dir}:')
        if 'latest_step' in ckpt_info:
            print(f'  Latest: step {ckpt_info["latest_step"]}, loss={ckpt_info["latest_loss"]:.4f}')
        if 'best_step' in ckpt_info:
            print(f'  Best:   step {ckpt_info["best_step"]}, loss={ckpt_info["best_loss"]:.4f}')

    else:
        print(f'\n[CKPT] Aucun checkpoint existant')

    # ═══ Estimation du temps ═══
    # NOTE: Le vrai benchmark sera fait par train_ultra.py au début du training.
    # Pas d'estimation fabriquée ici — les chiffres réels seront mesurés.
    if num_gpus > 0 and not args.check_only:
        effective_batch = args.batch_size * args.grad_accum * num_gpus
        print(f'\n[BENCH] Les performances seront mesurées par un vrai benchmark au démarrage.')
        print(f'  GPU: {num_gpus}x {gpu_type}')
        print(f'  Batch effectif: {effective_batch} ({args.batch_size} x {args.grad_accum} x {num_gpus} GPUs)')
        print(f'  Le temps restant sera calculé à partir de la vitesse mesurée.')

    # ═══ Config finale ═══
    print(f'\n[CONFIG] Configuration finale:')
    print(f'  Model:    CogNet-{args.model_size.upper()} (16 blocks, 8 channels, 384 ch_dim, 8192 ff)')
    print(f'  Vocab:    136 (CharTokenizer)')
    print(f'  Seq len:  {args.seq_len}')
    print(f'  Batch:    {args.batch_size} x grad_accum={args.grad_accum} x GPUs={num_gpus} = {args.batch_size * args.grad_accum * num_gpus}')
    print(f'  LR:       {args.min_lr} → {args.max_lr}')
    print(f'  Steps:    {args.max_steps:,}')
    print(f'  HF token: {"SET" if args.hf_token else "NOT SET"}')
    print(f'  BF16:     {args.bf16}')
    print(f'  Compile:  {args.compile}')
    print(f'  FSDP:     {args.use_fsdp} ({num_gpus} GPUs)')
    print(f'  Prefetch: {args.cuda_prefetch}')
    print(f'  SeqWarm:  {args.seq_warmup}')
    print(f'  AsyncCkpt:{args.async_ckpt}')
    print(f'  8-bit:    {args.use_8bit}')

    # ═══ Check-only ═══
    if args.check_only:
        print('\n[CHECK] Vérification terminée — tout est prêt!')
        return

    # ═══ SLURM ═══
    if args.slurm:
        gpu_count = args.gpus or num_gpus or 4
        submit_slurm(args, gpu_count)
        return

    # ═══ Data prep ═══
    if args.prep_only:
        ok = prepare_data_python(args.data_dir, args.hf_token, skip=False)
        print('\n[DATA] Préparation terminée!' if ok else '\n[DATA] ÉCHEC!')
        return

    if not args.skip_data_prep:
        ok = prepare_data_python(args.data_dir, args.hf_token)
        if not ok:
            print('[DATA] ÉCHEC de la préparation des données!')
            response = input('[DATA] Continuer quand même? (o/n) [n] ').strip().lower()
            if response not in ('o', 'oui', 'y', 'yes'):
                sys.exit(1)

    # ═══ Entraînement ═══
    print('\n' + '=' * 60)
    print('  DÉMARRAGE DE L\'ENTRAÎNEMENT')
    print('=' * 60)
    print(f'  Début: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 60 + '\n')

    success = launch_training(args, num_gpus)

    print('\n' + '=' * 60)
    if success:
        print('  ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS')
    else:
        print('  ENTRAÎNEMENT TERMINÉ AVEC ERREURS')
    print(f'  Fin: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 60)

    # Vérifier le résultat final
    ckpt_info = check_existing_checkpoints(args.ckpt_dir)
    if ckpt_info and 'best_path' in ckpt_info:
        print(f'\n  Meilleur checkpoint: {ckpt_info["best_path"]}')
        if 'best_loss' in ckpt_info:
            print(f'  Meilleure loss: {ckpt_info["best_loss"]:.4f}')

    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
