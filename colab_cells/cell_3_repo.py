# ════════════════════════════════════════════════════════════════
# Cell 3: Clone repo + cài dependencies
# ════════════════════════════════════════════════════════════════
import os, subprocess, sys
from pathlib import Path

GITHUB_USERNAME = 'Bang334'
GITHUB_REPO = 'dr-diagnostic-system'
GITHUB_BRANCH = 'feat/integrate-grade-model-semi'
REPO_DIR = Path('/content') / GITHUB_REPO
REPO_URL = f'https://github.com/{GITHUB_USERNAME}/{GITHUB_REPO}.git'

if REPO_DIR.exists():
    print("[*] Đang tự động cập nhật code mới nhất từ GitHub...")
    subprocess.run(['git', '-C', str(REPO_DIR), 'fetch', 'origin', GITHUB_BRANCH], check=True)
    subprocess.run(['git', '-C', str(REPO_DIR), 'reset', '--hard', f'origin/{GITHUB_BRANCH}'], check=True)
else:
    subprocess.run(['git', 'clone', '-b', GITHUB_BRANCH, REPO_URL, str(REPO_DIR)], check=True)

os.chdir(REPO_DIR)
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-q',
    'pandas>=2.0', 'Pillow>=10.0', 'ipywidgets>=8.1', 'kaggle>=2.2.2'
], check=True)

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

print(f'[v] Đã sẵn sàng tại {REPO_DIR}')
