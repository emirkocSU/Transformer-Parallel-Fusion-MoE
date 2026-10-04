# ==================================================================================================
#  A/B/C MoE FUSION DENEYİ — VERİDEN PİLOT SONUNA KADAR TEK HÜCRE  (Google Colab, A100)
#  Çalışma zamanı: Runtime > Change runtime type > A100 GPU.  Hücreyi bir kez çalıştırın.
#  Kod zip'i (moe_fusion_experiment.zip) bulunamazsa Colab yükleme penceresi açılır.
#  Bağlantı koparsa hücreyi tekrar çalıştırın: tamamlanan aşamalar/koşular atlanır.
# ==================================================================================================
RUN_MODE = "pilot"                 # "smoke" (yalnız doğrulama) | "pilot" (A/B/C ~25M token). MAIN bu hücreden başlatılmaz.
DATA_DIR = "/content/moe_data"     # mevcut veri klasörünüz (train.npy, validation.npy, tokenizer.json, ...)
PROJECT_DIR = "/content/moe_fusion_experiment"
OUT_DIR = "/content/moe_fusion_runs"
USE_DRIVE = True                   # sonuçları (checkpoint hariç) Google Drive'a yedekler
DRIVE_DIR = "/content/drive/MyDrive/moe_fusion_experiment"
BACKUP_DATA_TO_DRIVE = True        # veri setini bir kez Drive'a kopyalar (sonraki oturumlar için)
CHECKPOINT_LOCATION = "local"      # "local": devam checkpoint'i yerel diskte (koşu başına TEK dosya, 5.7-7.0 GB, koşu bitince silinir)
                                   # "drive": Drive'a yazılır -> bağlantı kopsa bile yeni oturumda kaldığı adımdan devam eder
FINAL_WEIGHTS_TO_DRIVE = True      # model_final_bf16.pt (~1 GB/koşu) yerel disk yerine Drive'a
ALLOW_DATA_REBUILD = True          # veri yerelde ve Drive'da yoksa: sabit FineWeb-Edu sürümünden yeniden üret
                                   # (orijinal manifestin sha256'larıyla bit düzeyinde karşılaştırılır)
FORCE_REUPLOAD = False             # True: mevcut kodu silip zip'i yeniden yükle
EXPECTED_VERSION = "1.2.0"

import glob, os, re, shutil, subprocess, sys, time, zipfile

assert RUN_MODE in ("smoke", "pilot"), "Bu hücre yalnızca smoke/pilot çalıştırır (yanlışlıkla büyük maliyet önlemi)."
print(f"RUN_MODE = {RUN_MODE}")

# ---- 1) GPU hızlı kontrol --------------------------------------------------------------------------
smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
print("GPU:", smi.stdout.strip() or "BULUNAMADI")
if smi.returncode != 0:
    raise SystemExit("GPU yok. Runtime > Change runtime type > A100 GPU seçin.")
if "A100" not in smi.stdout:
    print("UYARI: GPU A100 değil. Karşılaştırma yine adil olur (hepsi aynı GPU) ama hızlar A100 değerleri olmaz.")

# ---- 2) Google Drive (opsiyonel) -----------------------------------------------------------------
if USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        os.makedirs(DRIVE_DIR, exist_ok=True)
    except Exception as e:  # noqa: BLE001
        print("Drive bağlanamadı, yalnızca yerel diske yazılacak:", e)
        USE_DRIVE = False

# ---- 3) Proje kodu (zip) -------------------------------------------------------------------------
def _version(d):
    p = os.path.join(d, "src", "moefusion", "__init__.py")
    if not os.path.exists(p):
        return None
    m = re.search(r'__version__\s*=\s*"([^"]+)"', open(p).read())
    return m.group(1) if m else None

if FORCE_REUPLOAD and os.path.isdir(PROJECT_DIR):
    shutil.rmtree(PROJECT_DIR)
if _version(PROJECT_DIR) != EXPECTED_VERSION:
    cands = sorted(glob.glob("/content/moe_fusion_experiment*.zip"), key=os.path.getmtime, reverse=True)
    if USE_DRIVE:
        cands += glob.glob(os.path.join(DRIVE_DIR, "moe_fusion_experiment*.zip"))
    zpath = cands[0] if cands else None
    if zpath is None:
        from google.colab import files
        print("Lütfen moe_fusion_experiment.zip dosyasını seçin ...")
        up = files.upload()
        zips = [k for k in up if k.endswith(".zip")]
        if not zips:
            raise SystemExit("Zip yüklenmedi.")
        zpath = os.path.join("/content", zips[0])
    print("Kod zip'i:", zpath)
    if os.path.isdir(PROJECT_DIR):
        shutil.rmtree(PROJECT_DIR)
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
        if all(n.startswith("moe_fusion_experiment/") for n in names):
            zf.extractall("/content")
        else:
            zf.extractall(PROJECT_DIR)
    if _version(PROJECT_DIR) != EXPECTED_VERSION:
        raise SystemExit(f"Zip içeriği beklenen sürüm değil: {_version(PROJECT_DIR)} != {EXPECTED_VERSION}")
    if USE_DRIVE and not zpath.startswith(DRIVE_DIR):
        shutil.copy2(zpath, os.path.join(DRIVE_DIR, "moe_fusion_experiment.zip"))
print("Proje:", PROJECT_DIR, "sürüm", _version(PROJECT_DIR))

# ---- 4) Eksik bağımlılıklar (torch'a DOKUNULMAZ) --------------------------------------------------
need = []
for mod, pkg in (("yaml", "pyyaml"), ("tokenizers", "tokenizers"), ("matplotlib", "matplotlib"), ("pytest", "pytest"), ("psutil", "psutil")):
    try:
        __import__(mod)
    except Exception:
        need.append(pkg)
if ALLOW_DATA_REBUILD:
    for mod, pkg in (("pyarrow", "pyarrow"), ("huggingface_hub", "huggingface_hub")):
        try:
            __import__(mod)
        except Exception:
            need.append(pkg)
if need:
    print("Kuruluyor:", need)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], check=True)

# ---- 5) Pipeline: ortam -> veri doğrulama -> testler -> bütçe -> bellek probu -> smoke -> benchmark ->
#         profil -> A, B, C_same, C_matched pilot eğitimleri -> analiz/grafik/rapor -> zip ------------
cmd = [sys.executable, "scripts/run_pipeline.py", "--mode", RUN_MODE, "--data-dir", DATA_DIR, "--out", OUT_DIR]
if USE_DRIVE:
    cmd += ["--drive-dir", DRIVE_DIR]
    if BACKUP_DATA_TO_DRIVE:
        cmd += ["--backup-data-to-drive"]
    if FINAL_WEIGHTS_TO_DRIVE:
        cmd += ["--final-weights-to-drive"]
    cmd += ["--ckpt-location", CHECKPOINT_LOCATION]
elif CHECKPOINT_LOCATION == "drive":
    print("UYARI: Drive bağlı değil; checkpoint yerel diske yazılacak.")
if ALLOW_DATA_REBUILD:
    cmd += ["--allow-data-rebuild"]
t0 = time.time()
proc = subprocess.Popen(cmd, cwd=PROJECT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                        env=dict(os.environ, PYTHONUNBUFFERED="1"))
for line in proc.stdout:
    print(line, end="")
rc = proc.wait()
print(f"\nPipeline çıkış kodu {rc}  ({(time.time() - t0) / 60:.1f} dk)")

# ---- 6) Sonuçları göster ---------------------------------------------------------------------------
root = os.path.join(OUT_DIR, RUN_MODE)
from IPython.display import Image, Markdown, display

rep = os.path.join(root, "FINAL_REPORT.md")
if os.path.exists(rep):
    display(Markdown(open(rep, encoding="utf-8").read().split("## Plots")[0]))
    for png in ("loss_vs_wallclock", "loss_vs_tokens", "ppl_vs_wallclock", "step_time_comparison", "throughput_comparison",
                "vram_comparison", "expert_utilization", "benchmark_step_time"):
        p = os.path.join(root, "plots", f"{png}.png")
        if os.path.exists(p):
            display(Image(p))
zp = os.path.join(OUT_DIR, f"moe_fusion_{RUN_MODE}_results.zip")
if os.path.exists(zp):
    print("Sonuç arşivi:", zp, "(Drive kopyası:", DRIVE_DIR if USE_DRIVE else "yok", ")")
    try:
        from google.colab import files
        files.download(zp)
    except Exception:
        pass
if rc != 0:
    print("Pipeline durdu. Yukarıdaki 'PIPELINE STOPPED' satırı nedeni gösterir; düzeltip hücreyi tekrar çalıştırın.")
