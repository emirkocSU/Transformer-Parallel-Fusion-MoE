# ==================================================================================================
#  A/B/C MoE FUSION DENEYİ — TEŞHİS (DIAG) AŞAMASI, TEK HÜCRE  (Google Colab, A100)
#  Soru: pilotta B (paralel) neden A'yı (seri) 0,18 nat geçti? 4 grup x (A,B) pilot bütçesinde (25M token):
#  seed42 (yeniden temel) | dense kontrol | warm-up %10 | seed43  + router kararlılığı ölçümü. ~1,7 saat.
#  Sonuç: /content/moe_fusion_runs/DIAG_REPORT.md (önceden kayıtlı R1-R4 kuralları). MAIN'den ÖNCE çalıştırın.
#  Çalışma zamanı: Runtime > Change runtime type > A100 GPU.  Hücreyi bir kez çalıştırın.
#  Kod zip'i (moe_fusion_experiment.zip, sürüm aşağıda) bulunamazsa Colab yükleme penceresi açılır.
#  Veri /content'te yoksa Drive yedeğinden (MyDrive/moe_fusion_experiment/moe_data) otomatik geri yüklenir.
#  Bağlantı koparsa hücreyi tekrar çalıştırın: tamamlanan aşamalar ve TAMAMLANAN koşular atlanır;
#  yarıda kalan koşu (yeni oturumda) baştan başlar. Tarayıcı sekmesini açık tutun.
# ==================================================================================================
RUN_MODE = "diag"
DATA_DIR = "/content/moe_data"     # mevcut veri klasörünüz (train.npy, validation.npy, tokenizer.json, ...)
PROJECT_DIR = "/content/moe_fusion_experiment"
OUT_DIR = "/content/moe_fusion_runs"
USE_DRIVE = True                   # sonuçları (checkpoint hariç) Google Drive'a yedekler
DRIVE_DIR = "/content/drive/MyDrive/moe_fusion_experiment"
BACKUP_DATA_TO_DRIVE = True        # veri setini bir kez Drive'a kopyalar (sonraki oturumlar için)
ALLOW_DATA_REBUILD = True          # veri yerelde ve Drive'da yoksa: sabit FineWeb-Edu sürümünden yeniden üret
                                   # (orijinal manifestin sha256'larıyla bit düzeyinde karşılaştırılır)
FORCE_REUPLOAD = False             # True: mevcut kodu silip zip'i yeniden yükle
EXPECTED_VERSION = "1.4.2"

import glob, os, re, shutil, subprocess, sys, time, zipfile

print("TEŞHİS planı: 8 koşu x 384 adım x 65.536 token (pilot bütçesi); micro-batch 16 (pilotla aynı). Tahmini ~1,7 saat.")

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
_VER_RE = re.compile(r'__version__\s*=\s*"([^"]+)"')

def _version(d):
    p = os.path.join(d, "src", "moefusion", "__init__.py")
    if not os.path.exists(p):
        return None
    m = _VER_RE.search(open(p, encoding="utf-8").read())
    return m.group(1) if m else None

def _zip_version(z):
    """Sürümü zip'i AÇMADAN okur; bozuk/ilgisiz zip -> None."""
    try:
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if n.replace("\\", "/").endswith("src/moefusion/__init__.py"):
                    m = _VER_RE.search(zf.read(n).decode("utf-8"))
                    return m.group(1) if m else None
    except Exception:
        return None
    return None

def _extract(z):
    tmp = "/content/_moe_extract_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(tmp)
    roots = [r for r, _, fs in os.walk(tmp) if r.endswith(os.path.join("src", "moefusion")) and "__init__.py" in fs]
    if len(roots) != 1:
        raise SystemExit(f"Zip yapısı tanınmadı ({z}): src/moefusion bulunamadı")
    proj = os.path.dirname(os.path.dirname(roots[0]))
    shutil.rmtree(PROJECT_DIR, ignore_errors=True)
    shutil.move(proj, PROJECT_DIR)
    shutil.rmtree(tmp, ignore_errors=True)

if FORCE_REUPLOAD and os.path.isdir(PROJECT_DIR):
    shutil.rmtree(PROJECT_DIR)
if _version(PROJECT_DIR) != EXPECTED_VERSION:
    cands = glob.glob("/content/*.zip")
    if USE_DRIVE:
        cands += glob.glob(os.path.join(DRIVE_DIR, "*.zip"))
    good = []
    for z in sorted(set(cands), key=os.path.getmtime, reverse=True):
        v = _zip_version(z)
        if v == EXPECTED_VERSION:
            good.append(z)
        elif v is not None:
            print(f"  atlandı (eski sürüm {v}): {z}")
    if good:
        zpath = good[0]
    else:
        from google.colab import files
        print(f"Lütfen moe_fusion_experiment.zip (sürüm {EXPECTED_VERSION}) dosyasını seçin ...")
        up = files.upload()
        zips = [os.path.abspath(k) for k in up if k.endswith(".zip")]
        if not zips:
            raise SystemExit("Zip yüklenmedi.")
        zpath = zips[0]
        if _zip_version(zpath) != EXPECTED_VERSION:
            raise SystemExit(f"Yüklenen zip sürümü {_zip_version(zpath)}, beklenen {EXPECTED_VERSION}. "
                             "En son gönderilen zip'i yükleyin.")
    print("Kod zip'i:", zpath, "sürüm", _zip_version(zpath))
    _extract(zpath)
    if _version(PROJECT_DIR) != EXPECTED_VERSION:
        raise SystemExit(f"Çıkarılan kod sürümü {_version(PROJECT_DIR)} != {EXPECTED_VERSION}")
    if USE_DRIVE and not zpath.startswith(DRIVE_DIR):
        shutil.copy2(zpath, os.path.join(DRIVE_DIR, "moe_fusion_experiment.zip"))  # eski kopyanın üzerine yazar
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

# ---- 5) Pipeline: ortam -> veri (Drive'dan geri yükleme + sha256 doğrulama) -> testler -> bütçe -> bellek probu ->
#         smoke -> eğitimler (MODELS sırasıyla) -> karar/analiz/grafik/rapor -> zip --------------------------
#         (MAIN'de benchmark/profil yapılmaz: pilotta ölçüldü, token sayısından bağımsız) -----------------
cmd = [sys.executable, "scripts/run_diagnostics.py", "--data-dir", DATA_DIR, "--out", OUT_DIR]
if USE_DRIVE:
    cmd += ["--drive-dir", DRIVE_DIR]
    if BACKUP_DATA_TO_DRIVE:
        cmd += ["--backup-data-to-drive"]
if ALLOW_DATA_REBUILD:
    cmd += ["--allow-data-rebuild"]
t0 = time.time()
proc = subprocess.Popen(cmd, cwd=PROJECT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                        env=dict(os.environ, PYTHONUNBUFFERED="1"))
for line in proc.stdout:
    print(line, end="")
rc = proc.wait()
print(f"\nPipeline çıkış kodu {rc}  ({(time.time() - t0) / 60:.1f} dk)")

# ---- 6) Sonuçları göster ve paketle --------------------------------------------------------------------
import shutil as _sh
from IPython.display import Image, Markdown, display

drep = os.path.join(OUT_DIR, "DIAG_REPORT.md")
if os.path.exists(drep):
    display(Markdown(open(drep, encoding="utf-8").read()))
    for _t in ("diag_seed42", "diag_dense", "diag_warmup10", "diag_seed43"):
        _p = os.path.join(OUT_DIR, _t, "plots", "loss_vs_tokens.png")
        if os.path.exists(_p):
            print(_t)
            display(Image(_p))
    _tmp = "/content/_diag_pack"
    _sh.rmtree(_tmp, ignore_errors=True)
    os.makedirs(_tmp)
    for _t in ("diag_seed42", "diag_dense", "diag_warmup10", "diag_seed43"):
        if os.path.isdir(os.path.join(OUT_DIR, _t)):
            _sh.copytree(os.path.join(OUT_DIR, _t), os.path.join(_tmp, _t), ignore=_sh.ignore_patterns("*.pt", "*.tmp"))
    for _f in ("DIAG_REPORT.md", "DIAG_SUMMARY.json"):
        _sh.copy2(os.path.join(OUT_DIR, _f), _tmp)
    zp = _sh.make_archive(os.path.join(OUT_DIR, "moe_fusion_diag_results"), "zip", _tmp)
    _sh.rmtree(_tmp, ignore_errors=True)
    if USE_DRIVE:
        _sh.copy2(zp, DRIVE_DIR)
    print("Sonuç arşivi:", zp, "(Drive kopyası:", DRIVE_DIR if USE_DRIVE else "yok", ")")
    try:
        from google.colab import files
        files.download(zp)
    except Exception:
        pass
if rc != 0:
    print("Teşhis durdu. Yukarıdaki 'STOPPED' satırı nedeni gösterir; hücreyi tekrar çalıştırın (tamamlananlar atlanır).")
