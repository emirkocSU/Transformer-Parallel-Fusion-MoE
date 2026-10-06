# Bulgular ve analiz (Türkçe)

> Bu belge deneyin tüm aşamalarının Türkçe analizidir. Rakamların hepsi [`results/`](../results/) altındaki otomatik
> üretilmiş raporlardan ve loglardan alınmıştır. İngilizce özet ve grafikler için [ana README](../README.md).

## 1. Soru

Modern MoE dil modellerinin çoğu (DeepSeek-V3, Kimi K2, GLM-4.5) seri blok kullanır: aynı katmandaki MoE,
attention'ın çıktısını görür. GPT-J ve PaLM ise iki alt katmanı aynı girdiden **paralel** çalıştırır.
Bu deneyin sorusu: paralel bir MoE bloğu bir şey kaybediyor mu ve her 4 bloktan sonra eklenen **FusionMoE**
katmanları bu kaybı geri kazanabilir mi?

| model | tanım | MoE katmanı | expert genişliği | token başına işlem |
|---|---|---|---|---|
| **A** | seri: `a = x + Attn(x)`, `y = a + MoE(a)` | 12 | 1920 | 1.068 GFLOPs |
| **B** | paralel: `y = x + Attn(x) + MoE(x)` | 12 | 1920 | 1.068 GFLOPs |
| **C_same** | B + 4., 8. ve 12. bloktan sonra FusionMoE | 12 + 3 | 1920 | 1.227 GFLOPs (+%15) |
| **C_matched** | C_same ile aynı yapı, expert'ler daraltılmış | 12 + 3 | 1536 | 1.068 GFLOPs (B ile aynı) |

Tüm modeller: 12 katman, d=768, 8 SwiGLU expert, top-2 yönlendirme. A ve B aynı başlangıç ağırlıklarına sahip.
FusionMoE "4 adımda bir" değil, **4 blokta bir** giriyor ve her ileri geçişte tüm token'lara uygulanıyor.

## 2. Aşamalar

| aşama | token / model | ne yapıldı | sonuç |
|---|---|---|---|
| Pilot | 25.2M | A, B, C_same, C_matched | B, A'dan 0.179 nats iyi (beklenmedik); C_same ≈ B; C_matched B'den kötü (+0.059) |
| Tanı (DIAG) | 25.2M | A ve B, 4 koşul | B > A farkı gerçek, MoE'ye özgü değil, warm-up yapaylığı değil |
| MAIN | 100M | dört model | fusion kararı: INCONCLUSIVE |

## 3. Tanı aşaması: paralel neden seriden iyi?

| kural | ölçüm | karar |
|---|---|---|
| R1: gerçek mi, seed gürültüsü mü? | seed 42: −0.172, seed 43: −0.119 | **sağlam** |
| R2: MoE'ye özgü mü? | dense FFN kontrolü: −0.136 | **MoE'ye özgü değil** |
| R3: kısa warm-up yapaylığı mı? | %10 warm-up: −0.197 (iki model de iyileşiyor, fark açılıyor) | **warm-up değil** |
| R4: router kararsızlığı mı? | ilk çeyrekte A'nın routing değişkenliği B'nin 1.24 / 1.19 katı | zayıf (1.5 eşiğinin altında) |

**Sonuç:** Fark bu eğitim rejimine (küçük ve az eğitilmiş model, ortak reçete) ait bir özellik; MoE yönlendirmesiyle
ilgili değil. İki seed arasındaki fark ilk kez ölçüldü: **0.053 nats**. Bu değer, tek seed'li MAIN için karar eşiği
oldu (Amendment 4): 0.053'ten küçük farklar "kesin" sayılmaz.

Ek gözlemler:
- Dense kontrolde, 25M token'da MoE henüz dense FFN'den iyi değil (paralelde eşit: 5.3298 ve 5.3305; seride dense
  0.036 önde). Erken eğitimde beklenen bir durum: her expert token'ların yalnızca ~1/4'ünü görüyor ve router başta
  kararsız.
- Seed 43'te B'nin kaybı son iki ölçüm arasında +0.011 yükseldi; gradyan ve router stabildi. Açıklanamayan, tek
  koşuluk bir sapma. 0.053 eşiği bu koşuyu da içeren kötümser değer olarak seçildi.

## 4. MAIN (100M token)

| model | final kayıp | perplexity | eğitim süresi | token/s |
|---|---|---|---|---|
| A seri | 4.1564 | 63.84 | 32.3 dk | 51,562 |
| B paralel | 4.0672 | 58.40 | 32.3 dk | 51,570 |
| C_same | **4.0379** | **56.71** | 37.7 dk | 44,228 |
| C_matched | 4.0651 | 58.27 | 35.2 dk | 47,326 |

| karşılaştırma | ΔL [%95 GA] | iyileşen dizi | önceden kayıtlı karar |
|---|---|---|---|
| B − A | −0.0891 [−0.0899, −0.0883] | %99.6 | daha iyi |
| C_same − B | −0.0293 [−0.0297, −0.0290] | %92.2 | sonuçsuz (0.02–0.053 arası) |
| C_matched − B | −0.0021 [−0.0025, −0.0018] | %54.3 | yaklaşık eşit |
| C_matched − C_same | +0.0272 [+0.0268, +0.0276] | %10.6 | sonuçsuz |

### 4.1 Farkların zaman içindeki seyri

| token | 4M | 25M | 50M | 67M | 100M |
|---|---|---|---|---|---|
| B − A | −0.232 | −0.132 | −0.130 | −0.116 | −0.088 |
| C_same − B | +0.004 | −0.007 | −0.022 | −0.027 | −0.029 |
| C_matched − C_same | +0.027 | +0.025 | +0.028 | +0.027 | +0.026 |
| C_matched − B | +0.031 | +0.017 | +0.006 | 0.000 | −0.002 |

(Ara değerler 2,048 dizilik periyodik alt kümeden, final değerler 15,088 dizinin tamamından.)

## 5. Seri ve paralel: aynı sürede bitip seri neden daha kötü?

**Hesap azaltılmıyor.** A ve B'nin parametre sayısı (477.654M), token başına işlemi (1.068 GFLOPs) ve medyan adım
süresi (1269.3 ms) birebir aynı. Paralel blok aynı attention ve MoE işlemlerini yapıyor; tek fark MoE'nin girdisi
(A'da `x + Attn(x)`, B'de `x`).

**Hızlanma neden yok?** PaLM'daki ~%15 hızlanma iki dalın aynı anda çalışmasından değil, attention ve MLP girdi
matris çarpımlarının tek bir matmul'da birleştirilmesinden geliyor. Bizde:
- Pilotta ayrı CUDA stream'leri ölçüldü; kazanç ≤ %1, çünkü bu boyutta her kernel A100'ü zaten dolduruyor.
- MoE'de FFN girdisi router ve expert'e özel ağırlıklardan geçtiği için PaLM tipi birleştirme doğrudan uygulanamıyor.

**Seri neden kötü?** Hesap eksikliği değil, öğrenme dinamiği. Paralel model erken dönemde daha hızlı öğreniyor; fark
−0.23'ten −0.089'a daralıyor ve B, A'nın final kalitesine token'ların %77'sinde ulaşıyor. Bu, GPT-J/NeoX ve PaLM'ın
"büyük ölçekte fark kalmıyor" bulgularıyla çelişmiyor. Farkın ne zaman kapanacağını bu koşulardan bilemeyiz.

## 6. Fusion (C) ve B: asıl soru

### Kalite
- **C_same**, yani B'ye hiçbir şey eksiltmeden 3 MoE katmanı eklenmiş model, B'den 0.029 nats iyi ve dizilerin
  %92'sinde iyileşme var. Ama bu fark yaklaşık **20M token'dan sonra** ortaya çıkıyor ve 65M token'dan sonra
  −0.028 civarında sabitleniyor. 25M token'da duran pilot bu yüzden farkı göremedi.
- **C_matched "B + ekstra MoE" değil.** Ana 12 katmanın expert'leri 1920'den 1536'ya daraltılıyor ve açılan bütçe 3
  fusion katmanına veriliyor; toplam parametre ve işlem B ile aynı.
- **C_matched − B = (daraltma maliyeti) + (fusion kazancı).** Daraltma maliyeti baştan sona +0.027 civarında sabit.
  Fusion kazancı 20M token'dan sonra büyüyor. İkisi 100M token'da birbirini tam karşılıyor; sonuç eşitlik (−0.002).
  Pilottaki +0.059, bu dengeye ulaşılmadan önceki erken dönem değeriydi.

### Verimlilik
| | B | C_same | C_matched |
|---|---|---|---|
| eğitim süresi | 32.3 dk | 37.7 dk (+%17) | 35.2 dk (+%9) |
| B'nin bittiği anda (32.3 dk) kayıp | **4.0656** | 4.0821 | 4.0857 |
| B'nin final kalitesine ulaştığı nokta | 100M token | 90.0M token (~%3 fazla FLOPs) | 99.0M token |

- C_matched aynı FLOPs ile %9 daha yavaş (12 yerine 15 MoE dağıtımı, daha dar matmul'lar) ve kalite aynı: bu
  uygulamada B ondan her açıdan iyi.
- C_same, B'nin kalitesine yaklaşık %3 fazla FLOPs ve %5 fazla süreyle ulaşıyor. Kosinüs takviminin ara noktaları
  yavaş modelin aleyhine olduğu için gerçek tablo biraz daha iyi olabilir; yine de C lehine bir kazanç yok.
- Fırsat maliyeti tahmini: B'nin pilot (25M) ve MAIN (100M) kayıplarına iki noktalı kuvvet yasası oturtunca, aynı
  %15 ek hesabı **daha fazla token** olarak B'ye vermek ~0.09 nats kazandırıyor (kaba tahmin). Fusion'ın kazancı
  0.029. Bu, Amendment 5'in ilk adımında doğrudan ölçülecek.

### Cevap
- Kalitede (ek hesapla): zayıf-orta düzeyde bir emare var; küçük, tutarlı ama tekrarlanmamış.
- Eşit hesapta: kazanç yok.
- Verimlilikte: fusion lehine emare yok, mevcut kanıtlar aleyhte.
- "B etkileşim kaybeder, C geri kazanır" varsayımı bu rejimde geçerli değil, çünkü B A'ya karşı kaybetmiyor. C_same'in
  kazancının en basit açıklaması ek derinlik ve kapasite; C_matched'in sonucu da bunu destekliyor.

## 7. Akla gelen fikirler ve değerlendirmesi

**"Erken dönemde B ile başlasak, fusion'ı sonra eklesek?"** Fikir, literatürde model büyütme (Net2Net, progressive
stacking, staged training, sparse upcycling) olarak var. Ancak bizde C_same erken dönemde B'den kötü değildi (+0.005),
yani kaçınılacak bir kayıp yok; tasarruf yalnızca ilk %25'teki fusion hesabı (~%3). Geç eklenen katmanlar daha az
eğitim alır ve final kazanç büyük ihtimalle −0.029'dan küçük olur. Beklenen fark tek seed eşiğinin altında. Bu yüzden
yalnızca iso-FLOP testi C_same lehine çıkarsa denenmeli. Yeni katman, eklendiği anda modelin çıktısını değiştirmemek
için sıfır başlatmalı çıkış projeksiyonuyla eklenmeli. (L1/L2'yi birlikte kullanma benzetmesi kısmen yerinde: ikisi
de rejime göre en iyi davranışı seçiyor. Ama Huber durumsuzdur; mimari geçişin ise optimizer durumu, öğrenme oranı
takvimi ve eğitilmemiş parametreler gibi bir maliyeti var.)

**"3 fusion katmanı fazla mı, 1'e düşürsek?"** Kod bunu destekliyor: `fusion_interval: 8` (8. bloktan sonra) veya
`final_only` (12. bloktan sonra). Tek katman ~%5 ek işlem demek. Verimli sayılabilmesi için mevcut kazancın büyük
kısmını (~0.02–0.03) tek başına vermesi gerekir. Hangi konumun değerli olduğu önce eğitilmiş C_same üzerinde
katman çıkarma (knock-out) analiziyle ölçülmeli.

**"Daha uzun eğitim (200M+ token) daha iyi sonuç verir mi?"** C_same − B 65M token'dan sonra sabitlendi ve
daraltma maliyeti baştan beri sabit; uzun koşunun beklenen bilgi getirisi düşük. 100M token'lık bir koşu, kosinüs
takvimi bittiği için 200M'ye uzatılamaz; dört modelin baştan koşulması gerekir.

## 8. Sınırlamalar

- Küçük ölçek: ~160M aktif parametre, 100M token (parametre başına ~0.6 token; Chinchilla optimumu ~20).
- MAIN tek seed; güven aralıkları yalnızca doğrulama verisi gürültüsünü kapsar.
- Tüm mimariler için tek, ayarlanmamış öğrenme oranı; paralel avantajın bir kısmı reçeteye bağlı olabilir.
- Duvar saati sonuçları bu uygulamaya (sıralı dağıtımlı referans MoE) özgü.

## 9. Sonraki adımlar (Amendment 5)

1. **B_isoflop** (~1 saat): B, C_same'in toplam FLOPs'u kadar (1753 adım, 114.9M token) kendi takvimiyle eğitilir.
   C_same bunu en az 0.02 nats geçemezse fusion, ek hesabın daha verimli bir kullanımı değildir; verimlilik hattı
   kapanır.
2. **Katman çıkarma** (~15 dk, eğitim yok): eğitilmiş C_same'de her fusion katmanı tek tek, ikişer ve hep birlikte
   çıkarılır.
3. **Tek fusion katmanı** (~2.5 saat): en değerli konumda tek katman, kendi iso-FLOP B'sine karşı, 2 seed.
4. **Seed 43 tekrarı**: hâlâ kararsız kalan karşılaştırmalar için.

İsteğe bağlı: 1. adım C_same lehine çıkarsa kademeli büyütme (C_grow).
