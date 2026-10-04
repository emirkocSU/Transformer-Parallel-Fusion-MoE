Evet. **B kesinlikle gerekli**; yoksa C iyi çıkarsa bunun “fusion”dan mı yoksa sadece paralel mimariden mi geldiğini ayıramayız.

Literatüre bakınca deneyin temeli de sağlam: paralel Attention–MLP yaklaşımının GPT-J/PaLM çizgisinde precedenti var; sublayer sırasının değiştirilebildiğini Sandwich Transformer göstermiş; MoE tarafında Switch, OLMoE, DeepSeekMoE ve Mixtral bize router/uzman tasarımı için iyi referanslar veriyor. [GitHub](https://github.com/QihongRuan/stanford-cs336-notes/blob/main/notes/lec03.md?utm_source=chatgpt.com)

## Deneyin ana sorusu

Aslında tek bir hipotezi test ediyoruz:

> **Her Transformer bloğunda Attention → MoE bağımlılığına gerçekten ihtiyacımız var mı, yoksa çoğu bloğu paralel çalıştırıp belirli aralıklarla fusion yapmak yeterli mi?**

Üç model:

\[
\boxed{A:\ Serial}
\]

\[
x_l
\rightarrow Attention
\rightarrow MoE
\rightarrow x_{l+1}
\]

\[
\boxed{B:\ Pure\ Parallel}
\]

\[
Attention(x_l)\parallel MoE(x_l)
\]

\[
x_{l+1}=x_l+Attn(x_l)+MoE(x_l)
\]

ve bizim esas deneyimiz:

\[
\boxed{C:\ Parallel+Periodic\ Fusion}
\]

örneğin:

```text
Parallel
   ↓
Parallel
   ↓
Parallel
   ↓
Parallel
   ↓
FUSION MoE
   ↓
Parallel
   ↓
Parallel
   ↓
Parallel
   ↓
Parallel
   ↓
FUSION MoE
```

Burada ilk deneyde **fusion interval = 4** seçmek mantıklı. Sonra gerekirse 2/4/8 ablation yaparız.

---

## 1. Model boyutu

A100 olduğu için aşırı küçük 10M model kullanmak istemiyorum. O ölçekte elde ettiğimiz sonuç büyük LLM mimarisi hakkında zayıf kanıt olur.

Ama ilk deneyde 1B de gereksiz pahalı.

Ben **~100–200M active parameter** civarında başlamayı tercih ederim.

Örneğin başlangıç noktası:

| Özellik | İlk deney |
|---|---:|
| Layers | 12 |
| \(d_{model}\) | 768 |
| Attention heads | 12 |
| Head dimension | 64 |
| Context | 1024 |
| Experts | 8 |
| Active experts | Top-2 |
| Expert FFN | SwiGLU |
| Normalization | RMSNorm |
| Position | RoPE |
| Precision | BF16 |

Bu modern LLM'lere yeterince benzeyen ama A100 üzerinde tekrar tekrar eğitebileceğimiz bir model verir.

Mixtral'ın her token için router ile iki uzman seçmesi de Top-2 MoE için iyi bir referans. [arXiv](https://arxiv.org/abs/2401.04088?utm_source=chatgpt.com)

---

## 2. En kritik mesele: adil karşılaştırma

Burada kolayca yanlış deney yapabiliriz.

C'ye ekstra Fusion MoE eklersek:

\[
Parameters_C > Parameters_A
\]

olabilir.

Sonra C daha iyi çıkarsa:

> “Mimari daha iyi.”

diyemeyiz.

Belki sadece **daha fazla parametresi vardır.**

Bu yüzden iki ayrı karşılaştırma yapmamız lazım.

**Quality-controlled test:** A/B/C mümkün olduğunca aynı active parameter/FLOPs bütçesinde.

**System-efficiency test:** Gerçek mimarileri olduğu gibi çalıştırıp:

\[
tokens/sec
\]

\[
wall-clock
\]

\[
VRAM
\]

ölçmek.

Bence ikisini birbirinden ayırmamız çok önemli.

---

## 3. MoE'yi basit tutacağız

İlk deneyde DeepSeek-V3 seviyesinde karmaşık router tasarlamak istemiyorum.

Basit:

\[
Router(x)=softmax(W_rx)
\]

sonra:

\[
TopK(Router(x),2)
\]

ve token iki experte gönderilir.

8 expert:

```text
             Router
          ↙    ↓    ↘
       E1 E2 E3 ... E8

       Top-2 seç
```

MoE eğitiminde routing dengesizliği ve training instability gerçek sorunlar; Switch Transformer bunu özellikle ele alıyor. MegaBlocks da GPU üzerinde dinamik MoE routing'in sistem maliyetinin ciddi olabileceğini gösteriyor. [Journal of Machine Learning Research](https://www.jmlr.org/papers/v23/21-0998.html?utm_source=chatgpt.com)

Bu yüzden en azından:

\[
L =
L_{LM}
+
\lambda L_{balance}
\]

kullanmalıyız.

Ve şunları loglamalıyız:

**expert utilization**, router entropy, expert başına token sayısı ve load-balancing loss.

Yoksa örneğin 8 expert koyup modelin sürekli 2 tanesini kullandığını fark etmeyebiliriz.

---

## 4. Dataset

İlk ciddi test için benim tercihim **FineWeb-Edu'dan sabit bir subset** olur.

FineWeb-Edu eğitim amaçlı yüksek kaliteli web metinlerinden oluşturulmuş ve model ablation çalışmaları için de kullanılmış açık bir corpus. [Hugging Face](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu?utm_source=chatgpt.com)

Ama üç modele **tam olarak aynı tokenlar ve aynı sırayla** verilmesi şart.

Örneğin ilk aşama:

\[
100M-300M\ training\ tokens
\]

yeterli olabilir.

Bu deney “iyi bir chatbot üretme” deneyi değil.

Amacımız:

\[
architecture_A
\quad vs\quad
architecture_B
\quad vs\quad
architecture_C
\]

karşılaştırması.

Chinchilla çalışmasının temel sonucu da model büyüklüğü ile eğitim token miktarının birlikte değerlendirilmesi gerektiğini gösteriyor; bu yüzden token bütçesini baştan sabitlemek önemli. [arXiv](https://arxiv.org/abs/2203.15556?utm_source=chatgpt.com)

---

## 5. Tokenizer

Tokenizer'ı üç model için yeniden eğitmeyeceğiz.

Tek tokenizer:

\[
Vocabulary \approx 32K
\]

ve:

```text
Dataset
   ↓
same tokenizer
   ↓
same token IDs
   ↓
A / B / C
```

Böylece tokenizer bir confounder olmaz.

---

## 6. Training koşulları tamamen aynı olacak

A/B/C için:

```text
same dataset
same tokenizer
same train/validation split
same random seed
same batch size
same optimizer
same learning-rate schedule
same warmup
same context length
same token budget
same precision
```

Optimizer başlangıç için AdamW olabilir.

A100'de:

\[
BF16
\]

kullanırız.

Attention tarafında da mümkünse **FlashAttention-2** kullanırız. A100 üzerinde FA2, attention hesaplamasını ciddi biçimde optimize ediyor; makalede GPT tarzı modellerde A100 başına 225 TFLOPs/s'ye kadar rapor edilmiş. [arXiv](https://arxiv.org/abs/2307.08691?utm_source=chatgpt.com)

Ama A/B/C'nin üçünde de aynı attention kernel'i kullanılacak.

---

## 7. Asıl ölçmemiz gereken şey

Burada sadece:

> Validation loss hangisinde düşük?

diye bakmak yeterli değil.

Bizim hipotezimiz:

\[
\boxed{\text{quality per unit compute/time}}
\]

Bu yüzden dört ana grafik istiyorum.

### Validation loss vs tokens

Bu bize mimarinin **öğrenme verimliliğini** gösterir.

\[
x=training\ tokens
\]

\[
y=validation\ loss
\]

---

### Validation loss vs wall-clock

Bence **en önemli grafik bu.**

\[
x=minutes
\]

\[
y=validation\ loss
\]

Örneğin:

```text
        loss
         │
 A       │\
         │ \
 B       │  \
         │   \ C
         │     \
         └──────────── time
```

C aynı loss'a daha erken ulaşıyorsa fikrimiz işe yarıyor.

---

### Tokens/sec

Direkt:

\[
Throughput =
\frac{tokens}{second}
\]

A vs B vs C.

---

### GPU utilization / VRAM

A100'ün gerçekten paralellikten yararlanıp yararlanmadığını görmek istiyoruz.

Çünkü teorik:

\[
Attention\parallel MoE
\]

yazmamız GPU'nun bunu otomatik olarak aynı anda kusursuz çalıştıracağı anlamına gelmiyor.

Bu çok önemli.

---

# 8. Gerçek paralellik meselesi

Burada deneyin en teknik ve en kritik kısmı var.

PyTorch'ta şunu yazmamız:

```python
attn = attention(x)
moe = moe(x)
```

**bunları gerçekten paralel çalıştırmaz.**

Python satırları sırayla dispatch edilir.

B modelinin iddiasını gerçek anlamda test etmek istiyorsak ilerleyen aşamada CUDA stream/fused-kernel veya uygun kernel-level execution düşünmemiz gerekiyor.

PaLM tipi parallel layers'ın hız avantajının bir bölümü de matrix multiplication işlemlerinin birlikte/fused yapılabilmesinden geliyor. [GitHub](https://github.com/QihongRuan/stanford-cs336-notes/blob/main/notes/lec03.md?utm_source=chatgpt.com)

Bu nedenle deneyi iki seviyeye ayıracağım:

**Architecture experiment**

> Parallel formulation kaliteyi koruyor mu?

ve ayrı olarak:

**Systems experiment**

> A100 üzerinde bunu gerçekten daha hızlı çalıştırabiliyor muyuz?

Bunları birbirine karıştırmamalıyız.

---

# 9. C için fusion tasarımı

İlk versiyonda aşırı karmaşık bir şey yapmayalım.

Dört parallel block:

\[
P_1,P_2,P_3,P_4
\]

ardından:

\[
F_1 = x + MoE(RMSNorm(x))
\]

Sonra tekrar:

\[
P_5...P_8
\]

ve:

\[
F_2
\]

Yani 12-layer örneğinde:

```text
Embedding
    │
   P1
    │
   P2
    │
   P3
    │
   P4
    │
 ★ Fusion MoE
    │
   P5
    │
   P6
    │
   P7
    │
   P8
    │
 ★ Fusion MoE
    │
   P9
    │
  P10
    │
  P11
    │
  P12
    │
 ★ Final Fusion MoE
    │
 LM Head
```

Burada **final fusion** özellikle mantıklı; Sandwich Transformer'ın başarılı varyantlarında daha fazla FFN'nin üst katmanlara doğru bulunmasının faydalı olabildiğine dair sonuç da hipotezimizle ilginç biçimde uyumlu. Ama çalışma bunun her görevde garanti olmadığını da gösteriyor. [ACL Anthology](https://aclanthology.org/2020.acl-main.270/?utm_source=chatgpt.com)

---

# 10. İlk deneyden sonra gerçek ablation

Eğer C umut verirse ikinci deney:

\[
C_2:\ fusion\ every\ 2
\]

\[
C_4:\ fusion\ every\ 4
\]

\[
C_8:\ fusion\ every\ 8
\]

\[
C_{final}:\ only\ final\ fusion
\]

olmalı.

O zaman çok güzel bir eğri elde ederiz:

\[
fusion\ frequency
\leftrightarrow
quality
\leftrightarrow
speed
\]

Ve belki optimum:

\[
\boxed{Fusion\ every\ 4}
\]

değil, 6 veya 8 çıkar. Bunu önceden varsaymamalıyız.

---

# Başarı kriterimiz

Deneyden **önce** başarıyı tanımlayalım ki sonuca bakıp kriter değiştirmeyelim.

Ben ana kriteri şöyle koyardım:

\[
\boxed{
C,\ A'nın validation\ loss'una
daha\ kısa\ wall-clock\ sürede\ ulaşabiliyor\ mu?
}
\]

İkinci kriter:

\[
\boxed{
PPL_C \approx PPL_A
\quad\text{ve}\quad
throughput_C > throughput_A
}
\]

B ise bize mekanizmayı anlatacak:

- **B hızlı + kötü, C hızlı + iyi:** → **fusion gerçekten işe yarıyor.**
- **B ≈ C:** → fusion gereksiz.
- **A > B,C:** → serial dependency önemli.
- **C > A aynı compute'ta:** → çok ilginç; mimari yalnız hız değil öğrenme açısından da avantaj sağlıyor olabilir.

Dolayısıyla deneyin bilimsel omurgası gerçekten:

\[
\boxed{A\quad vs\quad B\quad vs\quad C}
\]

olmalı.

Ve A100'ün olması güzel; bu artık “oyuncak fikir çalışıyor mu?” testinden ziyade **küçük ölçekli gerçek bir architecture ablation** yapmamıza izin veriyor.

Bir sonraki aşamada kodlamadan önce **tam deney spesifikasyonunu** dondurmak en doğrusu: modelin kesin parametre sayısı, 8 expert'in boyutları, Top-2 routing, fusion katmanlarının parametre bütçesinin A ile nasıl eşitleneceği, token sayısı ve A100'de tahmini çalışma süresi. En hassas kısım özellikle **C'ye ekstra Fusion MoE eklerken A/B/C'nin compute bütçesini nasıl adil tutacağımız** olacak.
