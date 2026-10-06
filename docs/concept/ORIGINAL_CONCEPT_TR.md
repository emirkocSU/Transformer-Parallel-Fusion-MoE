
**GLM-4.5 / 4.6 / 4.7: seri kullanıyor.** Hugging Face implementasyonundaki decoder katmanı açıkça şu sırada çalışıyor:

\[
x
\rightarrow RMSNorm
\rightarrow SelfAttention
\rightarrow +x
\rightarrow RMSNorm
\rightarrow MLP/MoE
\rightarrow +x
\]

Yani bizim konuştuğumuz klasik:

\[
\boxed{Attention \rightarrow FFN}
\]

yapısı. Kodda önce `self_attn(...)` çağrılıyor, sonucu residual'a ekleniyor; **ondan sonra** `self.mlp(...)` çağrılıyor. Dolayısıyla paralel değiller. [GitHub](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm4_moe/modeling_glm4_moe.py?utm_source=chatgpt.com)

Kimi K2 de aynı açıdan **seri**. Ama burada FFN biraz daha ilginç: Kimi K2 bir **Mixture-of-Experts (MoE)** modeli. Toplam yaklaşık 1 trilyon parametresi var, token başına yaklaşık 32B parametre aktive ediyor; 384 uzmandan 8 tanesi seçiliyor. Attention tarafında ise klasik MHA yerine **MLA (Multi-head Latent Attention)** kullanıyor. [GitHub](https://github.com/MoonshotAI/Kimi-K2?utm_source=chatgpt.com)

Kabaca:

\[
x
\rightarrow
\boxed{MLA}
\rightarrow
\boxed{MoE}
\rightarrow
next\ layer
\]

Yani:

```text
Kimi K2

token representation
       ↓
      MLA
   (attention)
       ↓
    residual
       ↓
      MoE
 (FFN'nin gelişmiş hali)
       ↓
    residual
       ↓
next Transformer layer
```

Üstelik Kimi'nin kendi config dosyasında mimari doğrudan `DeepseekV3ForCausalLM` olarak belirtilmiş; Moonshot da deployment dokümanında Kimi K2'nin DeepSeek-V3 CausalLM mimarisini yeniden kullandığını açıkça söylüyor. [Hugging Face](https://huggingface.co/moonshotai/Kimi-K2-Base/blob/main/config.json?utm_source=chatgpt.com)

Buradaki önemli nokta şu:

**MoE, FFN'nin yerine geçen bambaşka bir attention mekanizması değil.**

Normal Transformer:

\[
Attention
\rightarrow
\boxed{FFN}
\]

Kimi gibi MoE Transformer:

\[
Attention
\rightarrow
\boxed{MoE\ FFN}
\]

Normal FFN'de her token aynı büyük FFN'den geçerken:

\[
x\rightarrow FFN(x)
\]

MoE'de router diyor ki:

> “Bu token için hangi uzman FFN'ler daha uygun?”

Örneğin:

```text
"The Python function throws an exception"

              ↓
            Router
          ↙   ↓   ↘
     Expert 7
     Expert 42
     Expert 106
     ...
```

ve 384 uzmanın yalnızca 8'i çalışıyor.

Bu da çok ilginç biçimde **senin biraz önce sorduğun verimlilik problemine başka bir çözüm** getiriyor:

Sen:

> “Attention ile FFN'yi paralel yapıp zaman kazanamaz mıyız?”

dedin.

PaLM'in yaklaşımı kabaca:

\[
Attention(x)\parallel FFN(x)
\]

Kimi/DeepSeek/GLM tarafının büyük optimizasyonlarından biri ise farklı:

\[
Attention(x)
\rightarrow
\underbrace{yalnızca\ gerekli\ FFN\ uzmanlarını\ çalıştır}_{MoE}
\]

Yani FFN'yi tamamen paralelleştirmek yerine **FFN'nin tamamını her token için çalıştırmıyorlar.**

### Bugünkü birkaç açık modelin resmi

| Model | Attention | FFN tarafı | Attention–FFN |
|---|---|---|---|
| **GLM-4.5/4.6/4.7** | GQA-tarzı attention | MoE | **Seri** |
| **Kimi K2** | MLA | MoE, 384 expert / 8 seçili | **Seri** |
| **DeepSeek-V3** | MLA | MoE | **Seri** |
| PaLM | Multi-head attention | Dense FFN | **Paralel** |

Yani ilginç sonuç şu:

\[
\boxed{\text{Modern model} \neq \text{mutlaka Attention ∥ FFN}}
\]

Hatta **Kimi K2, DeepSeek-V3 ve GLM gibi çok güçlü modern açık modeller hâlâ Attention → FFN/MoE bağımlılığını koruyor.**

Çünkü senin önceki sorunda fark ettiğimiz avantajdan vazgeçmek istemiyorlar:

\[
Attention(x)=\text{bağlamdan yeni bilgi getir}
\]

sonra

\[
MoE(Attention(x))=\textbf{o yeni bilgiyi hemen işle}
\]

Paralel yaparsak ikinci adım aynı katmandaki yeni attention bilgisini göremeyecekti.

Bence buradan sonra **Kimi K2'nin tek bir Transformer katmanını açıp**, `input → MLA → residual → MoE router → 8 expert → residual` şeklinde sayılarla takip etmek çok öğretici olur. O zaman klasik Transformer'dan bugünkü LLM'lere neyin değiştiğini bayağı net görürüz.


Bu fusion mau ffm daha önce denenmiş mi
Evet, çok yakın fikirler denenmiş, ama yaptığımız hızlı literatür taramasına göre senin tarif ettiğin kombinasyonun aynısını — her blokta parallel Attention + MoE, sonra belirli aralıklarla ekstra “fusion MoE-FFN” — standartlaşmış, bilinen bir LLM mimarisi olarak görmedim.

En yakın çalışmalar üç ayrı parçayı deniyor:

PaLM / GPT-J yaklaşımı: Attention ve MLP aynı girdiden paralel çalışıyor:

x
′
=
x
+
A
t
t
e
n
t
i
o
n
(
L
N
(
x
)
)
+
M
L
P
(
L
N
(
x
)
)
x 
′
 =x+Attention(LN(x))+MLP(LN(x))
PaLM bunun büyük ölçekte yaklaşık %15 eğitim hızlanması sağladığını ve 62B deneyinde belirgin kalite kaybı görmediklerini bildiriyor. 

Sandwich Transformer (Press et al., ACL 2020): Araştırmacılar Attention → FFN → Attention → FFN sırasının kutsal olmadığını test etmişler. Attention'ları daha erken, FFN'leri daha geç kümelendiren mimarilerin bazı language-modeling benchmarklarında daha iyi perplexity verdiğini göstermişler. 

Macaron/Conformer tipi yapılar: Bir attention'ın etrafına ekstra FFN koyma fikri de mevcut:

1
2
F
F
N
→
A
t
t
e
n
t
i
o
n
→
1
2
F
F
N
2
1
​
 FFN→Attention→ 
2
1
​
 FFN
Yani “attention ile FFN etkileşimini ek FFN kullanarak güçlendirme” fikrinin de öncülleri var. 

Ama senin önerin bunların biraz farklı bir birleşimi:

P
=
A
t
t
e
n
t
i
o
n
(
x
)
+
M
o
E
(
x
)
P=Attention(x)+MoE(x)
​
 
bunu birkaç layer tekrarla:

P
1
→
P
2
→
P
3
→
P
4
P 
1
​
 →P 
2
​
 →P 
3
​
 →P 
4
​
 
sonra:

F
u
s
i
o
n
M
o
E
(
P
4
)
FusionMoE(P 
4
​
 )
​
 
sonra tekrar:

P
5
→
P
6
→
P
7
→
P
8
→
F
u
s
i
o
n
M
o
E
P 
5
​
 →P 
6
​
 →P 
7
​
 →P 
8
​
 →FusionMoE
Yani örneğin:

Parallel block
      ↓
Parallel block
      ↓
Parallel block
      ↓
Parallel block
      ↓
★ Fusion MoE
      ↓
Parallel block
      ↓
...
Bu, Sandwich Transformer'dan da farklı. Sandwich sublayer sıralamasını değiştiriyor; bizim hipotezimiz ise parallel block'un kaybettiği aynı-katman Attention→FFN etkileşimini periyodik olarak geri kazandırmaya çalışıyor.

Ve bence burada test edilmesi gereken asıl hipotez şu:

Her layer’da serial fusion ger
c
¸
ekten gerekli mi?
Her layer’da serial fusion ger 
c
¸
​
 ekten gerekli mi?
​
 
Belki değildir.

Belki:

A
t
t
e
n
t
i
o
n
∥
M
o
E
Attention∥MoE
4 kez yapıp

F
u
s
i
o
n
M
o
E
FusionMoE
bir kez yapmak yeterlidir.

Eğer öyleyse çok ilginç bir sonuç çıkar: klasik modelin her katmanda ödediği

A
t
t
e
n
t
i
o
n
→
F
F
N
Attention→FFN
dependency maliyetinin bir kısmını kaldırıp, seyrek fusion noktalarıyla kaliteyi geri kazanabiliriz.

Ama bir düzeltme yapayım: buna şimdiden “yeni mimari bulduk” diyemeyiz. Transformer sublayer reordering, parallel blocks, sandwich/Macaron yapıları ve çeşitli fusion yapıları oldukça geniş literatür. Tam kombinasyon için daha sistematik paper/code search yapmak gerekir.

Deneyi yapacaksak bence yalnız iki değil üç model kesin olmalı:

A
:
 
S
e
r
i
a
l
 
A
t
t
e
n
t
i
o
n
→
M
o
E
A: Serial Attention→MoE
​
 
B
:
 
P
a
r
a
l
l
e
l
 
A
t
t
e
n
t
i
o
n
∥
M
o
E
B: Parallel Attention∥MoE
​
 
C
:
 
P
a
r
a
l
l
e
l
 
A
t
t
e
n
t
i
o
n
∥
M
o
E
+
p
e
r
i
o
d
i
c
 
F
u
s
i
o
n
M
o
E
C: Parallel Attention∥MoE+periodic FusionMoE
​
 
Çünkü B olmadan C'nin gerçekten fusion sayesinde iyileştiğini kanıtlayamayız. Bu üçlü deney fikrimizi çok daha bilimsel test eder.

Tamam senin fikrin ne peki sence başarı sağlayacak mı kısaca cevapla
Bence başarı şansı var, özellikle hız/kalite oranında.

Tahminim:

Saf paralel → hızlanır ama bir miktar kalite kaybeder.
Paralel + periyodik Fusion MoE → bu kaybın önemli bölümünü geri kazanabilir.
En olası kazanç, seri modelden daha yüksek kalite değil, benzer kaliteyi daha kısa sürede / daha az latency ile elde etmek.
En kritik soru da fusion sıklığı: bence her 3–4 paralel blokta bir fusion ilk denenmesi gereken yer. Eğer burada perplexity baseline'a yaklaşırken tokens/sec anlamlı yükselirse, fikir gerçekten işe yarıyor demektir.

Tamam o zaman bu test için lazım olan şeyler nedir önce bunu çıkartalım detaylı bir araştırma yap genel bir çerçeve oluşturacağız, ayrıca colab üyeliğim var A100 Gpu içerisinde çalıştırabiliriz: olmalı:

\boxed{A:\ Serial\ Attention\rightarrow MoE}

\boxed{B:\ Parallel\ Attention\parallel MoE}

\boxed{C:\ Parallel\ Attention\parallel MoE + periodic\ FusionMoE}

Çünkü B olmadan C’nin gerçekten fusion
Evet. B kesinlikle gerekli; yoksa C iyi çıkarsa bunun “fusion”dan mı yoksa sadece paralel mimariden mi geldiğini ayıramayız.

Literatüre bakınca deneyin temeli de sağlam: paralel Attention–MLP yaklaşımının GPT-J/PaLM çizgisinde precedenti var; sublayer sırasının değiştirilebildiğini Sandwich Transformer göstermiş; MoE tarafında Switch, OLMoE, DeepSeekMoE ve Mixtral bize router/uzman tasarımı için iyi referanslar veriyor. 

Deneyin ana sorusu
Aslında tek bir hipotezi test ediyoruz:

Her Transformer bloğunda Attention → MoE bağımlılığına gerçekten ihtiyacımız var mı, yoksa çoğu bloğu paralel çalıştırıp belirli aralıklarla fusion yapmak yeterli mi?

Üç model:

A
:
 
S
e
r
i
a
l
A: Serial
​
 
x
l
→
A
t
t
e
n
t
i
o
n
→
M
o
E
→
x
l
+
1
x 
l
​
 →Attention→MoE→x 
l+1
​
 
B
:
 
P
u
r
e
 
P
a
r
a
l
l
e
l
B: Pure Parallel
​
 
A
t
t
e
n
t
i
o
n
(
x
l
)
∥
M
o
E
(
x
l
)
Attention(x 
l
​
 )∥MoE(x 
l
​
 )
x
l
+
1
=
x
l
+
A
t
t
n
(
x
l
)
+
M
o
E
(
x
l
)
x 
l+1
​
 =x 
l
​
 +Attn(x 
l
​
 )+MoE(x 
l
​
 )
ve bizim esas deneyimiz:

C
:
 
P
a
r
a
l
l
e
l
+
P
e
r
i
o
d
i
c
 
F
u
s
i
o
n
C: Parallel+Periodic Fusion
​
 
örneğin:

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
Burada ilk deneyde fusion interval = 4 seçmek mantıklı. Sonra gerekirse 2/4/8 ablation yaparız.

1. Model boyutu
A100 olduğu için aşırı küçük 10M model kullanmak istemiyorum. O ölçekte elde ettiğimiz sonuç büyük LLM mimarisi hakkında zayıf kanıt olur.

Ama ilk deneyde 1B de gereksiz pahalı.

Ben ~100–200M active parameter civarında başlamayı tercih ederim.

Örneğin başlangıç noktası:

Özellik	İlk deney
Layers	12
d
m
o
d
e
l
d 
model
​
 	768
Attention heads	12
Head dimension	64
Context	1024
Experts	8
Active experts	Top-2
Expert FFN	SwiGLU
Normalization	RMSNorm
Position	RoPE
Precision	BF16
Bu modern LLM'lere yeterince benzeyen ama A100 üzerinde tekrar tekrar eğitebileceğimiz bir model verir.

Mixtral'ın her token için router ile iki uzman seçmesi de Top-2 MoE için iyi bir referans. 

2. En kritik mesele: adil karşılaştırma
Burada kolayca yanlış deney yapabiliriz.

C'ye ekstra Fusion MoE eklersek:

P
a
r
a
m
e
t
e
r
s
C
>
P
a
r
a
m
e
t
e
r
s
A
Parameters 
C
​
 >Parameters 
A
​
 
olabilir.

Sonra C daha iyi çıkarsa:

“Mimari daha iyi.”

diyemeyiz.

Belki sadece daha fazla parametresi vardır.

Bu yüzden iki ayrı karşılaştırma yapmamız lazım.

Quality-controlled test: A/B/C mümkün olduğunca aynı active parameter/FLOPs bütçesinde.

System-efficiency test: Gerçek mimarileri olduğu gibi çalıştırıp:

t
o
k
e
n
s
/
s
e
c
tokens/sec
w
a
l
l
−
c
l
o
c
k
wall−clock
V
R
A
M
VRAM
ölçmek.

Bence ikisini birbirinden ayırmamız çok önemli.

3. MoE'yi basit tutacağız
İlk deneyde DeepSeek-V3 seviyesinde karmaşık router tasarlamak istemiyorum.

Basit:

R
o
u
t
e
r
(
x
)
=
s
o
f
t
m
a
x
(
W
r
x
)
Router(x)=softmax(W 
r
​
 x)
sonra:

T
o
p
K
(
R
o
u
t
e
r
(
x
)
,
2
)
TopK(Router(x),2)
ve token iki experte gönderilir.

8 expert:

             Router
          ↙    ↓    ↘
       E1 E2 E3 ... E8

       Top-2 seç
MoE eğitiminde routing dengesizliği ve training instability gerçek sorunlar; Switch Transformer bunu özellikle ele alıyor. MegaBlocks da GPU üzerinde dinamik MoE routing'in sistem maliyetinin ciddi olabileceğini gösteriyor. 

Bu yüzden en azından:

L
=
L
L
M
+
λ
L
b
a
l
a
n
c
e
L=L 
LM
​
 +λL 
balance
​
 
kullanmalıyız.

Ve şunları loglamalıyız:

expert utilization, router entropy, expert başına token sayısı ve load-balancing loss.

Yoksa örneğin 8 expert koyup modelin sürekli 2 tanesini kullandığını fark etmeyebiliriz.

4. Dataset
İlk ciddi test için benim tercihim FineWeb-Edu'dan sabit bir subset olur.

FineWeb-Edu eğitim amaçlı yüksek kaliteli web metinlerinden oluşturulmuş ve model ablation çalışmaları için de kullanılmış açık bir corpus. 

Ama üç modele tam olarak aynı tokenlar ve aynı sırayla verilmesi şart.

Örneğin ilk aşama:

100
M
−
300
M
 
t
r
a
i
n
i
n
g
 
t
o
k
e
n
s
100M−300M training tokens
yeterli olabilir.

Bu deney “iyi bir chatbot üretme” deneyi değil.

Amacımız:

a
r
c
h
i
t
e
c
t
u
r
e
A
v
s
a
r
c
h
i
t
e
c
t
u
r
e
B
v
s
a
r
c
h
i
t
e
c
t
u
r
e
C
architecture 
A
​
 vsarchitecture 
B
​
 vsarchitecture 
C
​
 
karşılaştırması.

Chinchilla çalışmasının temel sonucu da model büyüklüğü ile eğitim token miktarının birlikte değerlendirilmesi gerektiğini gösteriyor; bu yüzden token bütçesini baştan sabitlemek önemli. 

5. Tokenizer
Tokenizer'ı üç model için yeniden eğitmeyeceğiz.

Tek tokenizer:

V
o
c
a
b
u
l
a
r
y
≈
32
K
Vocabulary≈32K
ve:

Dataset
   ↓
same tokenizer
   ↓
same token IDs
   ↓
A / B / C
Böylece tokenizer bir confounder olmaz.

6. Training koşulları tamamen aynı olacak
A/B/C için:

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
Optimizer başlangıç için AdamW olabilir.

A100'de:

B
F
16
BF16
kullanırız.

Attention tarafında da mümkünse FlashAttention-2 kullanırız. A100 üzerinde FA2, attention hesaplamasını ciddi biçimde optimize ediyor; makalede GPT tarzı modellerde A100 başına 225 TFLOPs/s'ye kadar rapor edilmiş. 

Ama A/B/C'nin üçünde de aynı attention kernel'i kullanılacak.

7. Asıl ölçmemiz gereken şey
Burada sadece:

Validation loss hangisinde düşük?

diye bakmak yeterli değil.

Bizim hipotezimiz:

quality per unit compute/time
quality per unit compute/time
​
 
Bu yüzden dört ana grafik istiyorum.

Validation loss vs tokens
Bu bize mimarinin öğrenme verimliliğini gösterir.

x
=
t
r
a
i
n
i
n
g
 
t
o
k
e
n
s
x=training tokens
y
=
v
a
l
i
d
a
t
i
o
n
 
l
o
s
s
y=validation loss
Validation loss vs wall-clock
Bence en önemli grafik bu.

x
=
m
i
n
u
t
e
s
x=minutes
y
=
v
a
l
i
d
a
t
i
o
n
 
l
o
s
s
y=validation loss
Örneğin:

        loss
         │
 A       │\
         │ \
 B       │  \
         │   \ C
         │     \
         └──────────── time
C aynı loss'a daha erken ulaşıyorsa fikrimiz işe yarıyor.

Tokens/sec
Direkt:

T
h
r
o
u
g
h
p
u
t
=
t
o
k
e
n
s
s
e
c
o
n
d
Throughput= 
second
tokens
​
 
A vs B vs C.

GPU utilization / VRAM
A100'ün gerçekten paralellikten yararlanıp yararlanmadığını görmek istiyoruz.

Çünkü teorik:

A
t
t
e
n
t
i
o
n
∥
M
o
E
Attention∥MoE
yazmamız GPU'nun bunu otomatik olarak aynı anda kusursuz çalıştıracağı anlamına gelmiyor.

Bu çok önemli.

8. Gerçek paralellik meselesi
Burada deneyin en teknik ve en kritik kısmı var.

PyTorch'ta şunu yazmamız:

attn = attention(x)
moe = moe(x)

bunları gerçekten paralel çalıştırmaz.

Python satırları sırayla dispatch edilir.

B modelinin iddiasını gerçek anlamda test etmek istiyorsak ilerleyen aşamada CUDA stream/fused-kernel veya uygun kernel-level execution düşünmemiz gerekiyor.

PaLM tipi parallel layers'ın hız avantajının bir bölümü de matrix multiplication işlemlerinin birlikte/fused yapılabilmesinden geliyor. 

Bu nedenle deneyi iki seviyeye ayıracağım:

Architecture experiment

Parallel formulation kaliteyi koruyor mu?

ve ayrı olarak:

Systems experiment

A100 üzerinde bunu gerçekten daha hızlı çalıştırabiliyor muyuz?

Bunları birbirine karıştırmamalıyız.

9. C için fusion tasarımı
İlk versiyonda aşırı karmaşık bir şey yapmayalım.

Dört parallel block:

P
1
,
P
2
,
P
3
,
P
4
P 
1
​
 ,P 
2
​
 ,P 
3
​
 ,P 
4
​
 
ardından:

F
1
=
x
+
M
o
E
(
R
M
S
N
o
r
m
(
x
)
)
F 
1
​
 =x+MoE(RMSNorm(x))
Sonra tekrar:

P
5
.
.
.
P
8
P 
5
​
 ...P 
8
​
 
ve:

F
2
F 
2
​
 
Yani 12-layer örneğinde:

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
Burada final fusion özellikle mantıklı; Sandwich Transformer'ın başarılı varyantlarında daha fazla FFN'nin üst katmanlara doğru bulunmasının faydalı olabildiğine dair sonuç da hipotezimizle ilginç biçimde uyumlu. Ama çalışma bunun her görevde garanti olmadığını da gösteriyor. 

10. İlk deneyden sonra gerçek ablation
Eğer C umut verirse ikinci deney:

C
2
:
 
f
u
s
i
o
n
 
e
v
e
r
y
 
2
C 
2
​
 : fusion every 2
C
4
:
 
f
u
s
i
o
n
 
e
v
e
r
y
 
4
C 
4
​
 : fusion every 4
C
8
:
 
f
u
s
i
o
n
 
e
v
e
r
y
 
8
C 
8
​
 : fusion every 8
C
f
i
n
a
l
:
 
o
n
l
y
 
f
i
n
a
l
 
f
u
s
i
o
n
C 
final
​
 : only final fusion
olmalı.

O zaman çok güzel bir eğri elde ederiz:

f
u
s
i
o
n
 
f
r
e
q
u
e
n
c
y
↔
q
u
a
l
i
t
y
↔
s
p
e
e
d
fusion frequency↔quality↔speed
Ve belki optimum:

F
u
s
i
o
n
 
e
v
e
r
y
 
4
Fusion every 4
​
 
değil, 6 veya 8 çıkar. Bunu önceden varsaymamalıyız.

Başarı kriterimiz
Deneyden önce başarıyı tanımlayalım ki sonuca bakıp kriter değiştirmeyelim.

Ben ana kriteri şöyle koyardım:

C
,
 
A
′
n
ı
n
v
a
l
i
d
a
t
i
o
n
 
l
o
s
s
′
u
n
a
d
a
h
a
 
k
ı
s
a
 
w
a
l
l
−
c
l
o
c
k
 
s
u
¨
r
e
d
e
 
u
l
a
s
\c
a
b
i
l
i
y
o
r
 
m
u
?
C, A 
′
 nınvalidation loss 
′
 unadaha kısa wall−clock s 
u
¨
 rede ula 
s
\c
​
 abiliyor mu?
​
 
İkinci kriter:

P
P
L
C
≈
P
P
L
A
ve
t
h
r
o
u
g
h
p
u
t
C
>
t
h
r
o
u
g
h
p
u
t
A
PPL 
C
​
 ≈PPL 
A
​
 vethroughput 
C
​
 >throughput 
A
​
 
​
 
B ise bize mekanizmayı anlatacak:

B hızlı + kötü, C hızlı + iyi: → fusion gerçekten işe yarıyor.
B ≈ C: → fusion gereksiz.
A > B,C: → serial dependency önemli.
C > A aynı compute'ta: → çok ilginç; mimari yalnız hız değil öğrenme açısından da avantaj sağlıyor olabilir.
Dolayısıyla deneyin bilimsel omurgası gerçekten:

A
v
s
B
v
s
C
AvsBvsC
​
 
olmalı.

Ve A100'ün olması güzel; bu artık “oyuncak fikir çalışıyor mu?” testinden ziyade küçük ölçekli gerçek bir architecture ablation yapmamıza izin veriyor.

Bir sonraki aşamada kodlamadan önce tam deney spesifikasyonunu dondurmak en doğrusu: modelin kesin parametre sayısı, 8 expert'in boyutları, Top-2 routing, fusion katmanlarının parametre bütçesinin A ile nasıl eşitleneceği, token sayısı ve A100'de tahmini çalışma süresi. En hassas kısım özellikle C'ye ekstra Fusion MoE eklerken A/B/C'nin compute bütçesini nasıl adil tutacağımız olacak.
