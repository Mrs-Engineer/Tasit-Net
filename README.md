# TaşıtNet: Taşıt Görüntülerinin Yapay Zeka ile Sınıflandırılması

Bu proje, taşıt (uçak, otomobil, gemi, kamyon) görüntülerinin yapay zeka (derin öğrenme) ile otomatik olarak sınıflandırılmasını sağlayan bir web uygulamasıdır. Proje, CIFAR-10 veri seti üzerinde eğitilmiş bir model ile Flask tabanlı kullanıcı dostu bir arayüz sunar.

## Proje Amacı
- Görüntüden taşıt tespiti ve sınıflandırması yapmak
- Kullanıcıdan alınan görselin taşıt olup olmadığını ve hangi taşıt türüne ait olduğunu belirlemek
- Sadece taşıt sınıfları için (airplane, automobile, ship, truck) doğru sonuç vermek

## Özellikler
- CIFAR-10 veri seti ile eğitilmiş derin öğrenme modeli
- Sadece taşıt sınıflarını tanıma (uçak, otomobil, gemi, kamyon)
- Kullanıcı dostu web arayüzü
- Yüklenen görselin sınıfını ve güven skorunu gösterme
- Sınıf olasılıklarını grafikle görselleştirme

## Kurulum ve Çalıştırma

1. Gerekli paketleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```
2. Modeli eğitin (ilk kez çalıştırıyorsanız):
   ```bash
   python train.py
   ```
   - CIFAR-10 veri seti otomatik olarak indirilecek ve model eğitilecektir.
   - Eğitilmiş model `results/cifar10_model.h5` olarak kaydedilir.
3. Web uygulamasını başlatın:
   ```bash
   python app.py
   ```
4. Tarayıcınızda `http://localhost:5000` adresine gidin.

## Kullanım
- "Görüntü Seç" butonuyla bir görsel yükleyin.
- "Sınıflandır" butonuna tıklayın.
- Sonuç kutusunda tahmin edilen taşıt türü ve güven skoru gösterilir.
- Eğer görsel taşıt değilse "Bu bir taşıt değildir." uyarısı çıkar.
- Sınıf olasılıkları grafik olarak gösterilir.

## Desteklenen Sınıflar
| İngilizce   | Türkçe    |
|------------|-----------|
| airplane   | uçak      |
| automobile | otomobil  |
| ship       | gemi      |
| truck      | kamyon    |

## Model Mimarisi ve Teknik Detaylar

### Model Katmanları
- **Girdi:** 32x32 piksel, 3 kanallı (RGB) görüntü
- **1. Konvolüsyon Bloğu:**
  - Conv2D (32 filtre, 3x3, ReLU)
  - BatchNormalization
  - MaxPooling2D (2x2)
- **2. Konvolüsyon Bloğu:**
  - Conv2D (64 filtre, 3x3, ReLU)
  - BatchNormalization
  - MaxPooling2D (2x2)
- **3. Konvolüsyon Bloğu:**
  - Conv2D (128 filtre, 3x3, ReLU)
  - BatchNormalization
  - MaxPooling2D (2x2)
- **Tam Bağlantılı Katmanlar:**
  - Flatten
  - Dense (512 nöron, ReLU)
  - Dropout (0.5)
  - Dense (10 nöron, softmax) — CIFAR-10'un 10 sınıfı için çıkış

### Eğitim Süreci
- **Veri Seti:** CIFAR-10 (otomatik indirilir)
- **Ön İşleme:**
  - Görüntüler [0, 1] aralığına normalize edilir
  - Etiketler one-hot encoding ile dönüştürülür
- **Kayıp Fonksiyonu:** Categorical Crossentropy
- **Optimizasyon:** Adam (learning_rate=0.001)
- **Batch Size:** 32
- **Epoch:** 10 (veya erken durdurma ile)
- **Callback'ler:**
  - EarlyStopping (val_loss, patience=5, en iyi ağırlıkları geri yükler)
  - ReduceLROnPlateau (val_loss, factor=0.5, patience=3, min_lr=1e-6)
- **Eğitim Sonrası:**
  - Model `results/cifar10_model.h5` olarak kaydedilir
  - Eğitim ve doğrulama kaybı/doğruluğu grafiği `results/training_metrics.png` olarak kaydedilir

### Tahmin Süreci
- Kullanıcıdan alınan görsel 32x32 boyutuna getirilir ve normalize edilir
- Model ile tahmin yapılır
- En yüksek olasılığa sahip sınıf seçilir
- Eğer tahmin edilen sınıf taşıt değilse "Bu bir taşıt değildir." mesajı döner
- Sonuçlar ve tüm sınıf olasılıkları arayüzde gösterilir

### Kullanılan Kütüphaneler
- **TensorFlow/Keras:** Derin öğrenme modeli ve eğitim
- **Flask:** Web arayüzü ve API
- **NumPy:** Veri işleme
- **Pillow:** Görüntü işleme
- **Matplotlib:** Eğitim grafikleri
- **scikit-learn:** (Eğitimde ek metrikler için kullanılabilir)

## Notlar
- Model sadece taşıt sınıfları için güvenilir sonuç verir.
- Diğer CIFAR-10 sınıfları için "Bu bir taşıt değildir." uyarısı çıkar.
- Görsellerin net ve aydınlık olması önerilir.

## Lisans
Bu proje eğitim ve ödev amaçlıdır. 