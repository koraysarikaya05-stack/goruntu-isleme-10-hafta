import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Kodun bulunduğu klasörün yolunu otomatik al
klasor_yolu = os.path.dirname(os.path.abspath(__file__))

imgL = cv2.imread(os.path.join(klasor_yolu, 'scene1.row3.col1.ppm'), cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread(os.path.join(klasor_yolu, 'scene1.row3.col3.ppm'), cv2.IMREAD_GRAYSCALE)

# Parametre adı 'minDisparity' olarak düzeltildi
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,
    blockSize=5,
    P1=8 * 3 * 5**2,
    P2=32 * 3 * 5**2,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

plt.figure(figsize=(10, 5))
plt.imshow(disparity, cmap='jet')
plt.colorbar(label='Disparity (Piksel Kayması)')
plt.title('Tsukuba Fark Haritası (Disparity Map)')
plt.axis('off')
plt.show()



"""
1. Fark Haritasındaki (Disparity Map) Renklerin Anlamı:
- Sıcak / Açık Renkler (Sarı, Turuncu, Kırmızı): Piksel kayması (disparity) değerinin yüksek olduğunu gösterir. 
  Kameraya en yakın nesneler (örneğin öndeki büst) iki kamera görüntüsü arasında daha fazla yer değiştirir.
- Soğuk / Koyu Renkler (Açık Mavi, Turkuaz): Piksel kayması değerinin düşük olduğunu gösterir. 
  Kameraya en uzak bölgeler (arka plandaki duvar ve kitaplık) iki görüntü arasında çok az yer değiştirir.
- En Soldaki Koyu Lacivert Şerit ve Nesne Kenarları: İki kameranın görüş açısının örtüşmediği 
  veya öndeki nesnenin arkasında kaldığı için eşleşme yapılamayan kör nokta (occlusion) alanlarıdır.

2. Gerçek Bir Derinlik Ölçümü Sağlıyor mu?
- Hayır, doğrudan gerçek (metrik) bir derinlik ölçümü sağlamaz; yalnızca göreceli (bağıl) derinlik bilgisi verir.
- Bu harita sadece hangi nesnenin önde, hangisinin arkada olduğunu piksel kayması cinsinden gösterir.
- Gerçek dünya birimleriyle (cm veya metre) derinlik (Z) hesaplayabilmek için kameranın odak uzaklığı (f) 
  ve iki kamera arasındaki fiziksel mesafe olan taban çizgisi (B) değerlerinin bilinmesi ve 
  Z = (f * B) / d formülünün uygulanması gerekir.
"""