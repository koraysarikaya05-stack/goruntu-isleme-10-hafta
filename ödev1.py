import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Resimleri oku
klasor_yolu = os.path.dirname(os.path.abspath(__file__))
imgL_uint8 = cv2.imread(os.path.join(klasor_yolu, 'scene1.row3.col1.ppm'), cv2.IMREAD_GRAYSCALE)
imgR_uint8 = cv2.imread(os.path.join(klasor_yolu, 'scene1.row3.col3.ppm'), cv2.IMREAD_GRAYSCALE)

imgL = imgL_uint8.astype(np.float32)
imgR = imgR_uint8.astype(np.float32)

# 2. Canny ile kenarları bul (Sadece kenar pikselleri 255, diğerleri 0 olur)
kenarlar = cv2.Canny(imgL_uint8, 100, 200)

def seyrek_stereo_sad(sol_resim, sag_resim, kenar_maskesi, blok_boyutu=7, max_disparity=32):
    yukseklik, genislik = sol_resim.shape
    yaricap = blok_boyutu // 2
    disparity_map = np.zeros((yukseklik, genislik), dtype=np.float32)
    
    for y in range(yaricap, yukseklik - yaricap):
        for x in range(yaricap + max_disparity, genislik - yaricap):
            # SADECE kenar olan pikselleri hesapla, diğerlerini atla!
            if kenar_maskesi[y, x] == 0:
                continue
                
            sol_blok = sol_resim[y - yaricap : y + yaricap + 1, 
                                 x - yaricap : x + yaricap + 1]
            
            en_iyi_disparity = 0
            en_kucuk_fark = float('inf')
            
            for d in range(max_disparity):
                sag_blok = sag_resim[y - yaricap : y + yaricap + 1, 
                                     (x - d) - yaricap : (x - d) + yaricap + 1]
                fark = np.sum(np.abs(sol_blok - sag_blok))
                
                if fark < en_kucuk_fark:
                    en_kucuk_fark = fark
                    en_iyi_disparity = d
                    
            disparity_map[y, x] = en_iyi_disparity
            
    return disparity_map

def yogun_stereo_sad(sol_resim, sag_resim, blok_boyutu=7, max_disparity=32):
    yukseklik, genislik = sol_resim.shape
    yaricap = blok_boyutu // 2
    disparity_map = np.zeros((yukseklik, genislik), dtype=np.float32)
    
    for y in range(yaricap, yukseklik - yaricap):
        for x in range(yaricap + max_disparity, genislik - yaricap):
            sol_blok = sol_resim[y - yaricap : y + yaricap + 1, 
                                 x - yaricap : x + yaricap + 1]
            
            en_iyi_disparity = 0
            en_kucuk_fark = float('inf')
            
            for d in range(max_disparity):
                sag_blok = sag_resim[y - yaricap : y + yaricap + 1, 
                                     (x - d) - yaricap : (x - d) + yaricap + 1]
                fark = np.sum(np.abs(sol_blok - sag_blok))
                
                if fark < en_kucuk_fark:
                    en_kucuk_fark = fark
                    en_iyi_disparity = d
                    
            disparity_map[y, x] = en_iyi_disparity
            
    return disparity_map

# --- SÜRE ÖLÇÜMÜ VE KARŞILAŞTIRMA ---
basla_seyrek = time.perf_counter()
seyrek_harita = seyrek_stereo_sad(imgL, imgR, kenarlar)
sure_seyrek = time.perf_counter() - basla_seyrek

basla_yogun = time.perf_counter()
yogun_harita = yogun_stereo_sad(imgL, imgR)
sure_yogun = time.perf_counter() - basla_yogun

print(f"Seyrek (Sparse) Stereo Süresi: {sure_seyrek:.2f} sn")
print(f"Yoğun (Dense) Stereo Süresi:   {sure_yogun:.2f} sn")

# --- GÖRSELLEŞTİRME ---
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(kenarlar, cmap='gray')
plt.title("1. Canny Kenar Haritası")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(seyrek_harita, cmap='jet')
plt.title(f"2. Seyrek (Sparse) Fark Haritası\n({sure_seyrek:.2f} sn)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(yogun_harita, cmap='jet')
plt.title(f"3. Yoğun (Dense) Fark Haritası\n({sure_yogun:.2f} sn)")
plt.axis('off')

plt.tight_layout()
plt.show()

"""
Seyrek (Sparse) ve Yoğun (Dense) Stereo Karşılaştırması (Hız ve Doğruluk):
- Hız Açısından: Seyrek stereo yönteminde tüm pikseller yerine yalnızca Canny kenar detektörü ile bulunan
  belirgin kenar pikselleri (görüntünün küçük bir yüzdesi) işlendiği için hesaplama maliyeti çok düşüktür
  ve yoğun (dense) yönteme göre katbekat daha hızlı çalışır.
- Doğruluk Açısından: Kenar bölgeleri yüksek kontrast ve belirgin doku değişimi içerdiğinden yanlış eşleşme
  (ambiguity) ihtimali çok azdır; bu nedenle hesaplanan noktalardaki doğruluk oranı yüksektir. Ancak
  nesnelerin iç kısımları ve düz yüzeyler boş kaldığı için tam (bütünsel) bir 3B derinlik haritası üretmez.
- Yoğun (Dense) Stereo: Görüntüdeki her piksel için derinlik ürettiğinden 3B sahne rekonstrüksiyonu için
  uygundur; fakat düz/dokusuz alanlarda hatalı eşleşmeler yapabilir ve işlem süresi çok daha uzundur.
"""