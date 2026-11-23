import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Resimleri oku
klasor_yolu = os.path.dirname(os.path.abspath(__file__))
imgL = cv2.imread(os.path.join(klasor_yolu, 'scene1.row3.col1.ppm'), cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread(os.path.join(klasor_yolu, 'scene1.row3.col3.ppm'), cv2.IMREAD_GRAYSCALE)

# Taşmaları önlemek için float tipine çeviriyoruz
imgL = imgL.astype(np.float32)
imgR = imgR.astype(np.float32)

def kendi_blok_eslestirme(sol_resim, sag_resim, blok_boyutu=5, max_disparity=32):
    yukseklik, genislik = sol_resim.shape
    yaricap = blok_boyutu // 2
    
    # Sonuç fark haritasını sıfırlarla başlat
    disparity_map = np.zeros((yukseklik, genislik), dtype=np.float32)
    
    # Sınır taşmalarını önlemek için yaricap kadar içeriden başlıyoruz
    for y in range(yaricap, yukseklik - yaricap):
        for x in range(yaricap + max_disparity, genislik - yaricap):
            
            # Sol resimdeki referans bloğu al
            sol_blok = sol_resim[y - yaricap : y + yaricap + 1, 
                                 x - yaricap : x + yaricap + 1]
            
            en_iyi_disparity = 0
            en_kucuk_fark = float('inf') # Başlangıçta sonsuz hata veriyoruz
            
            # Sağ resimde aynı satırda sola doğru kaydırarak en benzer bloğu ara
            for d in range(max_disparity):
                sag_blok = sag_resim[y - yaricap : y + yaricap + 1, 
                                     (x - d) - yaricap : (x - d) + yaricap + 1]
                
                # SAD (Mutlak Farklar Toplamı) hesapla
                fark = np.sum(np.abs(sol_blok - sag_blok))
                
                if fark < en_kucuk_fark:
                    en_kucuk_fark = fark
                    en_iyi_disparity = d
                    
            disparity_map[y, x] = en_iyi_disparity
            
    return disparity_map

print("Blok eşleştirme hesaplanıyor, lütfen bekleyin (yaklaşık 5-10 saniye sürebilir)...")
benim_disparity = kendi_blok_eslestirme(imgL, imgR, blok_boyutu=7, max_disparity=32)

# Görselleştirme
plt.figure(figsize=(10, 5))
plt.imshow(benim_disparity, cmap='jet')
plt.colorbar(label='Disparity (Piksel Kayması)')
plt.title('Kendi Yazdığımız Blok Eşleştirme (SAD) Fark Haritası')
plt.axis('off')
plt.show()