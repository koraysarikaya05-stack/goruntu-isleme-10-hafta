import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Resimleri oku
klasor_yolu = os.path.dirname(os.path.abspath(__file__))
imgL = cv2.imread(os.path.join(klasor_yolu, 'scene1.row3.col1.ppm'), cv2.IMREAD_GRAYSCALE).astype(np.float32)
imgR = cv2.imread(os.path.join(klasor_yolu, 'scene1.row3.col3.ppm'), cv2.IMREAD_GRAYSCALE).astype(np.float32)

def kendi_sgbm(sol_resim, sag_resim, blok_boyutu=5, max_disparity=32, P1=100, P2=400):
    yukseklik, genislik = sol_resim.shape
    yaricap = blok_boyutu // 2
    
    # Adım 1: Her piksel ve her disparity (d) için başlangıç SAD maliyet küpünü oluştur
    maliyet_kupu = np.full((yukseklik, genislik, max_disparity), 1e6, dtype=np.float32)
    
    for y in range(yaricap, yukseklik - yaricap):
        for x in range(yaricap + max_disparity, genislik - yaricap):
            sol_blok = sol_resim[y - yaricap : y + yaricap + 1, 
                                 x - yaricap : x + yaricap + 1]
            for d in range(max_disparity):
                sag_blok = sag_resim[y - yaricap : y + yaricap + 1, 
                                     (x - d) - yaricap : (x - d) + yaricap + 1]
                maliyet_kupu[y, x, d] = np.sum(np.abs(sol_blok - sag_blok))

    # Adım 2: Semi-Global Maliyet Biriktirme (Soldan Sağa ve Sağdan Sola Yollar)
    # Komşu pikseller arasındaki ani disparity değişimlerine P1 ve P2 cezası uygulanır
    L_soldan_saga = np.copy(maliyet_kupu)
    L_sagdan_sola = np.copy(maliyet_kupu)
    
    # Soldan sağa geçiş
    for y in range(yaricap, yukseklik - yaricap):
        for x in range(yaricap + max_disparity + 1, genislik - yaricap):
            onceki = L_soldan_saga[y, x - 1]
            min_onceki = np.min(onceki)
            for d in range(max_disparity):
                c0 = onceki[d] # Aynı derinlikte kalma (ceza yok)
                c1 = onceki[d - 1] + P1 if d > 0 else 1e6 # 1 piksel değişim (küçük ceza P1)
                c2 = onceki[d + 1] + P1 if d < max_disparity - 1 else 1e6
                c3 = min_onceki + P2 # Büyük sıçrama / nesne kenarı (büyük ceza P2)
                
                L_soldan_saga[y, x, d] += min(c0, c1, c2, c3) - min_onceki

    # Sağdan sola geçiş
    for y in range(yaricap, yukseklik - yaricap):
        for x in range(genislik - yaricap - 2, yaricap + max_disparity - 1, -1):
            onceki = L_sagdan_sola[y, x + 1]
            min_onceki = np.min(onceki)
            for d in range(max_disparity):
                c0 = onceki[d]
                c1 = onceki[d - 1] + P1 if d > 0 else 1e6
                c2 = onceki[d + 1] + P1 if d < max_disparity - 1 else 1e6
                c3 = min_onceki + P2
                
                L_sagdan_sola[y, x, d] += min(c0, c1, c2, c3) - min_onceki

    # İki yönden gelen birikmiş maliyetleri topla ve en düşük maliyetli disparity'yi seç
    toplam_maliyet = L_soldan_saga + L_sagdan_sola
    disparity_map = np.argmin(toplam_maliyet, axis=2).astype(np.float32)
    
    # Başlangıçtaki sol boşluk sınırını sıfırla
    disparity_map[:, :yaricap + max_disparity] = 0
    return disparity_map

print("Kendi SGBM algoritması çalışıyor, lütfen bekleyin...")
sgbm_harita = kendi_sgbm(imgL, imgR, blok_boyutu=5, max_disparity=32, P1=150, P2=600)

plt.figure(figsize=(10, 5))
plt.imshow(sgbm_harita, cmap='jet')
plt.colorbar(label='Disparity (Piksel Kayması)')
plt.title('Kendi Yazdığımız SGBM Fark Haritası')
plt.axis('off')
plt.show()

"""
Smoothness (Yumuşak Geçiş) Parametresi Ne İşe Yarar?
- Standart blok eşleştirmede her piksel komşularından bağımsız hesaplandığı için özellikle düz/dokusuz 
  yüzeylerde gürültü ve ani derinlik sıçramaları oluşur.
- Smoothness (yumuşaklık) parametreleri (SGBM'de P1 ve P2 ceza katsayıları), yan yana duran komşu piksellerin 
  derinlik (disparity) değerlerinin birbirine yakın olmasını zorunlu kılar.
- P1: Eğimli veya kavisli yüzeylerdeki 1 piksellik küçük derinlik değişimlerine uygulanan hafif cezadır; 
  yüzeylerin pürüzsüz görünmesini sağlar.
- P2: 1 pikselden büyük ani derinlik sıçramalarına uygulanan ağır cezadır (P2 > P1). Bu sayede düz yüzeylerdeki 
  gürültüler bastırılırken, gerçek nesne kenarlarındaki keskin derinlik geçişleri korunur.
"""