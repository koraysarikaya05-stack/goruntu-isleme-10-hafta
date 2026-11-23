import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Resimleri oku
klasor_yolu = os.path.dirname(os.path.abspath(__file__))
imgL = cv2.imread(os.path.join(klasor_yolu, 'scene1.row3.col1.ppm'), cv2.IMREAD_GRAYSCALE).astype(np.float32)
imgR = cv2.imread(os.path.join(klasor_yolu, 'scene1.row3.col3.ppm'), cv2.IMREAD_GRAYSCALE).astype(np.float32)

# --- 3 FARKLI BENZERLİK / MALİYET FONKSİYONU ---
# Hepsini "küçük değer = daha iyi eşleşme" (maliyet) mantığıyla standartlaştırıyoruz.

def sad_maliyet(sol_blok, sag_blok):
    # Sum of Absolute Differences (Mutlak Farklar Toplamı)
    return np.sum(np.abs(sol_blok - sag_blok))

def ssd_maliyet(sol_blok, sag_blok):
    # Sum of Squared Differences (Farkların Kareleri Toplamı)
    return np.sum((sol_blok - sag_blok) ** 2)

def zncc_maliyet(sol_blok, sag_blok):
    # Zero-Mean Normalized Cross-Correlation (Sıfır Ortalamalı Normalize Çapraz Korelasyon)
    sol_fark = sol_blok - np.mean(sol_blok)
    sag_fark = sag_blok - np.mean(sag_blok)
    
    pay = np.sum(sol_fark * sag_fark)
    payda = np.sqrt(np.sum(sol_fark ** 2) * np.sum(sag_fark ** 2))
    
    if payda == 0:
        return 1.0 # Düz/tek renk alanlarda sıfıra bölünme koruması
    
    zncc_skor = pay / payda # -1 ile +1 arası değer alır (+1 en iyi eşleşme)
    return 1.0 - zncc_skor  # Maliyet mantığına çevirmek için 1'den çıkarıyoruz

# --- GENEL BLOK EŞLEŞTİRME FONKSİYONU ---
def blok_eslestir(sol_resim, sag_resim, maliyet_fonksiyonu, blok_boyutu=7, max_disparity=32):
    yukseklik, genislik = sol_resim.shape
    yaricap = blok_boyutu // 2
    disparity_map = np.zeros((yukseklik, genislik), dtype=np.float32)
    
    for y in range(yaricap, yukseklik - yaricap):
        for x in range(yaricap + max_disparity, genislik - yaricap):
            sol_blok = sol_resim[y - yaricap : y + yaricap + 1, 
                                 x - yaricap : x + yaricap + 1]
            
            en_iyi_disparity = 0
            en_kucuk_maliyet = float('inf')
            
            for d in range(max_disparity):
                sag_blok = sag_resim[y - yaricap : y + yaricap + 1, 
                                     (x - d) - yaricap : (x - d) + yaricap + 1]
                
                maliyet = maliyet_fonksiyonu(sol_blok, sag_blok)
                
                if maliyet < en_kucuk_maliyet:
                    en_kucuk_maliyet = maliyet
                    en_iyi_disparity = d
                    
            disparity_map[y, x] = en_iyi_disparity
            
    return disparity_map

# --- TEST VE SÜRE ÖLÇÜMÜ ---
yontemler = [
    ("SAD (Mutlak Fark)", sad_maliyet),
    ("SSD (Kare Fark)", ssd_maliyet),
    ("ZNCC (Normalize Korelasyon)", zncc_maliyet)
]

sonuclar = []
for isim, fonk in yontemler:
    basla = time.perf_counter()
    harita = blok_eslestir(imgL, imgR, fonk, blok_boyutu=7, max_disparity=32)
    sure = time.perf_counter() - basla
    print(f"{isim} tamamlandı -> Süre: {sure:.2f} sn")
    sonuclar.append((isim, harita, sure))

# --- GÖRSELLEŞTİRME ---
plt.figure(figsize=(15, 5))
for i, (isim, harita, sure) in enumerate(sonuclar):
    plt.subplot(1, 3, i + 1)
    plt.imshow(harita, cmap='jet')
    plt.title(f"{isim}\n({sure:.2f} sn)")
    plt.axis('off')

plt.tight_layout()
plt.show()