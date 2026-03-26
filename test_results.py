import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

# 1. MODEL TANIMI (Olmazsa olmaz, sınıfı buraya da kopyalıyoruz)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.upsample = nn.Upsample(size=(125, 125), mode='bilinear', align_corners=False)
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )
    def forward(self, x):
        return self.main(self.upsample(x))

# 2. AYARLAR VE YÜKLEME
device = torch.device("cpu") # İşlemci üzerinde test edelim
netG = Generator().to(device)

try:
    # Eğittiğin modeli yükle
    netG.load_state_dict(torch.load("cms_super_res_model.pth", map_location=device))
    netG.eval()
    print("Model başarıyla yüklendi!")

    # 3. TEST VERİSİ ÇEKME (Dosyadan taze bir örnek alalım)
    dosya_yolu = r"C:\Users\Onur Kılıç\Downloads\QCDToGGQQ_IMGjet_RH1all_jet0_run0_n36272_LR.parquet"
    parquet_file = pq.ParquetFile(dosya_yolu)
    batch = next(parquet_file.iter_batches(batch_size=1))
    df = batch.to_pandas()

    # Veriyi hazırla (LR ve HR)
    lr_raw = df['X_jets_LR'].iloc[0]
    hr_raw = df['X_jets'].iloc[0]
    
    lr_img = np.stack([np.hstack(lr_raw[j]).reshape(64,64) for j in range(3)])
    hr_img = np.stack([np.hstack(hr_raw[j]).reshape(125,125) for j in range(3)])
    
    lr_tensor = torch.tensor(lr_img).float().unsqueeze(0).to(device)
    hr_tensor = torch.tensor(hr_img).float().unsqueeze(0).to(device)

    # 4. TAHMİN
    with torch.no_grad():
        prediction = netG(lr_tensor)

    # 5. GÖRSELLEŞTİRME (Zafer Tablosu)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(lr_tensor[0, 0].numpy(), cmap='inferno')
    plt.title("Giriş (LR 64x64)")

    plt.subplot(1, 3, 2)
    plt.imshow(prediction[0, 0].numpy(), cmap='inferno')
    plt.title("Modelin Tahmini (125x125)")

    plt.subplot(1, 3, 3)
    plt.imshow(hr_tensor[0, 0].numpy(), cmap='inferno')
    plt.title("Gerçek Hedef (HR 125x125)")

    plt.tight_layout()
    print("Görsel oluşturuldu. Pencere açılıyor...")
    plt.show()

except Exception as e:
    print(f"Hata: {e}")