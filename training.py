import torch
import torch.nn as nn
import torch.optim as optim
import pyarrow.parquet as pq
import numpy as np

# --- 1. MODEL (Zaten sende var, aynen kalsın) ---
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

# --- 2. AYARLAR ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator().to(device)
criterion = nn.MSELoss() # Ne kadar hata yaptığını ölçer
optimizer = optim.Adam(netG.parameters(), lr=0.001) # Hataları düzeltir

dosya_yolu = r"C:\Users\Onur Kılıç\Downloads\QCDToGGQQ_IMGjet_RH1all_jet0_run0_n36272_LR.parquet"

# --- 3. EĞİTİM DÖNGÜSÜ (Burası Yeni!) ---
def preprocess_batch(df):
    """Pandas batch'ini PyTorch tensorüne çevirir"""
    lr_list = []
    hr_list = []
    
    for i in range(len(df)):
        # LR (64x64)
        lr_raw = df['X_jets_LR'].iloc[i]
        lr_img = np.stack([np.hstack(lr_raw[j]).reshape(64,64) for j in range(3)])
        
        # HR (125x125) - Hedefimiz bu!
        hr_raw = df['X_jets'].iloc[i]
        hr_img = np.stack([np.hstack(hr_raw[j]).reshape(125,125) for j in range(3)])
        
        lr_list.append(lr_img)
        hr_list.append(hr_img)
        
    return (torch.tensor(np.array(lr_list)).float().to(device), 
            torch.tensor(np.array(hr_list)).float().to(device))

try:
    print(f"Eğitim başlıyor... Cihaz: {device}")
    parquet_file = pq.ParquetFile(dosya_yolu)
    
    # Dosyadan 32'şerli paketler halinde oku (Batch Size = 32)
    for i, batch in enumerate(parquet_file.iter_batches(batch_size=32)):
        df = batch.to_pandas()
        lr_batch, hr_batch = preprocess_batch(df)
        
        # --- EĞİTİM ADIMI ---
        optimizer.zero_grad() # Gradyanları sıfırla
        outputs = netG(lr_batch) # Tahmin yap
        loss = criterion(outputs, hr_batch) # Hatayı hesapla
        loss.backward() # Geriye yayılım (Backpropagation)
        optimizer.step() # Ağırlıkları güncelle
        
        if i % 10 == 0:
            print(f"Batch {i}, Kayıp (Loss): {loss.item():.6f}")
            
        if i == 100: # Test için ilk 100 batch yeterli (Hızlı sonuç almak için)
            break

    print("İlk eğitim turu tamamlandı! Modeli kaydediyorum...")
    torch.save(netG.state_dict(), "cms_super_res_model.pth")

except Exception as e:
    print(f"Eğitim hatası: {e}")