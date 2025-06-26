# GAN Image Generator

StyleGAN2-ADA, StyleGAN3 ve Stable Diffusion modelleri ile görüntü üretimi için uygulama.

## Özellikler

- **Çoklu Model Desteği**: StyleGAN2-ADA, StyleGAN3 ve Stable Diffusion modellerini destekler
- **LoRA Fine-tuning**: Stable Diffusion modelleri için LoRA ile özelleştirme
- **Kullanıcı Dostu Arayüz**: Streamlit tabanlı modern web arayüzü
- **Toplu Görüntü Üretimi**: Birden fazla seed ile aynı anda görüntü üretimi
- **Gelişmiş Parametreler**: Truncation, noise mode ve transform desteği
- **Analiz Araçları**: Kapsamlı eğitim metrik analizi ve görselleştirme
- **Otomatik Hata Yönetimi**: CUDA extension sorunları için otomatik CPU geçişi
- **Temiz Kod Mimarisi**: SOLID prensipleri ve clean architecture uygulaması

## Kurulum

### Gereksinimler

- Python 3.8+ (Python 3.11 veya öncesi önerilir CUDA uyumluluğu için)
- CUDA desteği (isteğe bağlı, CPU modu mevcut)
- 8GB+ RAM (büyük modeller için)

### Adım Adım Kurulum

1. **Projeyi klonlayın**
   ```bash
   git clone <repository-url>
   cd GAN_Image_Generator
   ```

2. **Sanal ortam oluşturun**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Bağımlılıkları yükleyin**
   ```bash
   pip install -r requirements.txt
   ```

4. **Model dosyalarınızı yerleştirin**
   - StyleGAN model dosyalarınızı (`.pkl`) `data/models/` klasörüne koyun
   - Desteklenen modeller: StyleGAN2-ADA, StyleGAN3

5. **StyleGAN kaynak kodlarını indirin**
   ```bash
   # StyleGAN2-ADA
   git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
   
   # StyleGAN3
   git clone https://github.com/NVlabs/stylegan3.git
   ```

## Kullanım

### Ana Uygulama

Streamlit uygulamasını başlatın:

```bash
streamlit run app.py
```

Tarayıcınızda açılan arayüzde:

1. **Model Seçimi**: Sol panelden model dosyasını seçin
2. **Parametreler**: Seed değerleri, truncation ve diğer ayarları yapın
3. **Üretim**: "Generate Images" butonuna tıklayın
4. **İndirme**: Tek tek veya toplu olarak görüntüleri indirin

### Komut Satırı Araçları

#### StyleGAN Analizi
```bash
# Kapsamlı analiz
python scripts/comprehensive_analysis.py --metrics-dir data/metrics

# FID görselleştirme
python scripts/fid_visualization.py --metrics-dir data/metrics --output-dir outputs/plots
```

#### Stable Diffusion LoRA Eğitimi
```bash
# LoRA modeli eğitimi
python scripts/train_lora.py \
    --data_dir "path/to/fish_disease_dataset" \
    --output_dir "outputs/lora_models" \
    --epochs 20 \
    --batch_size 2 \
    --learning_rate 1e-4
```

#### Stable Diffusion Görüntü Üretimi
```bash
# LoRA modeli ile görüntü üretimi
python scripts/generate_images.py \
    --lora_weights "outputs/lora_models/best_model" \
    --output_dir "outputs/generated_images" \
    --num_samples 5
```

## Proje Yapısı

```
GAN_Image_Generator/
├── app.py                 # Ana Streamlit uygulaması
├── requirements.txt       # Python bağımlılıkları
├── config/
│   └── settings.py       # Yapılandırma ayarları
├── src/
│   ├── generators/       # StyleGAN model yönetimi ve görüntü üretimi
│   ├── diffusion/        # Stable Diffusion LoRA eğitimi ve inference
│   ├── analysis/         # Metrik analizi araçları
│   ├── ui/              # Kullanıcı arayüzü bileşenleri  
│   └── utils/           # Yardımcı fonksiyonlar
├── scripts/             # Komut satırı araçları
│   ├── comprehensive_analysis.py  # GAN eğitim analizi
│   ├── fid_visualization.py       # FID görselleştirme
│   ├── train_lora.py              # LoRA model eğitimi
│   └── generate_images.py         # LoRA ile görüntü üretimi
├── data/
│   ├── models/          # Model dosyaları (.pkl)
│   └── metrics/         # Eğitim metrikleri (.jsonl)
├── outputs/             # Üretilen dosyalar
├── docs/               # Dokümantasyon
└── tests/              # Test dosyaları
```

## Konfigürasyon

`config/settings.py` dosyasında özelleştirilebilir ayarlar:

- Model dizin yolları
- Varsayılan parametreler
- UI ayarları
- CUDA konfigürasyonu

## Sorun Giderme

### Yaygın Sorunlar

1. **CUDA Extension Hatası (StyleGAN3)**
   - Uygulama otomatik olarak CPU moduna geçer
   - Python 3.11 veya öncesi kullanın

2. **Distutils Hatası**
   ```bash
   pip install --upgrade setuptools
   ```

3. **Model Yükleme Hatası**
   - Model formatının doğru olduğundan emin olun
   - Dosya yollarını kontrol edin

4. **Bellek Yetersizliği**
   - Daha küçük batch size kullanın
   - CPU moduna geçin

### Performans Optimizasyonu

- CUDA kullanımı için uygun PyTorch versiyonu
- Yeterli GPU belleği (8GB+ önerilir)
- SSD depolama (model yükleme hızı için)

## Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'i push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için LICENSE dosyasına bakın.

---

# GAN Image Generator (English)

Professional application for high-quality image generation using StyleGAN2-ADA, StyleGAN3, and Stable Diffusion models.

## Features

- **Multi-Model Support**: Supports StyleGAN2-ADA, StyleGAN3, and Stable Diffusion models
- **LoRA Fine-tuning**: Customize Stable Diffusion models with LoRA training
- **User-Friendly Interface**: Modern web interface built with Streamlit
- **Batch Image Generation**: Generate multiple images with different seeds simultaneously
- **Advanced Parameters**: Truncation, noise mode, and transform support
- **Analysis Tools**: Comprehensive training metrics analysis and visualization
- **Automatic Error Handling**: Automatic CPU fallback for CUDA extension issues
- **Clean Code Architecture**: SOLID principles and clean architecture implementation

## Installation

### Requirements

- Python 3.8+ (Python 3.11 or earlier recommended for CUDA compatibility)
- CUDA support (optional, CPU mode available)
- 8GB+ RAM (for large models)

### Step-by-Step Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd GAN_Image_Generator
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Place model files**
   - Put your StyleGAN model files (`.pkl`) in `data/models/` folder
   - Supported models: StyleGAN2-ADA, StyleGAN3

5. **Download StyleGAN source codes**
   ```bash
   # StyleGAN2-ADA
   git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
   
   # StyleGAN3
   git clone https://github.com/NVlabs/stylegan3.git
   ```

## Usage

### Main Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

In the opened browser interface:

1. **Model Selection**: Choose model file from the left panel
2. **Parameters**: Set seed values, truncation, and other settings
3. **Generation**: Click "Generate Images" button
4. **Download**: Download images individually or in bulk

### Command Line Analysis

To analyze training metrics:

#### StyleGAN Analysis
```bash
# Comprehensive analysis
python scripts/comprehensive_analysis.py --metrics-dir data/metrics

# FID visualization
python scripts/fid_visualization.py --metrics-dir data/metrics --output-dir outputs/plots
```

#### Stable Diffusion LoRA Training
```bash
# Train LoRA model
python scripts/train_lora.py \
    --data_dir "path/to/fish_disease_dataset" \
    --output_dir "outputs/lora_models" \
    --epochs 20 \
    --batch_size 2 \
    --learning_rate 1e-4
```

#### Stable Diffusion Image Generation
```bash
# Generate images with LoRA model
python scripts/generate_images.py \
    --lora_weights "outputs/lora_models/best_model" \
    --output_dir "outputs/generated_images" \
    --num_samples 5
```

## Project Structure

```
GAN_Image_Generator/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── config/
│   └── settings.py       # Configuration settings
├── src/
│   ├── generators/       # StyleGAN model management and image generation
│   ├── diffusion/        # Stable Diffusion LoRA training and inference
│   ├── analysis/         # Metrics analysis tools
│   ├── ui/              # User interface components
│   └── utils/           # Helper functions
├── scripts/             # Command line tools
│   ├── comprehensive_analysis.py  # GAN training analysis
│   ├── fid_visualization.py       # FID visualization
│   ├── train_lora.py              # LoRA model training
│   └── generate_images.py         # LoRA image generation
├── data/
│   ├── models/          # Model files (.pkl)
│   └── metrics/         # Training metrics (.jsonl)
├── outputs/             # Generated files
├── docs/               # Documentation
└── tests/              # Test files
```

## Configuration

Customizable settings in `config/settings.py`:

- Model directory paths
- Default parameters
- UI settings
- CUDA configuration

## Troubleshooting

### Common Issues

1. **CUDA Extension Error (StyleGAN3)**
   - Application automatically falls back to CPU mode
   - Use Python 3.11 or earlier

2. **Distutils Error**
   ```bash
   pip install --upgrade setuptools
   ```

3. **Model Loading Error**
   - Ensure model format is correct
   - Check file paths

4. **Memory Insufficient**
   - Use smaller batch size
   - Switch to CPU mode

### Performance Optimization

- Appropriate PyTorch version for CUDA usage
- Sufficient GPU memory (8GB+ recommended)
- SSD storage (for model loading speed)

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is licensed under the MIT License. See LICENSE file for details.
