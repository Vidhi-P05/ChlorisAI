# 🌸 ChlorisAI - Discover the Language of Flowers

A beautiful, intelligent web application for flower identification with comprehensive botanical information.

## ✨ Features

- **🌺 Accurate Identification**: Advanced AI model for 102 flower species
- **📚 Rich Information**: Scientific names, medicinal uses, habitat, blooming seasons
- **🎨 Beautiful Interface**: Modern, responsive design with floating petals animation
- **⚡ Fast Performance**: Optimized FastAPI backend with instant predictions
- **🌿 Educational**: Learn about flowers with detailed botanical information

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install fastapi uvicorn python-multipart torch torchvision pillow jinja2
```

### 2. Run Application
```bash
python app.py
```

### 3. Open Browser
Go to **http://localhost:8080**

## 📁 Project Structure

```
ChlorisAI/
├── app.py                    # FastAPI backend with model
├── templates/
│   ├── index.html           # Beautiful frontend with animations
│   └── static/
│       └── js/
│           └── main.js      # Interactive JavaScript
├── checkpoints/
│   └── best_model.pth      # Your trained model
├── data/
│   ├── class_names.json      # Flower class names
│   └── flower_database.json # Comprehensive flower info
└── README.md                # This file
```

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Beautiful web interface |
| `/predict` | POST | Upload image and get prediction |
| `/health` | GET | Health check |

## 📱 Usage Experience

1. **🌸 Upload**: Click or drag-and-drop your flower image
2. **👁 Preview**: See your image with elegant preview
3. **🔍 Analyze**: Click "Identify Flower" for AI analysis
4. **📊 Results**: View comprehensive flower information:
   - Flower name (common name)
   - Scientific name (medical/botanical name)
   - Confidence score with animated progress bar
   - Detailed description
   - Medicinal uses
   - Natural habitat
   - Blooming season

## 🎨 Design Features

- **Floating Petals**: Animated background effects
- **Gradient Backgrounds**: Beautiful botanical color schemes
- **Responsive Design**: Works on all devices
- **Smooth Animations**: Hover effects and transitions
- **Modern Typography**: Clean, readable fonts
- **Interactive Elements**: Progress bars and micro-interactions

## 📊 Output Format

```json
{
  "flower_name": "pink primrose",
  "confidence": 0.95,
  "info": {
    "scientific_name": "Primula vulgaris",
    "description": "A delicate perennial flower...",
    "medicinal_use": "Traditional uses...",
    "habitat": "Woodlands and meadows...",
    "blooming_season": "Spring to Summer"
  }
}
```

## 🔧 Technical Details

- **Model**: EfficientNet-B0 (102 classes)
- **Backend**: FastAPI with Jinja2 templates
- **Frontend**: Modern HTML5 with CSS3 animations
- **Image Processing**: PIL with torchvision transforms
- **Device Support**: CPU/GPU automatic detection

## 🌿 Flower Database

Comprehensive information for all 102 flowers including:
- Scientific names (Latin binomial nomenclature)
- Botanical families
- Detailed descriptions
- Traditional medicinal uses
- Natural habitats
- Blooming seasons
- Geographic origins

## 🚀 Deployment

### Development Server
```bash
python app.py
```

### Production Server
```bash
pip install gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["python", "app.py"]
```

## 📋 Requirements

- Python 3.8+
- FastAPI & Uvicorn
- PyTorch & TorchVision
- PIL (Pillow)
- Jinja2 (for templates)
- NumPy

## 🎯 Model Performance

- **Accuracy**: 87%+ on test set
- **Input Size**: 224×224 RGB images
- **Model Size**: ~57MB
- **Inference Time**: <1 second per image
- **Classes**: 102 flower species

## 🌟 Special Features

- **🌸 ChlorisAI Branding**: Beautiful logo and branding
- **🎨 Botanical Theme**: Nature-inspired color palette
- **📚 Educational Value**: Learn about flowers while identifying
- **⚡ Real-time Processing**: Instant AI predictions
- **🌿 Comprehensive Data**: 102 flowers with detailed information

---

**🌻 ChlorisAI - Where AI Meets Botanical Beauty**

*Discover the language of flowers with cutting-edge artificial intelligence.*
