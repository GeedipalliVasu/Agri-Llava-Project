# 🌿 Agri-LLaVA: Leaf Disease Detection Web App

Agri-LLaVA is a web-based system that allows users to upload a crop leaf image and get:
- Disease prediction (Healthy / Early Blight / Late Blight)
- Confidence score
- Heatmap visualization of infected region (Grad-CAM)
- Infection severity %
- Optional future disease stage generation using a diffusion model

Frontend: React (Vite)  
Backend: Flask (Python + PyTorch)  
Database: MongoDB (for history & user data)

---

## ⚙️ Backend Setup (Flask API + PyTorch)

```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py

Backend will run at:

http://127.0.0.1:5000

🖥️ Frontend Setup (React + Vite)
cd agri-llava-frontend
npm install
Create a file named .env in frontend folder:

VITE_API_BASE=http://localhost:5000
Start frontend:

npm run dev
Frontend will run at:

http://localhost:5173
✅ Basic Flow
Start backend (python app.py)

Start frontend (npm run dev)

Open browser → upload leaf image → get prediction

Heatmap and severity % will be displayed

History is stored in MongoDB (if connected)

🔧 Optional GPU Setup (For Diffusion Model Feature)
pip install torch --index-url https://download.pytorch.org/whl/cu121
Then in app.py:

pipe.to("cuda")

🛑 Requirements
Component	Version
Python	3.8+
Node.js	16+
MongoDB	Local or Atlas
