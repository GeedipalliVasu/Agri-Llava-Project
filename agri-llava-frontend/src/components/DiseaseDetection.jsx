import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import "./DiseaseDetection.css";
import { useAuth } from "../context/AuthContext";
import { validateImage } from "../utils/imageValidation";

function DiseaseDetection() {
  const navigate = useNavigate();
  const auth = useAuth();
  const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:5000";
  const isProduction = API_BASE.includes("onrender.com");
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState("");
  const [resp, setResp] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    try {
      setLoading(true);
      await validateImage(file);

      // Create preview URL immediately and use it directly when predicting
      const previewUrl = URL.createObjectURL(file);
      setImage(file);
      setPreview(previewUrl);
      setResult(''); // Clear previous results

      // Automatically predict after successful image upload, pass file and previewUrl to avoid state race
      await handlePredict(file, previewUrl);
    } catch (error) {
      alert(error.message);
      e.target.value = ''; // Reset file input
      setLoading(false);
    }
  };

  // handlePredict accepts optional fileOverride and previewOverride so callers
  // (like handleImageChange) can pass the uploaded file directly and avoid
  // relying on state which updates asynchronously.
  const handlePredict = async (fileOverride = null, previewOverride = null) => {
    const fileToUse = fileOverride || image;
    const previewToUse = previewOverride || preview;

    if (!fileToUse) {
      alert("Please upload an image first!");
      setLoading(false);
      return;
    }

    setLoading(true);

    try {
      // Validate again before prediction
      await validateImage(fileToUse);
      const formData = new FormData();
      formData.append("image", fileToUse);

      const headers = { "Content-Type": "multipart/form-data" };
      if (auth && auth.token) headers.Authorization = `Bearer ${auth.token}`;
      const res = await axios.post(`${API_BASE}/predict`, formData, {
        headers,
      });

      const prediction = res.data.prediction;
      setResult(res.data.message);
      setResp(res.data);

      // 🧠 Save history in localStorage
      const history = JSON.parse(localStorage.getItem("predictionHistory")) || [];
      const newEntry = {
        image: previewToUse,
        prediction: prediction,
        time: new Date().toLocaleString(),
      };
      localStorage.setItem("predictionHistory", JSON.stringify([newEntry, ...history]));

      // Save for Image Generation page quick access
      localStorage.setItem("uploadedImage", previewToUse);
      localStorage.setItem("predictionResult", prediction);
    } catch (err) {
      console.error("Prediction error:", err);
      if (err.message && err.message.includes("upload a")) {
        // This is from our image validation
        alert(err.message);
      } else {
        alert("Error predicting image. Check backend connection!");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="disease-container">
      <h2>🌾 Crop Disease Detection</h2>
      {isProduction && (
        <div style={{ 
          backgroundColor: '#fff3cd', 
          border: '1px solid #ffc107', 
          padding: '12px', 
          borderRadius: '4px', 
          marginBottom: '16px',
          color: '#856404'
        }}>
          <p><strong>Note:</strong> Disease detection is currently unavailable on this deployment (requires GPU). 
          Please use a local version or contact support for GPU deployment options.</p>
        </div>
      )}
      <div className="upload-section">
        <input type="file" accept="image/*" onChange={handleImageChange} disabled={isProduction} />
        {preview && <img src={preview} alt="Preview" className="preview-img" />}
        <button onClick={handlePredict} disabled={loading || isProduction}>
          {isProduction ? "Unavailable (GPU Required)" : loading ? "Predicting..." : "Predict"}
        </button>
      </div>

      {result && (
        <div className="result-box">
          <h3>🧬 Result:</h3>
          <p>{result}</p>
          {resp && !resp.prediction.toLowerCase().includes("healthy") && (
            <div style={{ marginTop: 12, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
              {resp.heatmap && (
                <img src={`http://127.0.0.1:5000/${resp.heatmap}`} alt="heatmap" className="heatmap-img" />
              )}
              {resp.segmentation && (
                <img src={`http://127.0.0.1:5000/${resp.segmentation}`} alt="segmentation" className="heatmap-img" />
              )}
            </div>
          )}
          <div style={{ marginTop: 12 }}>
            <button className="predict-btn" onClick={() => navigate('/image')}>Go to Image Generation</button>
          </div>
        </div>
      )}
    </div>
  );
}

export default DiseaseDetection;







// ------------------------------------------------------------------
// import React, { useState } from "react";
// import "./DiseaseShowing.css";

// const DiseaseShowing = () => {
//   const [image, setImage] = useState(null);
//   const [result, setResult] = useState(null);
//   const [loading, setLoading] = useState(false);

//   const handleUpload = (e) => {
//     const file = e.target.files[0];
//     if (file) {
//       setImage(file);
//       setResult(null);
//     }
//   };

//   const handlePredict = async () => {
//     if (!image) return alert("Please upload an image!");

//     const formData = new FormData();
//     formData.append("image", image);
//     setLoading(true);

//     try {
//       const res = await fetch("http://127.0.0.1:5000/predict", {
//         method: "POST",
//         body: formData,
//       });

//       const data = await res.json();
//       setResult(data);
//     } catch (err) {
//       console.error(err);
//       alert("Error connecting to backend.");
//     } finally {
//       setLoading(false);
//     }
//   };

//   return (
//     <div className="disease-showing-container">
//       <h2>🌿 Disease Highlight Visualization</h2>
//       <p>Upload a crop leaf image to visualize the infected region (Grad-CAM).</p>

//       <div className="upload-section">
//         <input type="file" accept="image/*" onChange={handleUpload} />
//         <button className="predict-btn" onClick={handlePredict}>
//           {loading ? "Analyzing..." : "Show Diseased Area"}
//         </button>
//       </div>

//       {image && (
//         <div className="preview-section">
//           <div>
//             <h4>Original Image</h4>
//             <img
//               src={URL.createObjectURL(image)}
//               alt="Uploaded"
//               className="uploaded-img"
//             />
//           </div>

//           {/* === THIS LINE IS MODIFIED === */}
//           {/* We now check if the prediction is NOT healthy */}
//           {result && !result.prediction.toLowerCase().includes("healthy") && result.heatmap && (
//             <div>
//               <h4>Highlighted Diseased Area</h4>
//               <img
//                 src={`http://127.0.0.1:5000/${result.heatmap}`}
//                 alt="Grad-CAM Heatmap"
//                 className="heatmap-img"
//               />
//             </div>
//           )}

//           {/* === THIS LINE IS MODIFIED === */}
//           {/* We now check if the prediction INCLUDES healthy */}
//           {result && result.prediction.toLowerCase().includes("healthy") && (
//             <div className="healthy-note">
//               ✅ The leaf is healthy — no diseased area to show!
//             </div>
//           )}
//         </div>
//       )}
//     </div>
//   );
// };

// export default DiseaseShowing;
