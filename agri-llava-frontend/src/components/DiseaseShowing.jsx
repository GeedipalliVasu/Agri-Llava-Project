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

//           {result && result.prediction === "diseased" && result.heatmap && (
//             <div>
//               <h4>Highlighted Diseased Area</h4>
//               <img
//                 src={`http://127.0.0.1:5000/${result.heatmap}`}
//                 alt="Grad-CAM Heatmap"
//                 className="heatmap-img"
//               />
//             </div>
//           )}

//           {result && result.prediction === "healthy" && (
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









import React, { useState } from "react";
import "./DiseaseShowing.css";

const DiseaseShowing = () => {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setResult(null);
    }
  };

  const handlePredict = async () => {
    if (!image) return alert("Please upload an image!");

    const formData = new FormData();
    formData.append("image", image);
    setLoading(true);

    try {
      const res = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      alert("Error connecting to backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="disease-showing-container">
      <h2>🌿 Disease Highlight Visualization</h2>
      <p>Upload a crop leaf image to visualize the infected region (Grad-CAM).</p>

      <div className="upload-section">
        <input type="file" accept="image/*" onChange={handleUpload} />
        <button className="predict-btn" onClick={handlePredict}>
          {loading ? "Analyzing..." : "Show Diseased Area"}
        </button>
      </div>

      {image && (
        <div className="preview-section">
          {/* If diseased: show Original + Heatmap on top row, Segmentation below */}
          {result && !result.prediction.toLowerCase().includes("healthy") && (result.heatmap || result.segmentation) ? (
            <>
              <div className="two-view">
                <div>
                  <h4>Original</h4>
                  <img
                    src={URL.createObjectURL(image)}
                    alt="Uploaded"
                    className="uploaded-img"
                  />
                </div>
                <div>
                  <h4>Heatmap (Grad-CAM)</h4>
                  {result.heatmap ? (
                    <img
                      src={`http://127.0.0.1:5000/${result.heatmap}`}
                      alt="Grad-CAM Heatmap"
                      className="heatmap-img"
                    />
                  ) : (
                    <div className="healthy-note">Heatmap unavailable</div>
                  )}
                </div>
              </div>

              {result.segmentation && (
                <div className="segmentation-below">
                  <h4>Segmentation (Diseased Regions)</h4>
                  <img
                    src={`http://127.0.0.1:5000/${result.segmentation}`}
                    alt="Segmentation Overlay"
                    className="heatmap-img"
                  />
                  {Array.isArray(result.legend) && result.legend.length > 0 && (
                    <div style={{ marginTop: 8 }}>
                      <strong>Legend:</strong>
                      <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 4, justifyContent: "center" }}>
                        <span
                          style={{
                            display: "inline-block",
                            width: 16,
                            height: 16,
                            backgroundColor: result.legend[0].color,
                            border: "1px solid #999",
                          }}
                        />
                        <span>{result.legend[0].label}</span>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </>
          ) : (
            // Healthy case: show original and healthy note
            <div>
              <div>
                <h4>Original</h4>
                <img
                  src={URL.createObjectURL(image)}
                  alt="Uploaded"
                  className="uploaded-img"
                />
              </div>
              {result && result.prediction.toLowerCase().includes("healthy") && (
                <div className="healthy-note" style={{ marginTop: 16 }}>
                  ✅ The leaf is healthy — no diseased area to show!
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default DiseaseShowing;
