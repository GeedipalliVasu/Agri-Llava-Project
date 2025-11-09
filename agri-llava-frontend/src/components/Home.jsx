// import React, { useState } from "react";
// import DiseaseDetection from "./DiseaseDetection";
// import DiseaseShowing from "./DiseaseShowing";
// import ImageGeneration from "./ImageGeneration";
// import Timeline from "./Timeline";
// import "./Home.css";

// const Home = () => {
//   const [image, setImage] = useState(null);
//   const [showOptions, setShowOptions] = useState(false);
//   const [activeComponent, setActiveComponent] = useState(null);
//   const [sideHeading, setSideHeading] = useState("");

//   const handleUpload = (e) => {
//     const file = e.target.files[0];
//     if (file) setImage(URL.createObjectURL(file));
//   };

//   const handleBack = () => {
//     setActiveComponent(null);
//     setSideHeading("");
//   };

//   return (
//     <div className="home-container">
//       {!image ? (
//         <div className="upload-section">
//           <h2 className="quote">
//             “To forget how to dig the earth and to tend the soil is to forget ourselves.” 🌱
//           </h2>
//           <div className="upload-box">
//             <h3>Upload Crop Image</h3>
//             <input type="file" accept="image/*" onChange={handleUpload} />
//           </div>
//         </div>
//       ) : (
//         <div className="main-layout">
//           {/* Left half – Uploaded Image */}
//           <div className="left-panel">
//             <img src={image} alt="Uploaded Crop" className="uploaded-img" />
//           </div>

//           {/* Right half – Results & Options */}
//           <div className="right-panel">
//             <h2 className="side-heading">
//               {sideHeading || "🌾 Crop Health Analysis Dashboard"}
//             </h2>

//             <div className="content-box">
//               {!activeComponent && <DiseaseDetection />}
//               {activeComponent === "showing" && (
//                 <DiseaseShowing onBack={handleBack} />
//               )}
//               {activeComponent === "image" && (
//                 <ImageGeneration onBack={handleBack} />
//               )}
//               {activeComponent === "timeline" && (
//                 <Timeline onBack={handleBack} />
//               )}
//             </div>

//             {/* Buttons for options */}
//             {!activeComponent && (
//               <div className="options-container">
//                 {!showOptions ? (
//                   <button
//                     className="show-btn"
//                     onClick={() => setShowOptions(true)}
//                   >
//                     Show More Options
//                   </button>
//                 ) : (
//                   <div className="options-grid fade">
//                     <button
//                       className="option"
//                       onClick={() => {
//                         setActiveComponent("showing");
//                         setSideHeading("🌿 Disease Showing");
//                       }}
//                     >
//                       Disease Showing
//                     </button>
//                     <button
//                       className="option"
//                       onClick={() => {
//                         setActiveComponent("image");
//                         setSideHeading("🖼️ Image Generation");
//                       }}
//                     >
//                       Image Generation
//                     </button>
//                     <button
//                       className="option"
//                       onClick={() => {
//                         setActiveComponent("timeline");
//                         setSideHeading("🕒 Project Timeline");
//                       }}
//                     >
//                       Timeline
//                     </button>
//                   </div>
//                 )}
//               </div>
//             )}
//           </div>
//         </div>
//       )}
//     </div>
//   );
// };

// export default Home;
// import React, { useState } from "react";
// import DiseaseDetection from "./DiseaseDetection";
// import DiseaseShowing from "./DiseaseShowing";
// import ImageGeneration from "./ImageGeneration";
// import Timeline from "./Timeline";
// import "./Home.css";

// const Home = () => {
//   const [image, setImage] = useState(null);
//   const [showOptions, setShowOptions] = useState(false);
//   const [activeComponent, setActiveComponent] = useState(null);
//   const [sideHeading, setSideHeading] = useState("");
//   const [prediction, setPrediction] = useState("");

//   const handleUpload = (e) => {
//     const file = e.target.files[0];
//     if (file) {
//       setImage(URL.createObjectURL(file));
//       setPrediction(""); // clear previous prediction
//     }
//   };

//   const handlePredict = () => {
//     // For now, simulate model prediction result
//     setPrediction("🌿 The uploaded leaf is predicted as: Healthy");
//   };

//   const handleBack = () => {
//     setActiveComponent(null);
//     setSideHeading("");
//   };

//   return (
//     <div className="home-container">
//       <div className="main-content">
//         <div className="dashboard-card">
//           <h2>{sideHeading || "🌾 Crop Health Analysis Dashboard"}</h2>

//           {/* Upload section */}
//           <input type="file" accept="image/*" onChange={handleUpload} />

//           {/* Show uploaded image only when selected */}
//           {image && (
//             <>
//               <div className="upload-preview">
//                 <img src={image} alt="Uploaded Leaf" />
//               </div>
//               <button className="predict-btn" onClick={handlePredict}>
//                 Predict
//               </button>
//             </>
//           )}

//           {/* Show prediction result */}
//           {prediction && <div className="result-box">{prediction}</div>}

//           {/* Show other feature components */}
//           <div className="content-box">
//             {activeComponent === "showing" && <DiseaseShowing onBack={handleBack} />}
//             {activeComponent === "image" && <ImageGeneration onBack={handleBack} />}
//             {activeComponent === "timeline" && <Timeline onBack={handleBack} />}
//           </div>

//           {/* Show options only if not inside other components */}
//           {!activeComponent && (
//             <div className="options-container">
//               {!showOptions ? (
//                 <button className="show-btn" onClick={() => setShowOptions(true)}>
//                   Show More Options
//                 </button>
//               ) : (
//                 <div className="options-grid fade">
//                   <button
//                     className="option"
//                     onClick={() => {
//                       setActiveComponent("showing");
//                       setSideHeading("🌿 Disease Showing");
//                     }}
//                   >
//                     Disease Showing
//                   </button>
//                   <button
//                     className="option"
//                     onClick={() => {
//                       setActiveComponent("image");
//                       setSideHeading("🖼️ Image Generation");
//                     }}
//                   >
//                     Image Generation
//                   </button>
//                   <button
//                     className="option"
//                     onClick={() => {
//                       setActiveComponent("timeline");
//                       setSideHeading("🕒 Project Timeline");
//                     }}
//                   >
//                     Timeline
//                   </button>
//                 </div>
//               )}
//             </div>
//           )}
//         </div>
//       </div>
//     </div>
//   );
// };

// export default Home;

// import React, { useState } from "react";
// import { useNavigate } from "react-router-dom";
// import "./Home.css";

// export default function Home() {
//   const [image, setImage] = useState(null);
//   const [showOptions, setShowOptions] = useState(false);
//   const [predictionMessage, setPredictionMessage] = useState("");
//   const navigate = useNavigate();

//   const handleUpload = (e) => {
//     const f = e.target.files[0];
//     if (f) {
//       setImage(URL.createObjectURL(f));
//       setPredictionMessage("");
//     }
//   };

//   const handlePredict = () => {
//     // you can later call your backend API here
//     setPredictionMessage("🌿 The uploaded leaf is predicted as: healthy");
//   };

//   return (
//     <div className="home-container">
//       <div className="home-card">
//         <h2>🌾 Crop Health Analysis</h2>

//         <input type="file" accept="image/*" onChange={handleUpload} />

//         {image && (
//           <>
//             <div className="image-preview">
//               <img src={image} alt="Uploaded Leaf" />
//             </div>
//             <button className="predict-btn" onClick={handlePredict}>
//               Predict
//             </button>
//           </>
//         )}

//         {predictionMessage && (
//           <div className="result-box">{predictionMessage}</div>
//         )}

//         <div className="options-section">
//           {!showOptions ? (
//             <button
//               className="show-more-btn"
//               onClick={() => setShowOptions(true)}
//             >
//               Show More Options
//             </button>
//           ) : (
//             <div className="options-list">
//               <button onClick={() => navigate("/showing")}>Disease Showing</button>
//               <button onClick={() => navigate("/image")}>Image Generation</button>
//               <button onClick={() => navigate("/timeline")}>Timeline</button>
//               <button onClick={() => navigate("/history")}>History</button>
//             </div>
//           )}
//         </div>
//       </div>
//     </div>
//   );
// }

import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import "./Home.css";
import { FaLeaf, FaClock, FaHistory, FaImage } from "react-icons/fa";

const Home = () => {
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [showCamera, setShowCamera] = useState(false);
  const [stream, setStream] = useState(null);
  const videoRef = React.useRef(null);
  const navigate = useNavigate();
  const auth = useAuth();

  const handleUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(URL.createObjectURL(file));
      setPrediction(""); // clear previous prediction
    }
  };

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'environment' } // Use back camera on mobile
      });
      setStream(mediaStream);
      setShowCamera(true);
      
      // Use setTimeout to ensure DOM is updated
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
          videoRef.current.play().catch(err => {
            console.error("Play error:", err);
            alert("Could not start video. Please try again.");
          });
        }
      }, 100);
    } catch (err) {
      console.error("Camera access error:", err);
      alert("Could not access camera. Please check permissions.");
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setShowCamera(false);
  };

  const capturePhoto = () => {
    if (!videoRef.current) {
      alert("Camera not ready. Please wait a moment and try again.");
      return;
    }

    const video = videoRef.current;
    
    // Check if video is ready
    if (video.readyState !== video.HAVE_ENOUGH_DATA && video.readyState !== video.HAVE_CURRENT_DATA) {
      alert("Video is not ready yet. Please wait a moment.");
      return;
    }

    // Get dimensions with fallback
    let width = video.videoWidth;
    let height = video.videoHeight;
    
    if (width === 0 || height === 0) {
      // Fallback to video element dimensions
      width = video.clientWidth || 640;
      height = video.clientHeight || 480;
    }

    if (width === 0 || height === 0) {
      alert("Unable to get video dimensions. Please try again.");
      return;
    }

    try {
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      
      // Draw the video frame to canvas
      ctx.drawImage(video, 0, 0, width, height);
      
      // Convert to blob
      canvas.toBlob((blob) => {
        if (!blob) {
          alert("Failed to capture image. Please try again.");
          return;
        }
        
        try {
          const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
          const imageUrl = URL.createObjectURL(file);
          setImage(imageUrl);
          setPrediction("");
          
          // Update the file input to include the captured image
          const dataTransfer = new DataTransfer();
          dataTransfer.items.add(file);
          const fileInput = document.querySelector("input[type='file']");
          if (fileInput) {
            fileInput.files = dataTransfer.files;
          }
          
          // Stop camera after successful capture
          stopCamera();
        } catch (err) {
          console.error("Error creating file:", err);
          alert("Error processing captured image. Please try again.");
        }
      }, 'image/jpeg', 0.95);
    } catch (err) {
      console.error("Capture error:", err);
      alert("Failed to capture image: " + err.message);
    }
  };

  // Ensure video plays when modal opens
  useEffect(() => {
    if (showCamera && stream && videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.play().catch(err => {
        console.error("Video play error:", err);
      });
    }
  }, [showCamera, stream]);

  // Cleanup camera stream on unmount
  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [stream]);

  const handlePredict = async () => {
  const fileInput = document.querySelector("input[type='file']");
  if (!fileInput || !fileInput.files.length) {
    alert("Please upload an image first!");
    return;
  }

  const formData = new FormData();
  formData.append("image", fileInput.files[0]);

    try {
    const headers = {};
    if (auth && auth.token) headers.Authorization = `Bearer ${auth.token}`;
    const res = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formData,
      headers,
    });

    const data = await res.json();
    console.log("📩 Response from backend:", data);

    if (data.error) {
      alert(data.error);
      return;
    }

    setPrediction(data.message || "No message returned");

    // Show GradCAM image if available
    if (data.cam_image) {
      setPreview(data.cam_image);
    }
  } catch (error) {
    console.error("Prediction error:", error);
    alert("Error connecting to backend.");
  }
};




  return (
    <div className="home-container">
      {/* Upload Section - Always visible */}
      <div className="upload-header">
        <h2>🌿 Crop Disease Detection</h2>
        <p className="home-description">
          Upload a crop leaf image to detect plant diseases and maintain healthy growth.
        </p>
        <div className="upload-options">
          <label className="upload-btn">
            📁 Choose from Gallery
            <input type="file" accept="image/*" onChange={handleUpload} style={{ display: 'none' }} />
          </label>
          <button className="camera-btn" onClick={startCamera}>
            📷 Capture from Camera
          </button>
        </div>
      </div>

      {/* Camera Modal */}
      {showCamera && (
        <div className="camera-modal">
          <div className="camera-content">
            <video 
              ref={videoRef} 
              autoPlay 
              playsInline 
              muted
              className="camera-video"
              onLoadedMetadata={() => {
                if (videoRef.current) {
                  videoRef.current.play().catch(err => console.error("Play error:", err));
                }
              }}
            ></video>
            <div className="camera-controls">
              <button className="capture-btn" onClick={capturePhoto}>
                📸 Capture
              </button>
              <button className="cancel-btn" onClick={stopCamera}>
                ❌ Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Main Display Section */}
      {image && (
        <div className="dashboard-card">
          <h3>Preview:</h3>
          <div className="upload-preview">
            <img src={image} alt="Uploaded Crop" />
          </div>
          <button className="predict-btn" onClick={handlePredict}>
            Detect Disease
          </button>

          {prediction && (
  <div className="result-box">
    <strong>Result:</strong> {prediction}
  </div>
)}

        </div>
      )}

      {/* Options Section */}
      <div className="home-options">
        <h3 className="option-heading">Explore More</h3>
        <div className="option-grid">
          <div className="option-card" onClick={() => navigate("/image")}>
            <FaImage className="option-icon" />
            <h4>Image Generation</h4>
            <p>Generate synthetic plant images using AI.</p>
          </div>

          <div className="option-card" onClick={() => navigate("/showing")}>
            <FaLeaf className="option-icon" />
            <h4>Disease Visualization</h4>
            <p>View common crop diseases and symptoms.</p>
          </div>

          <div className="option-card" onClick={() => navigate("/timeline")}>
            <FaClock className="option-icon" />
            <h4>Timeline</h4>
            <p>Track your project milestones and updates.</p>
          </div>

          <div className="option-card" onClick={() => navigate("/history")}>
            <FaHistory className="option-icon" />
            <h4>History</h4>
            <p>Review your previous disease detection results.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;



