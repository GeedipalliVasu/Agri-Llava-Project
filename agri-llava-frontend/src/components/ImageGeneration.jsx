import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import "./ImageGeneration.css";

const ImageGeneration = () => {
  const navigate = useNavigate();
  const auth = useAuth();
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [disease, setDisease] = useState("Early Blight");
  const [stages, setStages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const onFileChange = (e) => {
    const f = e.target.files && e.target.files[0];
    if (f) {
      setFile(f);
      setPreview(URL.createObjectURL(f));
      setStages([]);
      setError("");
    }
  };

  const onGenerate = async () => {
    if (!file) {
      alert("Please upload a base image first.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const form = new FormData();
      form.append("image", file);
      form.append("disease", disease);
      const headers = {};
      if (auth && auth.token) headers.Authorization = `Bearer ${auth.token}`;
      const res = await fetch("http://127.0.0.1:5000/generate_stages", {
        method: "POST",
        body: form,
        headers,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Generation failed");
      setStages(data.stages || []);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="image-generation-container">
      <h2>🖼️ Next-Stage Disease Generation</h2>
      <p>
        Upload a leaf image and generate plausible next-stage disease visuals.
      </p>

      <div className="image-gen-card">
        <div
          style={{
            display: "flex",
            gap: 16,
            alignItems: "center",
            flexWrap: "wrap",
          }}
        >
          <input type="file" accept="image/*" onChange={onFileChange} />

          <button className="back-btn" onClick={onGenerate} disabled={loading}>
            {loading ? "Generating..." : "Generate Stages"}
          </button>
          <button className="back-btn" onClick={() => navigate("/")}>
            ⬅ Back to Home
          </button>
        </div>

        {error && (
          <div style={{ color: "#b00020", marginTop: 12 }}>{error}</div>
        )}

        {preview && (
          <div style={{ marginTop: 20 }}>
            <h3>Base Image</h3>
            <img src={preview} alt="Base" className="gen-image" />
          </div>
        )}

        {stages.length > 0 && (
          <div style={{ marginTop: 24 }}>
            <h3>Generated Stages</h3>
            <div className="stages-grid">
              {stages.map((p, idx) => (
                <div key={idx} className="stage-item">
                  <img
                    src={`http://127.0.0.1:5000/${p}`}
                    alt={`Stage ${idx + 1}`}
                    className="gen-image"
                  />
                  <div style={{ marginTop: 6, fontWeight: 600 }}>
                    Stage {idx + 1}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageGeneration;
