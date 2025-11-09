import React, { useState, useRef } from "react";
import "./DiseaseVisualization.css";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:5000";

export default function DiseaseVisualization() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [overlayOn, setOverlayOn] = useState(true);
  const [opacity, setOpacity] = useState(0.6);
  const canvasRef = useRef(null);

  const handleFile = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setResult(null);
  };

  const handlePredict = async () => {
    if (!file) return alert("Please upload an image first.");
    setLoading(true);
    const fd = new FormData();
    fd.append("image", file);
  // no preset params — backend will use defaults; still request inferno colormap
  fd.append('colormap', 'inferno');
    try {
      const res = await fetch(`${API_BASE}/predict`, { method: 'POST', body: fd, credentials: 'include' });
      if (!res.ok) {
        const text = await res.text().catch(()=>null);
        console.error('Predict failed', res.status, text);
        alert(`Prediction failed: ${res.status} ${res.statusText}\n${text || ''}`);
      } else {
        const data = await res.json();
        setResult(data);
      }
    } catch (e) {
      console.error('Predict request error', e);
      alert("Prediction failed — check backend logs or CORS. See console for details.");
    } finally {
      setLoading(false);
    }
  };

  // Draw combined image on canvas for download
  const handleDownload = async () => {
    if (!previewUrl) return;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = previewUrl;
    img.onload = async () => {
      const canvas = canvasRef.current;
      // use natural size for best resolution
      const iw = img.naturalWidth || img.width;
      const ih = img.naturalHeight || img.height;
      canvas.width = iw;
      canvas.height = ih;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0,0,canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      if (overlayOn && result && result.heatmap) {
        const heat = new Image();
        heat.crossOrigin = 'anonymous';
        heat.src = `${API_BASE}/${result.heatmap}`;
        heat.onload = () => {
          // draw heatmap scaled to the same canvas size
          ctx.globalAlpha = opacity;
          ctx.drawImage(heat, 0, 0, canvas.width, canvas.height);
          ctx.globalAlpha = 1.0;
          const url = canvas.toDataURL('image/png');
          const a = document.createElement('a');
          a.href = url;
          a.download = 'disease_visualization.png';
          a.click();
        };
        heat.onerror = () => alert('Could not load heatmap image for download');
      } else {
        const url = canvas.toDataURL('image/png');
        const a = document.createElement('a');
        a.href = url;
        a.download = 'original.png';
        a.click();
      }
    };
    img.onerror = () => alert('Could not load source image for download');
  };

  return (
    <div className="disease-vis-container">
      <h2>🌿 Disease Visualization</h2>
      <p className="muted">Upload a leaf image and visualize model heatmap and segmentation overlays.</p>

      <div className="controls">
        <input type="file" accept="image/*" onChange={handleFile} />
        <button className="btn" onClick={handlePredict} disabled={loading || !file}>
          {loading ? 'Analyzing…' : 'Analyze'}
        </button>
        <button className="btn secondary" onClick={handleDownload} disabled={!previewUrl}>
          Download Result
        </button>
      </div>

      {previewUrl && (
        <div className="vis-area">
          <div className="image-stack">
            <img src={previewUrl} alt="uploaded" className="base-img" />
            {result && result.heatmap && overlayOn && (
              <img
                src={`${API_BASE}/${result.heatmap}`}
                alt="heatmap"
                className="overlay-img"
                style={{ opacity }}
              />
            )}
          </div>

          <div className="vis-info">
            {result ? (
              <>
                <div className="pred-line">
                  <strong>Prediction:</strong> {result.prediction}
                </div>
                {result.confidence_percent && (
                  <div className="pred-line">
                    <strong>Confidence:</strong> {result.confidence_percent}
                  </div>
                )}

                <div className="controls-row">
                  <label>
                    <input type="checkbox" checked={overlayOn} onChange={(e)=>setOverlayOn(e.target.checked)} /> Overlay heatmap
                  </label>
                  <label className="opacity-control">
                    Opacity
                    <input type="range" min="0" max="1" step="0.05" value={opacity} onChange={(e)=>setOpacity(parseFloat(e.target.value))} />
                  </label>
                </div>

                {result.segmentation && (
                  <div style={{ marginTop: 10 }}>
                    <strong>Segmentation:</strong>
                    <div className="seg-preview">
                      <img src={`${API_BASE}/${result.segmentation}`} alt="seg" />
                    </div>
                  </div>
                )}

                {Array.isArray(result.legend) && result.legend.length > 0 && (
                  <div className="legend">
                    {result.legend.map((item, idx) => (
                      <div key={idx} className="legend-item">
                        <span className="color-box" style={{ background: item.color }} />
                        <span>{item.label}</span>
                      </div>
                    ))}
                  </div>
                )}
              </>
            ) : (
              <p className="muted">No analysis yet. Upload an image and click Analyze.</p>
            )}
          </div>
        </div>
      )}

      <canvas ref={canvasRef} style={{ display: 'none' }} />
    </div>
  );
}
