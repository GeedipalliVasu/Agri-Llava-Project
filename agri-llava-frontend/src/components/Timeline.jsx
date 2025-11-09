import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:5000";

export default function Timeline() {
  const navigate = useNavigate();
  const auth = useAuth();
  const [farmerId, setFarmerId] = useState(auth?.user?.id || "farmer1");
  const [plantId, setPlantId] = useState("plant1");
  const [file, setFile] = useState(null);
  const [crops, setCrops] = useState([
    "potato",
    "tomato",
    "apple",
    "pepper bell",
  ]);
  const [selectedCrop, setSelectedCrop] = useState(crops[0]);
  const [newCrop, setNewCrop] = useState("");
  const [disease, setDisease] = useState("Unknown");
  const [preview, setPreview] = useState("");
  const [loading, setLoading] = useState(false);
  const [trend, setTrend] = useState("insufficient");
  const [entries, setEntries] = useState([]);
  const [error, setError] = useState("");
  const [filterDisease, setFilterDisease] = useState("all");
  const [modalState, setModalState] = useState({
    open: false,
    src: "",
    title: "",
  });
  const [confirmDialog, setConfirmDialog] = useState({
    open: false,
    type: null,
  });

  const loadTimeline = async (id) => {
    try {
      setError("");
      const headers = {};
      if (auth && auth.token) headers.Authorization = `Bearer ${auth.token}`;
  const url = `${API_BASE}/timeline?farmer_id=${encodeURIComponent(id)}&plant_id=${encodeURIComponent(plantId)}&crop=${encodeURIComponent(selectedCrop)}`;
      const res = await fetch(url, { credentials: "include" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to load timeline");
      setTrend(data.trend);
      setEntries(data.entries || []);
    } catch (e) {
      setError(e.message);
    }
  };

  useEffect(() => {
    loadTimeline(farmerId);
  }, [farmerId, plantId, selectedCrop]);

  // Fetch persisted crops for this farmer from the server
  const fetchCrops = async (fid) => {
    try {
      const res = await fetch(`${API_BASE}/timeline/crops?farmer_id=${encodeURIComponent(fid)}`, { credentials: 'include' });
      const data = await res.json();
      if (res.ok && data.crops && Array.isArray(data.crops) && data.crops.length > 0) {
        setCrops(data.crops);
        // if current selected is not in list, pick first
        if (!data.crops.includes(selectedCrop)) setSelectedCrop(data.crops[0]);
      }
    } catch (e) {
      // ignore, keep defaults
    }
  };

  useEffect(() => {
    fetchCrops(farmerId);
  }, [farmerId]);

  const onFileChange = (e) => {
    const f = e.target.files && e.target.files[0];
    if (f) {
      setFile(f);
      setPreview(URL.createObjectURL(f));
    }
  };

  // Image handling functions
  const openImage = (src, title) => {
    if (!src) return;
    setModalState({ open: true, src, title: title || "image" });
  };

  const closeModal = () => setModalState({ open: false, src: "", title: "" });

  const handleClearTimeline = async () => {
    if (!confirm("Are you sure you want to clear all timeline entries? This action cannot be undone.")) {
      return;
    }
    try {
      setError("");
      const headers = {};
      if (auth && auth.token) headers.Authorization = `Bearer ${auth.token}`;
      
      const url = `${API_BASE}/timeline/clear`;
      const res = await fetch(url, {
        method: "POST",
        headers: {
          ...headers,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ farmer_id: farmerId }),
        credentials: "include"
      });
      
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to clear timeline");
      
      // Reset state after successful clear
      setEntries([]);
      setTrend("insufficient");
      alert("Timeline cleared successfully");
    } catch (e) {
      setError(e.message);
      alert("Failed to clear timeline: " + e.message);
    }
  };

  const handleEndTimeline = async () => {
    if (!confirm("Are you sure you want to end this timeline? This will mark it as completed and archive it.")) {
      return;
    }
    try {
      setError("");
      const headers = {};
      if (auth && auth.token) headers.Authorization = `Bearer ${auth.token}`;
      
      const url = `${API_BASE}/timeline/end`;
      const res = await fetch(url, {
        method: "POST",
        headers: {
          ...headers,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ farmer_id: farmerId }),
        credentials: "include"
      });
      
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to end timeline");
      
      alert("Timeline ended successfully");
      navigate("/"); // Redirect to home after ending
    } catch (e) {
      setError(e.message);
      alert("Failed to end timeline: " + e.message);
    }
  };

  const downloadImage = async (url, filename) => {
    try {
      const res = await fetch(url, { credentials: "include" });
      const blob = await res.blob();
      const a = document.createElement("a");
      const urlObj = URL.createObjectURL(blob);
      a.href = urlObj;
      a.download = filename || "image.jpg";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(urlObj);
    } catch (e) {
      alert("Download failed: " + e.message);
    }
  };

  const onUpload = async () => {
    if (!file) {
      alert("Please choose an image to upload.");
      return;
    }
    setLoading(true);
    setError("");
    try {
  const form = new FormData();
  form.append("image", file);
  form.append("farmer_id", farmerId);
  form.append("plant_id", plantId);
  form.append("crop", selectedCrop);
  form.append("disease", disease);
      const url = `${API_BASE}/timeline/upload`;
      const res = await fetch(url, {
        method: "POST",
        body: form,
        credentials: "include",
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Upload failed");
      setTrend(data.trend || "insufficient");
      await loadTimeline(farmerId);
      setFile(null);
      setPreview("");
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const trendColor =
    trend === "increasing"
      ? "#b00020"
      : trend === "decreasing"
        ? "#207820"
        : "#b8860b";

  const parseTs = (s) => {
    if (!s) return null;
    return new Date(s.replace(" ", "T"));
  };

  // Calculate timeline statistics and summary
  const stats = React.useMemo(() => {
    if (!entries.length)
      return {
        totalEntries: 0,
        avgSeverity: 0,
        maxSeverity: 0,
        diseaseCounts: {},
        lastUpdate: null,
        weeksTracked: 0,
        mostCommonDisease: null,
      };

    const diseaseCount = {};
    let totalSeverity = 0;
    let maxSeverity = 0;
    const uniqueWeeks = new Set();

    entries.forEach((entry) => {
      // Count diseases
      if (entry.prediction) {
        diseaseCount[entry.prediction] =
          (diseaseCount[entry.prediction] || 0) + 1;
      }

      // Track severity
      if (entry.severity_percent != null) {
        totalSeverity += entry.severity_percent;
        maxSeverity = Math.max(maxSeverity, entry.severity_percent);
      }

      // Count unique weeks
      if (entry.week_index != null) {
        uniqueWeeks.add(entry.week_index);
      }
    });

    // Find most common disease
    const mostCommonDisease =
      Object.entries(diseaseCount).sort(([, a], [, b]) => b - a)[0]?.[0] ||
      null;

    return {
      totalEntries: entries.length,
      avgSeverity: totalSeverity / entries.length,
      maxSeverity,
      diseaseCounts: diseaseCount,
      lastUpdate: entries[0]?.timestamp,
      weeksTracked: uniqueWeeks.size,
      mostCommonDisease,
    };
  }, [entries]);

  const lastEntryDate =
    entries.length > 0 ? parseTs(entries[entries.length - 1].timestamp) : null;
  const daysSinceLast = lastEntryDate
    ? Math.floor((Date.now() - lastEntryDate.getTime()) / (1000 * 60 * 60 * 24))
    : null;
  const needsReminder = daysSinceLast !== null && daysSinceLast >= 7;

  const uniqueDiseases = Array.from(
    new Set(entries.map((e) => e.prediction)),
  ).filter(Boolean);
  const filteredEntries =
    filterDisease === "all"
      ? entries
      : entries.filter((e) => e.prediction === filterDisease);

  const severities = filteredEntries.map((e) =>
    Number(e.severity_percent || 0),
  );

  const Chart = ({ values }) => {
    if (!values || values.length === 0) return null;
    const w = 420,
      h = 120,
      pad = 24;
    const maxV = Math.max(100, ...values);
    const minV = Math.min(0, ...values);
    const xs = (i) =>
      pad + (i * (w - 2 * pad)) / Math.max(1, values.length - 1);
    const ys = (v) =>
      h - pad - ((v - minV) * (h - 2 * pad)) / Math.max(1, maxV - minV);
    const path = values
      .map((v, i) => `${i === 0 ? "M" : "L"} ${xs(i)} ${ys(v)}`)
      .join(" ");
    return (
      <svg
        width={w}
        height={h}
        style={{
          background: "#fff",
          borderRadius: 8,
          boxShadow: "0 2px 8px rgba(0,0,0,0.06)",
        }}
      >
        <polyline
          fill="none"
          stroke="#e0e0e0"
          strokeWidth="1"
          points={`${pad},${h - pad} ${w - pad},${h - pad}`}
        />
        <path d={path} stroke="#207820" strokeWidth="2.5" fill="none" />
        {values.map((v, i) => (
          <circle key={i} cx={xs(i)} cy={ys(v)} r={3.5} fill="#207820" />
        ))}
      </svg>
    );
  };

  return (
    <div style={{ padding: "30px", maxWidth: 1200, margin: "0 auto" }}>
      {/* Header section */}
      <div style={{ marginBottom: 24 }}>
        <h2>🕒 Crop Health Timeline</h2>
        <p>
          Upload a leaf image once per week. We estimate diseased area and track
          the trend.
        </p>
      </div>

      {/* Action buttons */}
      <div
        style={{
          display: "flex",
          gap: 12,
          justifyContent: "flex-end",
          marginBottom: 16,
        }}
      >
        <button
          onClick={handleClearTimeline}
          disabled={loading || entries.length === 0}
          style={{
            padding: "8px 16px",
            borderRadius: 8,
            border: "1px solid #b00020",
            color: entries.length === 0 ? "#666" : "#b00020",
            background: "white",
            cursor: entries.length === 0 ? "not-allowed" : "pointer",
            display: "flex",
            alignItems: "center",
            gap: 6,
            opacity: loading ? 0.7 : 1,
            transition: "all 0.2s ease",
            ":hover": {
              background: entries.length === 0 ? "white" : "#fff5f5",
            },
          }}
        >
          <span style={{ fontSize: 16 }}>{loading ? "⏳" : "🗑️"}</span>
          {loading ? "Clearing..." : "Clear Timeline"}
        </button>
        <button
          onClick={handleEndTimeline}
          style={{
            padding: "8px 16px",
            borderRadius: 8,
            border: "1px solid #207820",
            color: "#207820",
            background: "white",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            gap: 6,
          }}
        >
          <span style={{ fontSize: 16 }}>✔️</span>
          End Timeline
        </button>
      </div>

      {/* Upload controls */}
      <div
        style={{
          display: "flex",
          gap: 12,
          alignItems: "center",
          flexWrap: "wrap",
          padding: 16,
          background: "#f8f9fa",
          borderRadius: 12,
          marginBottom: 24,
        }}
      >
        <input
          type="text"
          value={farmerId}
          onChange={(e) => setFarmerId(e.target.value)}
          placeholder="farmer id"
          disabled={!!(auth && auth.user)}
          title={
            auth && auth.user ? "Using logged-in user id" : "Enter farmer id"
          }
          style={{ padding: 8, borderRadius: 8, border: "1px solid #ccc" }}
        />
        <input type="file" accept="image/*" onChange={onFileChange} />
        <input
          type="text"
          value={plantId}
          onChange={(e) => setPlantId(e.target.value)}
          placeholder="plant id"
          style={{ padding: 8, borderRadius: 8, border: "1px solid #ccc" }}
        />
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <select
            value={selectedCrop}
            onChange={(e) => setSelectedCrop(e.target.value)}
            style={{ padding: 8, borderRadius: 6, border: "1px solid #ccc" }}
          >
            {crops.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
          <input
            type="text"
            value={newCrop}
            onChange={(e) => setNewCrop(e.target.value)}
            placeholder="Add crop"
            style={{ padding: 8, borderRadius: 6, border: "1px solid #ccc" }}
          />
          <button
            onClick={async () => {
              const c = (newCrop || "").trim();
              if (!c) return;
              try {
                const res = await fetch(`${API_BASE}/timeline/crops`, {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ farmer_id: farmerId, name: c }),
                  credentials: "include",
                });
                const d = await res.json();
                if (!res.ok) throw new Error(d.error || "Failed to add crop");
                // insert into local list and pick it
                setCrops((prev) => (prev.includes(c) ? prev : [c, ...prev]));
                setSelectedCrop(c);
                setNewCrop("");
              } catch (err) {
                alert("Could not save crop: " + err.message);
              }
            }}
            style={{ padding: "8px 12px" }}
          >
            Add
          </button>
        </div>
        {/* disease selection removed - only crop is required */}
        <button
          onClick={onUpload}
          disabled={loading || !file}
          style={{
            backgroundColor: "#207820",
            color: "white",
            padding: "10px 16px",
            border: "none",
            borderRadius: 8,
            cursor: "pointer",
          }}
        >
          {loading ? "Uploading..." : "Upload weekly image"}
        </button>
        <button
          onClick={() => loadTimeline(farmerId)}
          style={{ padding: "8px 12px" }}
        >
          Refresh
        </button>
        <button
          onClick={() => navigate("/")}
          style={{
            marginLeft: "auto",
            backgroundColor: "#207820",
            color: "white",
            padding: "10px 16px",
            borderRadius: "8px",
            border: "none",
            cursor: "pointer",
          }}
        >
          ⬅ Back to Home
        </button>
      </div>

      {error && <div style={{ marginTop: 12, color: "#b00020" }}>{error}</div>}

      {preview && (
        <div style={{ marginTop: 16 }}>
          <strong>Preview:</strong>
          <div>
            <img
              src={preview}
              alt="preview"
              style={{
                width: 220,
                height: 220,
                objectFit: "cover",
                borderRadius: 12,
                border: "2px solid #c8f1c8",
              }}
            />
          </div>
        </div>
      )}

      {needsReminder && (
        <div
          style={{
            marginTop: 12,
            padding: 12,
            borderRadius: 8,
            background: "#fff8e1",
            border: "1px solid #ffe082",
          }}
        >
          <strong>Reminder:</strong> It has been {daysSinceLast} day(s) since
          your last upload. Please submit this week’s image.
        </div>
      )}

      {/* Summary statistics */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
          gap: 16,
          margin: "24px 0",
          padding: 16,
          background: "#f8f9fa",
          borderRadius: 12,
          boxShadow: "0 1px 2px rgba(0,0,0,0.05)",
        }}
      >
        <div
          style={{
            padding: 16,
            background: "#fff",
            borderRadius: 8,
            boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
          }}
        >
          <div style={{ fontSize: 14, color: "#666", marginBottom: 4 }}>
            Total Entries
          </div>
          <div style={{ fontSize: 24, fontWeight: 700 }}>
            {stats.totalEntries}
          </div>
          <div style={{ fontSize: 13, color: "#666", marginTop: 4 }}>
            Over {stats.weeksTracked} weeks
          </div>
        </div>

        <div
          style={{
            padding: 16,
            background: "#fff",
            borderRadius: 8,
            boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
          }}
        >
          <div style={{ fontSize: 14, color: "#666", marginBottom: 4 }}>
            Disease Status
          </div>
          <div style={{ fontSize: 16, fontWeight: 600 }}>
            {stats.mostCommonDisease || "No diseases detected"}
          </div>
          <div style={{ fontSize: 13, color: "#666", marginTop: 4 }}>
            Most frequent occurrence
          </div>
        </div>

        <div
          style={{
            padding: 16,
            background: "#fff",
            borderRadius: 8,
            boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
          }}
        >
          <div style={{ fontSize: 14, color: "#666", marginBottom: 4 }}>
            Severity Metrics
          </div>
          <div style={{ fontSize: 24, fontWeight: 700 }}>
            {stats.totalEntries > 0
              ? `${stats.avgSeverity.toFixed(1)}%`
              : "N/A"}
          </div>
          <div style={{ fontSize: 13, color: "#666", marginTop: 4 }}>
            Average severity
            {stats.maxSeverity > 0 &&
              ` (Max: ${stats.maxSeverity.toFixed(1)}%)`}
          </div>
        </div>

        <div
          style={{
            padding: 16,
            background: "#fff",
            borderRadius: 8,
            boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
          }}
        >
          <div style={{ fontSize: 14, color: "#666", marginBottom: 4 }}>
            Last Updated
          </div>
          <div style={{ fontSize: 16, fontWeight: 600 }}>
            {stats.lastUpdate
              ? new Date(
                  stats.lastUpdate.replace(" ", "T"),
                ).toLocaleDateString()
              : "Never"}
          </div>
          <div style={{ fontSize: 13, color: "#666", marginTop: 4 }}>
            {daysSinceLast ? `${daysSinceLast} days ago` : ""}
          </div>
        </div>
      </div>

      {/* Trend and filters */}
      <div
        style={{
          display: "flex",
          gap: 24,
          alignItems: "flex-start",
          margin: "24px 0",
          padding: 16,
          background: "#fff",
          borderRadius: 12,
          boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
        }}
      >
        <div>
          <div style={{ marginBottom: 8, color: "#666" }}>Current trend</div>
          <span
            style={{
              color: trendColor,
              fontWeight: 700,
              textTransform: "capitalize",
              fontSize: 18,
            }}
          >
            {trend}
          </span>
        </div>

        <div>
          <div style={{ marginBottom: 8, color: "#666" }}>
            Filter by disease
          </div>
          <select
            value={filterDisease}
            onChange={(e) => setFilterDisease(e.target.value)}
            style={{
              padding: 8,
              borderRadius: 6,
              border: "1px solid #ccc",
              minWidth: 150,
            }}
          >
            <option value="all">All diseases</option>
            {uniqueDiseases.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </div>

        {severities.length > 0 && (
          <div style={{ flex: 1 }}>
            <div style={{ marginBottom: 8, color: "#666" }}>
              Severity trend (%)
            </div>
            <Chart values={severities} />
          </div>
        )}
      </div>

      <div style={{ marginTop: 16 }}>
        <h3>Weekly Entries</h3>

        {filteredEntries.length === 0 ? (
          <div
            style={{
              padding: 24,
              textAlign: "center",
              color: "#666",
              background: "#fff",
              borderRadius: 12,
            }}
          >
            {entries.length === 0 ? (
              <div>
                <div style={{ fontSize: 18, marginBottom: 8, color: "#333" }}>
                  No timeline entries yet
                </div>
                <div>
                  Upload your first weekly image to start tracking crop health.
                </div>
              </div>
            ) : (
              <div>
                <div style={{ fontSize: 18, marginBottom: 8, color: "#333" }}>
                  No matching entries
                </div>
                <div>
                  No entries found with disease type "{filterDisease}".
                  {filterDisease !== "all" &&
                    "Try changing the disease filter."}
                </div>
              </div>
            )}
          </div>
        ) : (
          (() => {
            const groups = {};
            entries.forEach((e) => {
              const w = e.week_index != null ? String(e.week_index) : "0";
              if (!groups[w]) groups[w] = [];
              groups[w].push(e);
            });
            const weeks = Object.keys(groups).sort(
              (a, b) => Number(b) - Number(a),
            );
            const latestWeek = weeks.length > 0 ? weeks[0] : null;

            return (
              <div
                style={{ display: "flex", flexDirection: "column", gap: 16 }}
              >
                {weeks.map((w) => {
                  const weekEntries =
                    filterDisease === "all"
                      ? groups[w]
                      : groups[w].filter((e) => e.prediction === filterDisease);

                  if (weekEntries.length === 0) return null;

                  return (
                    <section
                      key={w}
                      style={{
                        background: w === latestWeek ? "#f7fff7" : "#fff",
                        border:
                          w === latestWeek
                            ? "2px solid #207820"
                            : "1px solid #eee",
                        borderRadius: 12,
                        padding: 16,
                        boxShadow:
                          w === latestWeek
                            ? "0 4px 14px rgba(32,120,32,0.08)"
                            : "0 2px 8px rgba(0,0,0,0.04)",
                      }}
                    >
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                          marginBottom: 16,
                        }}
                      >
                        <div>
                          <div
                            style={{
                              fontWeight: 800,
                              fontSize: 18,
                              color: w === latestWeek ? "#207820" : "#333",
                            }}
                          >
                            Week {w}
                          </div>
                          <div
                            style={{
                              fontSize: 14,
                              color: "#666",
                              marginTop: 4,
                            }}
                          >
                            {weekEntries[0]?.timestamp}
                          </div>
                        </div>
                        <div style={{ textAlign: "right" }}>
                          <div style={{ fontSize: 14, color: "#666" }}>
                            {weekEntries.length} upload
                            {weekEntries.length !== 1 ? "s" : ""}
                          </div>
                          {w === latestWeek && (
                            <div
                              style={{
                                fontSize: 13,
                                fontWeight: 700,
                                color: "#207820",
                                marginTop: 4,
                              }}
                            >
                              Most recent
                            </div>
                          )}
                        </div>
                      </div>

                      <div
                        style={{
                          display: "grid",
                          gridTemplateColumns:
                            "repeat(auto-fit, minmax(280px, 1fr))",
                          gap: 16,
                        }}
                      >
                        {weekEntries.map((e, idx) => (
                          <div
                            key={idx}
                            style={{
                              background: "#fff",
                              padding: 16,
                              borderRadius: 10,
                              border: "1px solid #eee",
                            }}
                          >
                            <div
                              style={{ cursor: "pointer", marginBottom: 12, position: "relative" }}
                              onClick={() =>
                                openImage(
                                  e.heatmap
                                    ? `${API_BASE}/${e.heatmap}`
                                    : e.segmentation
                                      ? `${API_BASE}/${e.segmentation}`
                                      : "",
                                  e.prediction,
                                )
                              }
                            >
                              {e.heatmap ? (
                                <img
                                  src={`${API_BASE}/${e.heatmap}`}
                                  alt="heatmap"
                                  style={{
                                    width: "100%",
                                    height: 200,
                                    objectFit: "cover",
                                    borderRadius: 8,
                                  }}
                                />
                              ) : e.segmentation ? (
                                <img
                                  src={`${API_BASE}/${e.segmentation}`}
                                  alt="segmentation"
                                  style={{
                                    width: "100%",
                                    height: 200,
                                    objectFit: "cover",
                                    borderRadius: 8,
                                  }}
                                />
                              ) : (
                                <div
                                  style={{
                                    width: "100%",
                                    height: 200,
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "center",
                                    background: "#fafafa",
                                    borderRadius: 8,
                                    color: "#666",
                                  }}
                                >
                                  No preview available
                                </div>
                              )}

                              {/* Replace overlay badges with labeled boxes below the image */}
                            </div>

                            <div style={{ display: "flex", justifyContent: "space-between", marginTop: 8 }}>
                              <div>
                                {e.heatmap ? (
                                  <button
                                    onClick={() => downloadImage(`${API_BASE}/${e.heatmap}`, `heatmap_${e.filename || 'image'}.jpg`)}
                                    style={{ padding: "6px 10px", borderRadius: 6, border: "1px solid #ddd", background: "black" }}
                                  >
                                    Download
                                  </button>
                                ) : (
                                  <div style={{ color: "#999", fontSize: 13 }}>No download</div>
                                )}
                              </div>

                              <div>
                                {e.segmentation ? (
                                  <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                                    <div style={{ padding: "6px 10px", borderRadius: 6, background: "#f5f5f5", border: "1px solid #eee", fontSize: 13 }}>
                                      Segmentation
                                    </div>
                                    <button
                                      onClick={() => openImage(`${API_BASE}/${e.segmentation}`, "Segmentation")}
                                      style={{ padding: "6px 10px", borderRadius: 6, border: "1px solid #ddd", background: "black" }}
                                    >
                                      View
                                    </button>
                                  </div>
                                ) : (
                                  <div style={{ color: "#999", fontSize: 13 }}>No segmentation</div>
                                )}
                              </div>
                            </div>

                            <div style={{ marginBottom: 12 }}>
                              <div
                                style={{
                                  fontWeight: 700,
                                  fontSize: 16,
                                  marginBottom: 4,
                                }}
                              >
                                {e.prediction}
                              </div>
                              <div style={{ fontSize: 14, color: "#666" }}>
                                {e.timestamp}
                              </div>
                              <div style={{ fontSize: 13, color: "#444", marginTop: 6 }}>
                                <strong>Crop:</strong> {e.crop || selectedCrop}
                                <br />
                                  <strong>Farmer label:</strong> {e.disease || "Unknown"}
                                  {e._id && (
                                    <button
                                      onClick={async (ev) => {
                                        ev.stopPropagation();
                                        const newVal = window.prompt("Edit disease label:", e.disease || "");
                                        if (newVal === null) return;
                                        try {
                                          const res = await fetch(`${API_BASE}/timeline/entry/${e._id}/disease`, {
                                            method: "PUT",
                                            headers: { "Content-Type": "application/json" },
                                            body: JSON.stringify({ disease: newVal }),
                                            credentials: "include",
                                          });
                                          const data = await res.json();
                                          if (!res.ok) throw new Error(data.error || "Failed to update disease");
                                          // refresh timeline
                                          await loadTimeline(farmerId);
                                        } catch (err) {
                                          alert("Failed to update: " + err.message);
                                        }
                                      }}
                                      style={{ marginLeft: 8, padding: "4px 8px", fontSize: 12 }}
                                    >
                                      Edit
                                    </button>
                                  )}
                              </div>
                            </div>

                            {/* Health metrics */}
                            {e.health_metrics && (
                              <div style={{ marginBottom: 12, color: "#444" }}>
                                <div>
                                  <strong>Leaf area:</strong>{" "}
                                  {e.health_metrics.leaf_area_percent != null
                                    ? `${Number(e.health_metrics.leaf_area_percent).toFixed(1)}%`
                                    : "N/A"}
                                </div>
                                <div>
                                  <strong>Diseased (of leaf):</strong>{" "}
                                  {e.health_metrics.diseased_percent_of_leaf != null
                                    ? `${Number(e.health_metrics.diseased_percent_of_leaf).toFixed(1)}%`
                                    : "N/A"}
                                </div>
                                <div>
                                  <strong>Mean saturation:</strong>{" "}
                                  {e.health_metrics.mean_saturation != null
                                    ? `${Number(e.health_metrics.mean_saturation).toFixed(1)}`
                                    : "N/A"}
                                </div>
                              </div>
                            )}

                            <div
                              style={{
                                display: "flex",
                                justifyContent: "space-between",
                                alignItems: "center",
                              }}
                            >
                              <div>
                                <strong>Severity:</strong>{" "}
                                {e.severity_percent?.toFixed
                                  ? e.severity_percent.toFixed(2)
                                  : e.severity_percent}
                                %
                              </div>

                              <div style={{ display: "flex", gap: 8 }}>
                                {e.heatmap && (
                                  <button
                                    onClick={() =>
                                      openImage(
                                        `${API_BASE}/${e.heatmap}`,
                                        "Heatmap",
                                      )
                                    }
                                    style={{
                                      padding: "6px 12px",
                                      borderRadius: 6,
                                      border: "1px solid #ddd",
                                      background: "black",
                                      cursor: "pointer",
                                    }}
                                  >
                                    View heatmap
                                  </button>
                                )}
                                {e.segmentation && (
                                  <button
                                    onClick={() =>
                                      openImage(
                                        `${API_BASE}/${e.segmentation}`,
                                        "Segmentation",
                                      )
                                    }
                                    style={{
                                      padding: "6px 12px",
                                      borderRadius: 6,
                                      border: "1px solid #ddd",
                                      background: "black",
                                      cursor: "pointer",
                                    }}
                                  >
                                    View segmentation
                                  </button>
                                )}
                                {e.comparison && (
                                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                    <div style={{ fontSize: 12, color: "#666" }}>
                                      <div>
                                        Severity: <strong>{e.comparison.severity_trend}</strong>
                                      </div>
                                      <div>
                                        Leaf area: <strong>{e.comparison.leaf_area_percent_trend}</strong>
                                      </div>
                                    </div>
                                  </div>
                                )}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </section>
                  );
                })}
              </div>
            );
          })()
        )}
      </div>

      {/* Modal / Lightbox for full-size preview */}
      {modalState.open && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.75)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 9999,
            backdropFilter: "blur(4px)",
          }}
          onClick={closeModal}
        >
          <div
            style={{
              maxWidth: "90%",
              maxHeight: "90%",
              position: "relative",
              background: "#fff",
              padding: 16,
              borderRadius: 12,
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <img
              src={modalState.src}
              alt={modalState.title}
              style={{
                maxWidth: "100%",
                maxHeight: "calc(90vh - 100px)",
                height: "auto",
                borderRadius: 8,
                display: "block",
              }}
            />
            <div
              style={{
                position: "absolute",
                right: 16,
                top: 16,
                display: "flex",
                gap: 8,
              }}
            >
              <button
                onClick={() =>
                  downloadImage(
                    modalState.src,
                    `${modalState.title || "image"}.jpg`,
                  )
                }
                style={{
                  padding: "8px 16px",
                  borderRadius: 6,
                  border: "none",
                  background: "#207820",
                  color: "#fff",
                  cursor: "pointer",
                }}
              >
                Download
              </button>
              <button
                onClick={closeModal}
                style={{
                  padding: "8px 16px",
                  borderRadius: 6,
                  border: "1px solid #ddd",
                  background: "black",
                  cursor: "pointer",
                }}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
