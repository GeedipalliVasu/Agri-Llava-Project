import React, { useEffect, useState } from "react";
import "./History.css";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:5000";

const HistoryPage = () => {
  const [historyItems, setHistoryItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [clearing, setClearing] = useState(false);

  useEffect(() => {
    const fetchHistory = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${API_BASE}/history`, { credentials: 'include' });
        if (!res.ok) throw new Error('Could not fetch history from server');
        const data = await res.json();
        // If server returns an array, use it; otherwise fall back to localStorage
        if (Array.isArray(data)) {
          setHistoryItems(data);
        } else if (Array.isArray(data.records)) {
          setHistoryItems(data.records);
        } else {
          // fallback to localStorage
          const local = JSON.parse(localStorage.getItem("predictionHistory")) || [];
          setHistoryItems(local);
        }
      } catch (err) {
        // network/backend error — fall back to localStorage
        const local = JSON.parse(localStorage.getItem("predictionHistory")) || [];
        setHistoryItems(local);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
  }, []);

  const handleClearHistory = async () => {
    if (!confirm("Are you sure you want to clear all history? This action cannot be undone.")) {
      return;
    }
    
    setClearing(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/history/clear`, {
        method: 'POST',
        credentials: 'include'
      });
      
      if (!res.ok) throw new Error('Failed to clear history');
      
      // Clear local state
      setHistoryItems([]);
      // Clear localStorage as well
      localStorage.removeItem("predictionHistory");
      alert("History cleared successfully");
    } catch (err) {
      setError(err.message);
      alert("Failed to clear history: " + err.message);
    } finally {
      setClearing(false);
    }
  };

  return (
    <div className="history-container">
      <div className="history-header">
        <h2>📜 Prediction History</h2>
        {historyItems.length > 0 && (
          <button 
            className="clear-history-btn" 
            onClick={handleClearHistory}
            disabled={clearing || loading}
          >
            {clearing ? "Clearing..." : "Clear History"}
          </button>
        )}
      </div>

      {loading ? (
        <p>Loading…</p>
      ) : historyItems.length === 0 ? (
        <p className="empty-history">No history available yet.</p>
      ) : (
        <div className="history-grid">
          {historyItems.map((entry, index) => (
            <div key={index} className="history-card">
              {entry.heatmap ? (
                <img src={`${API_BASE}/${entry.heatmap}`.replace(/\\/g, "/")} alt="Leaf" className="history-img" />
              ) : entry.image ? (
                <img src={entry.image} alt="Leaf" className="history-img" />
              ) : null}
              <div className="history-details">
                <h4>Result: <span>{entry.prediction}</span></h4>
                <p className="time">🕒 {entry.timestamp || entry.time || entry.created_at}</p>
                {entry.confidence_percent && <p>Confidence: {entry.confidence_percent}</p>}
              </div>
            </div>
          ))}
        </div>
      )}

      {error && <p className="history-error">Note: {error} (showing local history)</p>}
    </div>
  );
};

export default HistoryPage;
