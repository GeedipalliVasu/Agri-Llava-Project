import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import "./Auth.css";
import { useAuth } from "../context/AuthContext";

function Signup() {
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const auth = useAuth();
  const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:5000";

  const validate = () => {
    if (!name) return "Name required";
    if (!email) return "Email required";
    if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) return "Invalid email";
    if (!password) return "Password required";
    if (password.length < 6) return "Password must be at least 6 characters";
    if (password !== confirm) return "Passwords do not match";
    return null;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    const v = validate();
    if (v) {
      setError(v);
      return;
    }
    setLoading(true);
    try {
      // Call backend signup (note backend requires name)
      const res = await fetch(`${API_BASE}/auth/signup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, email, password }),
        credentials: 'include'
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Signup failed');
      // Refresh auth context from server (cookie-based session)
      await auth.refreshUser();
      navigate("/");
    } catch (err) {
      setError(err.message || "Signup failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-card">
        <h2>Create account</h2>
        {error && <div className="auth-error">{error}</div>}
        <form onSubmit={handleSubmit}>
          <div className="auth-field">
            <label>Full name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Your full name"
            />
          </div>
          <div className="auth-field">
            <label>Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
            />
          </div>

          <div className="auth-field">
            <label>Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="At least 6 characters"
            />
          </div>

          <div className="auth-field">
            <label>Confirm password</label>
            <input
              type="password"
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
              placeholder="Repeat password"
            />
          </div>

          <div className="auth-actions">
            <button type="submit" className="btn-primary" disabled={loading}>
              {loading ? "Creating..." : "Create account"}
            </button>
            <Link to="/login" className="btn-link">
              Already have an account?
            </Link>
          </div>
        </form>
      </div>
    </div>
  );
}

export default Signup;
