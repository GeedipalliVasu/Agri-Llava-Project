import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import "./Auth.css";
import { useAuth } from "../context/AuthContext";

function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const auth = useAuth();
  const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:5000";

  const validate = () => {
    if (!email) return "Email required";
    if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) return "Invalid email";
    if (!password) return "Password required";
    if (password.length < 6) return "Password must be at least 6 characters";
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
      const res = await fetch(`${API_BASE}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
        credentials: 'include'
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Login failed');
      // Refresh auth context from server (cookie-based session)
      await auth.refreshUser();
      navigate("/");
    } catch (err) {
      setError(err.message || "Login failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-card">
        <h2>Log in</h2>
        {error && <div className="auth-error">{error}</div>}
        <form onSubmit={handleSubmit}>
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
              placeholder="••••••••"
            />
          </div>

          <div className="auth-actions">
            <button type="submit" className="btn-primary" disabled={loading}>
              {loading ? "Signing in..." : "Sign in"}
            </button>
            <Link to="/signup" className="btn-link">
              Create account
            </Link>
          </div>
        </form>
      </div>
    </div>
  );
}

export default Login;
