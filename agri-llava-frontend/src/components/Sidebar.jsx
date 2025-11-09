import React from "react";
import { NavLink, useNavigate } from "react-router-dom";
import "./Sidebar.css";
import { useAuth } from "../context/AuthContext";

const Sidebar = () => {
  const auth = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    if (auth && auth.logout) auth.logout();
    navigate("/login");
  };
  return (
    <div className="sidebar">
      <h2 className="sidebar-title">🌿 Agri-LLaVA</h2>
      <nav>
        <ul>
          <li>
            <NavLink
              to="/"
              className={({ isActive }) => (isActive ? "active" : "")}
            >
              🏠 Home
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/detect"
              className={({ isActive }) => (isActive ? "active" : "")}
            >
              🧬 Disease Detection
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/showing"
              className={({ isActive }) => (isActive ? "active" : "")}
            >
              🌿 Disease Visualization
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/image"
              className={({ isActive }) => (isActive ? "active" : "")}
            >
              🖼️ Image Generation
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/timeline"
              className={({ isActive }) => (isActive ? "active" : "")}
            >
              🕒 Timeline
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/history"
              className={({ isActive }) => (isActive ? "active" : "")}
            >
              📜 History
            </NavLink>
          </li>
          {auth && auth.token && (
            <li>
              <button onClick={handleLogout} className="sidebar-logout">
                🚪 Logout
              </button>
            </li>
          )}
        </ul>
      </nav>
    </div>
  );
};

export default Sidebar;
