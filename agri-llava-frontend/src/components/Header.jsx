import React from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import "./Header.css";

const Header = () => {
  const auth = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    if (auth && auth.logout) auth.logout();
    navigate("/login");
  };

  return (
    <header className="header">
      <div className="auth-buttons">
        {!auth || !auth.user ? (
          <>
            <NavLink to="/login" className="auth-btn">
              Login
            </NavLink>
            <NavLink to="/signup" className="auth-btn">
              Signup
            </NavLink>
          </>
        ) : (
          <>
            <span className="user-info">Welcome, {auth.user?.name}</span>
            <button onClick={handleLogout} className="logout-btn">
              Logout
            </button>
          </>
        )}
      </div>
    </header>
  );
};

export default Header;
