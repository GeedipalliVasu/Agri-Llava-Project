import React from "react";
import { Outlet } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import Header from "./components/Header";
import "./Layout.css";

const Layout = () => {
  return (
    <div className="layout-container">
      <Sidebar />
      <div className="main-content-wrapper">
        <Header />
        <main className="main-content">
          <Outlet /> {/* This renders the current route's content */}
        </main>
      </div>
    </div>
  );
};

export default Layout;
