// import React from "react";
// import Sidebar from "./components/Sidebar";
// import Home from "./components/Home";

// function App() {
//   return (
//     <div className="app-container">
//       <Sidebar />
//       <main className="main-content">
//         <Home />
//       </main>
//      </div>
//  );
//  }

//  export default App;

// import React from "react";
// import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
// import Sidebar from "./components/Sidebar";
// import DiseaseDetection from "./components/DiseaseDetection"; // your current page
// import Timeline from "./components/Timeline";
// import History from "./components/History";
// import "./App.css";

// function App() {
//   return (
//     <Router>
//       <div className="app-container">
//         <Sidebar />
//         <main className="main-content">
//           <Routes>
//             <Route path="/" element={<DiseaseDetection />} />
//             <Route path="/timeline" element={<Timeline />} />
//             <Route path="/history" element={<History />} />
//           </Routes>
//         </main>
//       </div>
//     </Router>
//   );
// }

// export default App;

// src/App.jsx
// import React from "react";
// import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

// import Home from "./Home";
// import DiseaseDetection from "./DiseaseDetection";
// import DiseaseShowing from "./DiseaseShowing";
// import ImageGeneration from "./ImageGeneration";
// import Timeline from "./Timeline";
// import "./App.css";


import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { AuthProvider, RequireAuth } from "./context/AuthContext";

import Layout from "./Layout";
import "./App.css";
import Home from "./components/Home";
import DiseaseDetection from "./components/DiseaseDetection";
import DiseaseVisualization from "./components/DiseaseVisualization";
import ImageGeneration from "./components/ImageGeneration";
import Timeline from "./components/Timeline";
import History from "./components/History"; // 👈 Add this if you made it
import Login from "./components/Login";
import Signup from "./components/Signup";

function App() {
  return (
    <AuthProvider>
      <Router>
        <Routes>
          {/* Auth pages (no Layout) */}
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          {/* All main pages inside Layout (with sidebar) - protected */}
          <Route
            path="/"
            element={
              <RequireAuth>
                <Layout />
              </RequireAuth>
            }
          >
            <Route index element={<Home />} />
            <Route path="detect" element={<DiseaseDetection />} />
            <Route path="showing" element={<DiseaseVisualization />} />
            <Route path="image" element={<ImageGeneration />} />
            <Route path="timeline" element={<Timeline />} />
            <Route path="history" element={<History />} />
          </Route>
        </Routes>
      </Router>
    </AuthProvider>
  );
}

export default App;
