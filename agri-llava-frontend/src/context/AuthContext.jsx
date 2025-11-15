import React, { createContext, useContext, useEffect, useState } from 'react';
import { Navigate, useLocation } from 'react-router-dom';

const AuthContext = createContext(null);

// Use relative URL by default so requests go to the same origin the page was loaded from
const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  const refreshUser = async () => {
    try {
      const res = await fetch(`${API_BASE}/auth/me`, { credentials: 'include' });
      const data = await res.json();
      setUser(data.user || null);
    } catch (e) {
      setUser(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refreshUser();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const login = (userObj) => setUser(userObj);
  const logout = () => setUser(null);

  return (
    <AuthContext.Provider value={{ user, login, logout, refreshUser, loading }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}

// Route guard component for react-router
export function RequireAuth({ children }) {
  const auth = useAuth();
  const location = useLocation();
  if (!auth || auth.loading) return null; // or a loader
  if (!auth.user) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }
  return children ?? null;
}

export default AuthContext;
