import React, { createContext, useContext, useState, useEffect } from 'react';
import { api } from '../services/api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [token, setToken] = useState(localStorage.getItem('token') || '');
  const [currentUser, setCurrentUser] = useState(null);
  const [authError, setAuthError] = useState('');
  const [isAuthLoading, setIsAuthLoading] = useState(true);

  const fetchCurrentUser = async () => {
    try {
      const user = await api.getCurrentUser();
      setCurrentUser(user);
      return user;
    } catch (err) {
      console.error('Invalid token or session expired', err);
      logout();
      return null;
    }
  };

  useEffect(() => {
    if (token) {
      fetchCurrentUser().finally(() => setIsAuthLoading(false));
    } else {
      setIsAuthLoading(false);
    }
  }, [token]);

  const login = async (username, password) => {
    setAuthError('');
    try {
      const res = await api.login(username, password);
      localStorage.setItem('token', res.access_token);
      setToken(res.access_token);
      const user = await api.getCurrentUser();
      setCurrentUser(user);
      return { success: true, user };
    } catch (err) {
      const msg = err?.detail || 'Tên đăng nhập hoặc mật khẩu không chính xác.';
      setAuthError(msg);
      return { success: false, error: msg };
    }
  };

  const logout = () => {
    localStorage.removeItem('token');
    setToken('');
    setCurrentUser(null);
    setAuthError('');
  };

  return (
    <AuthContext.Provider value={{
      token,
      currentUser,
      isAuthLoading,
      authError,
      setAuthError,
      login,
      logout,
      refreshUser: fetchCurrentUser,
      isAuthenticated: Boolean(token && currentUser),
      isDoctor: currentUser?.role === 'doctor' || currentUser?.role === 'admin',
      isAdmin: currentUser?.role === 'admin',
      isPatient: currentUser?.role === 'patient',
    }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
