import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';
import { AppDialogProvider } from './components/AppDialogProvider.jsx';
import { AuthProvider } from './contexts/AuthContext.jsx';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <AppDialogProvider>
      <AuthProvider>
        <App />
      </AuthProvider>
    </AppDialogProvider>
  </React.StrictMode>,
);
