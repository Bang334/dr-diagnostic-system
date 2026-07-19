import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import { AppDialogProvider } from './components/AppDialogProvider.jsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <AppDialogProvider>
      <App />
    </AppDialogProvider>
  </React.StrictMode>,
)
