import React, { useState } from 'react';
import { Activity, User, Key, AlertTriangle } from 'lucide-react';
import { useAuth } from '../../contexts/AuthContext';

export default function LoginPage() {
  const { login, authError } = useAuth();
  const [username, setUsername] = useState('dr.nguyen');
  const [password, setPassword] = useState('doctor123');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    await login(username, password);
    setIsSubmitting(false);
  };

  return (
    <div className="auth-page" style={{
      display: 'flex', 
      height: '100vh', 
      width: '100vw', 
      alignItems: 'center', 
      justifyContent: 'center', 
      backgroundColor: 'var(--background)',
      padding: '20px'
    }}>
      <div className="card auth-card animated-fade-in" style={{
        width: '100%', 
        maxWidth: '420px', 
        padding: '40px 32px',
        display: 'flex',
        flexDirection: 'column',
        gap: '24px',
        boxShadow: 'var(--shadow-lg)',
        borderRadius: 'var(--radius-lg)'
      }}>
        {/* Logo & Header */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}>
          <div className="brand-mark brand-mark-large">
            <Activity size={32} color="var(--primary)" />
          </div>
          <h2 style={{ fontSize: '24px', fontWeight: 'bold', fontFamily: 'var(--font-family-display)', margin: 0 }}>DR-Screening App</h2>
          <p style={{ fontSize: '13px', color: 'var(--text-muted)', textAlign: 'center', margin: 0 }}>Hệ thống sàng lọc và quản lý bệnh võng mạc tiểu đường thông minh</p>
        </div>

        {authError && (
          <div className="alert alert-error" role="alert">
            <AlertTriangle size={16} style={{ flexShrink: 0, marginTop: '2px' }} />
            <span>{authError}</span>
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <label htmlFor="login-username" style={{ fontSize: '13px', fontWeight: '500', color: 'var(--text-secondary)' }}>Tên đăng nhập</label>
            <div className="input-shell">
              <User size={16} color="var(--text-muted)" style={{ marginRight: '8px' }} />
              <input 
                type="text" 
                id="login-username"
                autoComplete="username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Tên tài khoản..." 
                required
                style={{ border: 'none', background: 'transparent', outline: 'none', width: '100%', padding: '12px 0', color: 'var(--text-primary)', fontSize: '14px' }}
              />
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <label htmlFor="login-password" style={{ fontSize: '13px', fontWeight: '500', color: 'var(--text-secondary)' }}>Mật khẩu</label>
            <div className="input-shell">
              <Key size={16} color="var(--text-muted)" style={{ marginRight: '8px' }} />
              <input 
                type="password" 
                id="login-password"
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••" 
                required
                style={{ border: 'none', background: 'transparent', outline: 'none', width: '100%', padding: '12px 0', color: 'var(--text-primary)', fontSize: '14px' }}
              />
            </div>
          </div>

          <button 
            type="submit" 
            className="btn btn-primary" 
            disabled={isSubmitting}
            style={{ width: '100%', padding: '12px', marginTop: '8px', fontSize: '15px', fontWeight: 'bold' }}
          >
            {isSubmitting ? 'Đang đăng nhập...' : 'Đăng nhập'}
          </button>
        </form>

        {/* Quick Demo Credentials */}
        <div style={{ borderTop: '1px solid var(--border-light)', paddingTop: '16px', fontSize: '12px', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: '6px' }}>
          <span><strong>Tài khoản Demo gợi ý:</strong></span>
          <div className="demo-credentials" style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <span>👨‍⚕️ Bác sĩ: <code style={{ cursor: 'pointer', background: 'rgba(0,0,0,0.05)', padding: '2px 4px', borderRadius: '4px' }} onClick={() => { setUsername('dr.nguyen'); setPassword('doctor123'); }}>dr.nguyen / doctor123</code></span>
            <span>⚙️ Admin: <code style={{ cursor: 'pointer', background: 'rgba(0,0,0,0.05)', padding: '2px 4px', borderRadius: '4px' }} onClick={() => { setUsername('admin'); setPassword('admin123'); }}>admin / admin123</code></span>
            <span>🏥 Bệnh nhân: <code style={{ cursor: 'pointer', background: 'rgba(0,0,0,0.05)', padding: '2px 4px', borderRadius: '4px' }} onClick={() => { setUsername('BN0001'); setPassword('benhnhan'); }}>BN0001 / benhnhan</code></span>
          </div>
        </div>
      </div>
    </div>
  );
}
