import React, { useState } from 'react';
import { 
  Activity, 
  Users, 
  UploadCloud, 
  CalendarCheck, 
  BarChart3, 
  BookOpenCheck, 
  LogOut, 
  Menu, 
  X,
  UserCheck
} from 'lucide-react';
import { useAuth } from '../../contexts/AuthContext';

export default function AppLayout({ activeTab, onTabChange, children }) {
  const { currentUser, logout } = useAuth();
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const navigateTo = (tab) => {
    onTabChange(tab);
    setIsSidebarOpen(false);
  };

  return (
    <div className="app-container" style={{ display: 'flex', minHeight: '100vh', backgroundColor: 'var(--background)' }}>
      {/* Mobile Sidebar Overlay */}
      {isSidebarOpen && (
        <div 
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.4)',
            zIndex: 90,
            backdropFilter: 'blur(2px)'
          }}
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* Main Sidebar */}
      <aside 
        className={`app-sidebar ${isSidebarOpen ? 'open' : ''}`}
        style={{
          width: '260px',
          backgroundColor: 'var(--surface)',
          borderRight: '1px solid var(--border-light)',
          display: 'flex',
          flexDirection: 'column',
          zIndex: 95
        }}
      >
        <div style={{ padding: '24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid var(--border-light)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{ width: '36px', height: '36px', borderRadius: '10px', backgroundColor: 'var(--primary)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff' }}>
              <Activity size={20} />
            </div>
            <div>
              <h2 style={{ fontSize: '16px', fontWeight: 'bold', color: 'var(--text-primary)', margin: 0, lineHeight: 1.2 }}>DR-Screening</h2>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Chẩn Đoán Võng Mạc AI</span>
            </div>
          </div>
          <button 
            className="mobile-close-btn"
            style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', display: 'none' }}
            onClick={() => setIsSidebarOpen(false)}
          >
            <X size={20} />
          </button>
        </div>

        {/* Navigation Items */}
        <nav style={{ padding: '16px 12px', display: 'flex', flexDirection: 'column', gap: '6px', flex: 1 }}>
          <button 
            className={`nav-item ${activeTab === 'dashboard' ? 'active' : ''}`}
            style={{ 
              justifyContent: 'flex-start', 
              gap: '12px',
              backgroundColor: activeTab === 'dashboard' ? 'rgba(13, 148, 136, 0.1)' : 'transparent',
              color: activeTab === 'dashboard' ? 'var(--primary)' : 'var(--text-secondary)',
              border: 'none',
              padding: '10px 14px',
              borderRadius: '8px',
              fontSize: '14px',
              fontWeight: activeTab === 'dashboard' ? '600' : '500',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              width: '100%',
              transition: 'all 0.15s'
            }}
            onClick={() => navigateTo('dashboard')}
          >
            <BarChart3 size={18} /> Tổng Quan & Nghiên Cứu
          </button>
          
          <button 
            className={`nav-item ${activeTab === 'screening' ? 'active' : ''}`}
            style={{ 
              justifyContent: 'flex-start', 
              gap: '12px',
              backgroundColor: activeTab === 'screening' ? 'rgba(13, 148, 136, 0.1)' : 'transparent',
              color: activeTab === 'screening' ? 'var(--primary)' : 'var(--text-secondary)',
              border: 'none',
              padding: '10px 14px',
              borderRadius: '8px',
              fontSize: '14px',
              fontWeight: activeTab === 'screening' ? '600' : '500',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              width: '100%',
              transition: 'all 0.15s'
            }}
            onClick={() => navigateTo('screening')}
          >
            <UploadCloud size={18} /> Phòng Khám Sàng Lọc AI
          </button>

          <button 
            className={`nav-item ${activeTab === 'patients' ? 'active' : ''}`}
            style={{ 
              justifyContent: 'flex-start', 
              gap: '12px',
              backgroundColor: activeTab === 'patients' ? 'rgba(13, 148, 136, 0.1)' : 'transparent',
              color: activeTab === 'patients' ? 'var(--primary)' : 'var(--text-secondary)',
              border: 'none',
              padding: '10px 14px',
              borderRadius: '8px',
              fontSize: '14px',
              fontWeight: activeTab === 'patients' ? '600' : '500',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              width: '100%',
              transition: 'all 0.15s'
            }}
            onClick={() => navigateTo('patients')}
          >
            <Users size={18} /> Hồ Sơ Bệnh Nhân
          </button>

          <button 
            className={`nav-item ${activeTab === 'recalls' ? 'active' : ''}`}
            style={{ 
              justifyContent: 'flex-start', 
              gap: '12px',
              backgroundColor: activeTab === 'recalls' ? 'rgba(13, 148, 136, 0.1)' : 'transparent',
              color: activeTab === 'recalls' ? 'var(--primary)' : 'var(--text-secondary)',
              border: 'none',
              padding: '10px 14px',
              borderRadius: '8px',
              fontSize: '14px',
              fontWeight: activeTab === 'recalls' ? '600' : '500',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              width: '100%',
              transition: 'all 0.15s'
            }}
            onClick={() => navigateTo('recalls')}
          >
            <CalendarCheck size={18} /> Lịch Tái Khám
          </button>

          <button
            className={`nav-item ${activeTab === 'evidence' ? 'active' : ''}`}
            style={{
              justifyContent: 'flex-start',
              gap: '12px',
              backgroundColor: activeTab === 'evidence' ? 'rgba(13, 148, 136, 0.1)' : 'transparent',
              color: activeTab === 'evidence' ? 'var(--primary)' : 'var(--text-secondary)',
              border: 'none',
              padding: '10px 14px',
              borderRadius: '8px',
              fontSize: '14px',
              fontWeight: activeTab === 'evidence' ? '600' : '500',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              width: '100%',
              transition: 'all 0.15s'
            }}
            onClick={() => navigateTo('evidence')}
          >
            <BookOpenCheck size={18} /> Pháp Lý & Khoa Học
          </button>
        </nav>

        {/* User Profile & Logout */}
        <div style={{ padding: '16px 20px', borderTop: '1px solid var(--border-light)', display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {currentUser && (
            <div style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
              Đang đăng nhập:<br />
              <strong style={{ color: 'var(--text-primary)', fontSize: '14px' }}>{currentUser.full_name}</strong>
              <span style={{ display: 'block', fontSize: '11px', color: 'var(--primary)', fontWeight: 'bold', marginTop: '2px' }}>
                {currentUser.role === 'admin' ? 'Quản Trị Viên Hệ Thống' : 'Bác sĩ Nhãn Khoa'}
              </span>
            </div>
          )}
          <button 
            className="btn btn-secondary" 
            style={{ width: '100%', justifyContent: 'center', gap: '8px', fontSize: '13px' }}
            onClick={logout}
          >
            <LogOut size={15} /> Đăng xuất
          </button>
        </div>
      </aside>

      {/* Main Content Area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0, overflowX: 'hidden' }}>
        {/* Mobile Header Bar */}
        <header className="mobile-header" style={{ padding: '14px 20px', backgroundColor: 'var(--surface)', borderBottom: '1px solid var(--border-light)', display: 'none', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <button 
              style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-primary)', padding: 0, display: 'flex' }}
              onClick={() => setIsSidebarOpen(true)}
            >
              <Menu size={22} />
            </button>
            <strong style={{ fontSize: '16px', color: 'var(--primary)' }}>DR-Screening</strong>
          </div>
          <button 
            className="btn btn-secondary" 
            style={{ padding: '6px 10px', fontSize: '12px', gap: '6px' }}
            onClick={logout}
          >
            <LogOut size={13} />
          </button>
        </header>

        {/* Page Body */}
        <main className="app-main-content" style={{ flex: 1, padding: '30px 40px', maxWidth: '1400px', width: '100%', margin: '0 auto', boxSizing: 'border-box' }}>
          {children}
        </main>
      </div>
    </div>
  );
}
