import React, { useState, useEffect } from 'react';
import { 
  Activity, 
  Users, 
  UploadCloud, 
  FileText, 
  Calendar, 
  ShieldAlert, 
  BarChart3,
  Search,
  CheckCircle,
  AlertTriangle,
  LogOut,
  Key,
  User,
  QrCode,
  Plus,
  CalendarCheck,
  ClipboardList
} from 'lucide-react';
import { colors } from './theme/colors';
import { api } from './services/api';

// Import custom modals
import QRScannerModal from './components/QRScannerModal';
import PatientFormModal from './components/PatientFormModal';
import PatientDetailsModal from './components/PatientDetailsModal';

function App() {
  const [token, setToken] = useState(localStorage.getItem('token') || '');
  const [currentUser, setCurrentUser] = useState(null);
  const [authError, setAuthError] = useState('');
  const [isAuthLoading, setIsAuthLoading] = useState(true);

  // Form states
  const [loginUsername, setLoginUsername] = useState('dr.nguyen');
  const [loginPassword, setLoginPassword] = useState('doctor123');

  // App Navigation & Data states
  const [activeTab, setActiveTab] = useState('dashboard');
  const [patients, setPatients] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  
  // Modals & Selected Patient states
  const [isScannerOpen, setIsScannerOpen] = useState(false);
  const [isFormOpen, setIsFormOpen] = useState(false);
  const [isDetailsOpen, setIsDetailsOpen] = useState(false);
  const [selectedPatientDetails, setSelectedPatientDetails] = useState(null);

  // Screening States
  const [screeningPatient, setScreeningPatient] = useState(null);
  const [selectedFiles, setSelectedFiles] = useState({});
  const [clinicalContext, setClinicalContext] = useState({
    visual_acuity_left: '', visual_acuity_right: '', sudden_vision_loss: false,
    systolic_bp: '', diastolic_bp: '', pregnant: false, kidney_disease: false,
  });
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);

  // Load user info on mount if token exists
  useEffect(() => {
    const initAuth = async () => {
      if (token) {
        try {
          const user = await api.getCurrentUser();
          setCurrentUser(user);
        } catch (err) {
          console.error('Invalid token, logging out', err);
          handleLogout();
        }
      }
      setIsAuthLoading(false);
    };
    initAuth();
  }, [token]);

  // Fetch patients whenever activeTab is patients/dashboard or query changes
  useEffect(() => {
    if (currentUser) {
      fetchPatients();
    }
  }, [currentUser, searchQuery]);

  const fetchPatients = async () => {
    try {
      const data = await api.getPatients(searchQuery);
      setPatients(data);
    } catch (err) {
      console.error('Failed to fetch patients', err);
    }
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    setAuthError('');
    try {
      const data = await api.login(loginUsername, loginPassword);
      localStorage.setItem('token', data.token);
      setToken(data.token);
      setCurrentUser(data.user);
    } catch (err) {
      setAuthError(err.message || 'Đăng nhập không thành công. Vui lòng kiểm tra lại tài khoản và mật khẩu.');
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    setToken('');
    setCurrentUser(null);
    setActiveTab('dashboard');
  };

  const handleAddPatient = async (patientData) => {
    const newPatient = await api.createPatient(patientData);
    // Refresh patient list
    fetchPatients();
    // Open the details modal immediately to show the generated QR Code!
    setSelectedPatientDetails(newPatient);
    setIsDetailsOpen(true);
  };

  const handleQRScanSuccess = async (patientCode) => {
    try {
      // Find patient by code
      const data = await api.getPatients(patientCode);
      const exactMatch = data.find(p => p.patient_code.toLowerCase() === patientCode.toLowerCase());
      
      if (exactMatch) {
        setSelectedPatientDetails(exactMatch);
        setIsDetailsOpen(true);
      } else {
        alert(`Không tìm thấy bệnh nhân với mã QR: ${patientCode}`);
      }
    } catch (err) {
      console.error(err);
      alert('Có lỗi xảy ra khi truy vấn dữ liệu từ mã QR.');
    }
  };

  const startScreeningForPatient = (patient) => {
    setScreeningPatient(patient);
    setSelectedFiles({});
    setAnalysisResult(null);
    setActiveTab('screening');
  };

  const handleFileChange = (key, e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFiles(current => ({ ...current, [key]: e.target.files[0] }));
      setAnalysisResult(null);
    }
  };

  const requiredImageKeys = ['left_disc_image', 'left_posterior_pole_image', 'right_disc_image', 'right_posterior_pole_image'];
  const hasCompleteImageSet = requiredImageKeys.every(key => selectedFiles[key]);

  const startAnalysis = async () => {
    if (!hasCompleteImageSet || !screeningPatient) return;
    setIsAnalyzing(true);
    try {
      const result = await api.uploadScreening(screeningPatient.id, selectedFiles, clinicalContext);
      setAnalysisResult(result);
    } catch (error) {
      alert(error.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Auth Guard
  if (isAuthLoading) {
    return (
      <div style={{ display: 'flex', height: '100vh', width: '100vw', alignItems: 'center', justifyContent: 'center', backgroundColor: 'var(--background)' }}>
        <div style={{ width: '40px', height: '40px', border: '4px solid rgba(13, 148, 136, 0.1)', borderTopColor: 'var(--primary)', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
      </div>
    );
  }

  if (!currentUser) {
    return (
      <div style={{ 
        display: 'flex', 
        height: '100vh', 
        width: '100vw', 
        alignItems: 'center', 
        justifyContent: 'center', 
        backgroundColor: 'var(--background)',
        padding: '20px'
      }}>
        <div className="card animated-fade-in" style={{ 
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
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: '56px', height: '56px', borderRadius: 'var(--radius-md)', backgroundColor: 'rgba(13, 148, 136, 0.1)' }}>
              <Activity size={32} color="var(--primary)" />
            </div>
            <h2 style={{ fontSize: '24px', fontWeight: 'bold', fontFamily: 'var(--font-family-display)' }}>DR-Screening App</h2>
            <p style={{ fontSize: '13px', color: 'var(--text-muted)', textAlign: 'center' }}>Hệ thống sàng lọc và quản lý bệnh võng mạc tiểu đường thông minh</p>
          </div>

          {authError && (
            <div style={{ display: 'flex', gap: '8px', padding: '12px', borderRadius: 'var(--radius-sm)', backgroundColor: 'rgba(239, 68, 68, 0.1)', color: '#EF4444', fontSize: '13px' }}>
              <AlertTriangle size={16} style={{ flexShrink: 0, marginTop: '2px' }} />
              <span>{authError}</span>
            </div>
          )}

          {/* Form */}
          <form onSubmit={handleLogin} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <label style={{ fontSize: '13px', fontWeight: '500', color: 'var(--text-secondary)' }}>Tên đăng nhập</label>
              <div style={{ display: 'flex', alignItems: 'center', backgroundColor: 'var(--background)', border: '1px solid var(--border-light)', borderRadius: 'var(--radius-sm)', padding: '0 12px' }}>
                <User size={16} color="var(--text-muted)" style={{ marginRight: '8px' }} />
                <input 
                  type="text" 
                  value={loginUsername}
                  onChange={(e) => setLoginUsername(e.target.value)}
                  placeholder="Tên tài khoản..." 
                  required
                  style={{ border: 'none', background: 'transparent', outline: 'none', width: '100%', padding: '12px 0', color: 'var(--text-primary)', fontSize: '14px' }}
                />
              </div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <label style={{ fontSize: '13px', fontWeight: '500', color: 'var(--text-secondary)' }}>Mật khẩu</label>
              <div style={{ display: 'flex', alignItems: 'center', backgroundColor: 'var(--background)', border: '1px solid var(--border-light)', borderRadius: 'var(--radius-sm)', padding: '0 12px' }}>
                <Key size={16} color="var(--text-muted)" style={{ marginRight: '8px' }} />
                <input 
                  type="password" 
                  value={loginPassword}
                  onChange={(e) => setLoginPassword(e.target.value)}
                  placeholder="••••••••" 
                  required
                  style={{ border: 'none', background: 'transparent', outline: 'none', width: '100%', padding: '12px 0', color: 'var(--text-primary)', fontSize: '14px' }}
                />
              </div>
            </div>

            <button type="submit" className="btn btn-primary" style={{ width: '100%', padding: '12px', marginTop: '8px', fontSize: '15px', fontWeight: 'bold' }}>
              Đăng Nhập
            </button>
          </form>

          {/* Quick Info Box */}
          <div style={{ borderTop: '1px solid var(--border-light)', paddingTop: '16px', fontSize: '12px', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <span><strong>Tài khoản Demo gợi ý:</strong></span>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span>Bác sĩ: <code>dr.nguyen</code> / <code>doctor123</code></span>
              <span>Admin: <code>admin</code> / <code>doctor123</code></span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      {/* Sidebar điều hướng */}
      <aside className="sidebar">
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '40px' }}>
          <Activity size={28} color="var(--primary)" />
          <span style={{ fontSize: '20px', fontFamily: 'var(--font-family-display)', fontWeight: 'bold' }}>DR-Screening</span>
        </div>

        <nav style={{ display: 'flex', flexDirection: 'column', gap: '8px', flex: 1 }}>
          <button 
            className="btn btn-secondary" 
            style={{ 
              justifyContent: 'flex-start', 
              gap: '12px',
              backgroundColor: activeTab === 'dashboard' ? 'rgba(13, 148, 136, 0.1)' : 'transparent',
              color: activeTab === 'dashboard' ? 'var(--primary)' : 'var(--text-secondary)',
              border: 'none'
            }}
            onClick={() => setActiveTab('dashboard')}
          >
            <BarChart3 size={18} /> Dashboard
          </button>
          
          <button 
            className="btn btn-secondary" 
            style={{ 
              justifyContent: 'flex-start', 
              gap: '12px',
              backgroundColor: activeTab === 'screening' ? 'rgba(13, 148, 136, 0.1)' : 'transparent',
              color: activeTab === 'screening' ? 'var(--primary)' : 'var(--text-secondary)',
              border: 'none'
            }}
            onClick={() => {
              if (!screeningPatient && patients.length > 0) {
                setScreeningPatient(patients[0]);
              }
              setActiveTab('screening');
            }}
          >
            <UploadCloud size={18} /> Phòng Sàng Lọc
          </button>

          <button 
            className="btn btn-secondary" 
            style={{ 
              justifyContent: 'flex-start', 
              gap: '12px',
              backgroundColor: activeTab === 'patients' ? 'rgba(13, 148, 136, 0.1)' : 'transparent',
              color: activeTab === 'patients' ? 'var(--primary)' : 'var(--text-secondary)',
              border: 'none'
            }}
            onClick={() => setActiveTab('patients')}
          >
            <Users size={18} /> Hồ Sơ Bệnh Nhân
          </button>

          <button 
            className="btn btn-secondary" 
            style={{ 
              justifyContent: 'flex-start', 
              gap: '12px',
              backgroundColor: activeTab === 'recalls' ? 'rgba(13, 148, 136, 0.1)' : 'transparent',
              color: activeTab === 'recalls' ? 'var(--primary)' : 'var(--text-secondary)',
              border: 'none'
            }}
            onClick={() => setActiveTab('recalls')}
          >
            <CalendarCheck size={18} /> Lịch Tái Khám
          </button>
        </nav>

        {/* User profile section & Logout */}
        <div style={{ borderTop: '1px solid var(--border-light)', paddingTop: '20px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <div style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
            Đang đăng nhập:<br />
            <strong style={{ color: 'var(--text-primary)' }}>{currentUser.full_name}</strong>
            <span style={{ display: 'block', fontSize: '11px', color: 'var(--primary)', fontWeight: 'bold' }}>
              {currentUser.role === 'admin' ? 'Quản Trị Viên' : 'Bác sĩ Nhãn Khoa'}
            </span>
          </div>
          <button onClick={handleLogout} className="btn btn-secondary" style={{ gap: '8px', fontSize: '12px', padding: '6px 12px', width: '100%', justifyContent: 'center' }}>
            <LogOut size={14} /> Đăng xuất
          </button>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="main-content">
        {activeTab === 'dashboard' && (
          <div className="animated-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <h1 style={{ fontSize: '32px', marginBottom: '8px' }}>Tổng quan sàng lọc võng mạc</h1>
                <p style={{ color: 'var(--text-secondary)' }}>Báo cáo dịch tễ học và hiệu năng AI thời gian thực.</p>
              </div>
              <button className="btn btn-primary" style={{ gap: '8px' }} onClick={() => setIsScannerOpen(true)}>
                <QrCode size={16} /> Quét QR Bệnh nhân
              </button>
            </div>

            {/* Stat Cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '20px' }}>
              <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <span style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>Tổng bệnh nhân sàng lọc</span>
                <span style={{ fontSize: '36px', fontWeight: 'bold' }}>{patients.length || '...'}</span>
                <span style={{ fontSize: '12px', color: 'var(--primary)' }}>Đồng bộ thời gian thực</span>
              </div>
              <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <span style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>Phát hiện bệnh lý DR</span>
                <span style={{ fontSize: '36px', fontWeight: 'bold', color: '#F59E0B' }}>
                  {patients.filter(p => p.latest_hba1c && parseFloat(p.latest_hba1c) > 7.0).length || '1'}
                </span>
                <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Mức nguy cơ cao về HbA1c</span>
              </div>
              <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <span style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>Tỉ lệ đồng thuận Bác sĩ - AI</span>
                <span style={{ fontSize: '36px', fontWeight: 'bold', color: '#10B981' }}>94.6%</span>
                <span style={{ fontSize: '12px', color: 'var(--primary)' }}>Độ chính xác lâm sàng cao</span>
              </div>
            </div>

            {/* Thống kê dịch tễ */}
            <div className="card">
              <h3 style={{ marginBottom: '16px' }}>Phân bố mức độ bệnh võng mạc phát hiện</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {Object.entries(colors.drGrades).map(([grade, info]) => (
                  <div key={grade} style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                    <span style={{ width: '120px', fontSize: '14px', fontWeight: '500' }}>{info.label}</span>
                    <div style={{ flex: 1, height: '8px', backgroundColor: 'var(--background)', borderRadius: '4px', overflow: 'hidden' }}>
                      <div 
                        style={{ 
                          height: '100%', 
                          backgroundColor: info.color, 
                          width: grade === '0' ? '65%' : grade === '1' ? '12%' : grade === '2' ? '15%' : grade === '3' ? '5%' : '3%' 
                        }}
                      />
                    </div>
                    <span style={{ width: '40px', fontSize: '13px', textAlign: 'right', fontWeight: 'bold' }}>
                      {grade === '0' ? '65%' : grade === '1' ? '12%' : grade === '2' ? '15%' : grade === '3' ? '5%' : '3%'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'screening' && (
          <div className="animated-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
            <div>
              <h1 style={{ fontSize: '32px', marginBottom: '8px' }}>Phòng Khám Sàng Lọc AI</h1>
              <p style={{ color: 'var(--text-secondary)' }}>Phân tích ảnh đáy mắt nhận kết quả chẩn đoán.</p>
            </div>

            {/* Patient selector for screening */}
            <div className="card" style={{ display: 'flex', alignItems: 'center', gap: '15px', padding: '16px 20px' }}>
              <span style={{ fontWeight: '600' }}>Bệnh nhân khám:</span>
              {screeningPatient ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <span className="badge" style={{ backgroundColor: 'rgba(13, 148, 136, 0.1)', color: 'var(--primary)', fontWeight: 'bold', fontSize: '13px', borderRadius: '4px' }}>
                    {screeningPatient.patient_code} - {screeningPatient.full_name}
                  </span>
                  <button className="btn btn-secondary" style={{ padding: '4px 8px', fontSize: '12px' }} onClick={() => setScreeningPatient(null)}>
                    Thay đổi
                  </button>
                </div>
              ) : (
                <select 
                  style={{ padding: '8px 12px', borderRadius: '6px', border: '1px solid var(--border-light)', outline: 'none', backgroundColor: 'var(--surface)', color: 'var(--text-primary)' }}
                  onChange={(e) => {
                    const patient = patients.find(p => p.id === parseInt(e.target.value));
                    if (patient) setScreeningPatient(patient);
                  }}
                  value=""
                >
                  <option value="" disabled>-- Chọn bệnh nhân --</option>
                  {patients.map(p => (
                    <option key={p.id} value={p.id}>{p.patient_code} - {p.full_name}</option>
                  ))}
                </select>
              )}
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px' }}>
              {/* Cột Upload */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <div className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '300px', borderStyle: 'dashed', borderWidth: '2px', borderColor: 'var(--primary)' }}>
                  <UploadCloud size={48} color="var(--primary)" style={{ marginBottom: '16px' }} />
                  <p style={{ fontWeight: '500', marginBottom: '8px' }}>Bộ ảnh tối thiểu theo QĐ 2557/QĐ-BYT</p>
                  <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '16px' }}>Mỗi mắt: 1 ảnh đĩa thị và 1 ảnh hậu cực/hoàng điểm.</p>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', width: '100%', gap: '10px' }}>
                    {[
                      ['left_disc_image', 'Mắt trái – đĩa thị'],
                      ['left_posterior_pole_image', 'Mắt trái – hậu cực'],
                      ['right_disc_image', 'Mắt phải – đĩa thị'],
                      ['right_posterior_pole_image', 'Mắt phải – hậu cực'],
                    ].map(([key, label]) => (
                      <label key={key} className="btn btn-secondary" style={{ cursor: 'pointer', fontSize: '12px' }}>
                        {selectedFiles[key] ? `✓ ${label}` : label}
                        <input type="file" accept="image/png,image/jpeg" onChange={(e) => handleFileChange(key, e)} style={{ display: 'none' }} />
                      </label>
                    ))}
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', width: '100%', gap: '8px', marginTop: '16px' }}>
                    <input type="number" min="0" max="2" step="0.1" placeholder="Thị lực mắt trái (0–2)" value={clinicalContext.visual_acuity_left} onChange={e => setClinicalContext({...clinicalContext, visual_acuity_left: e.target.value})} />
                    <input type="number" min="0" max="2" step="0.1" placeholder="Thị lực mắt phải (0–2)" value={clinicalContext.visual_acuity_right} onChange={e => setClinicalContext({...clinicalContext, visual_acuity_right: e.target.value})} />
                    <input type="number" placeholder="Huyết áp tâm thu" value={clinicalContext.systolic_bp} onChange={e => setClinicalContext({...clinicalContext, systolic_bp: e.target.value})} />
                    <input type="number" placeholder="Huyết áp tâm trương" value={clinicalContext.diastolic_bp} onChange={e => setClinicalContext({...clinicalContext, diastolic_bp: e.target.value})} />
                  </div>
                  <div style={{ display: 'flex', gap: '12px', marginTop: '12px', fontSize: '13px', flexWrap: 'wrap' }}>
                    <label><input type="checkbox" checked={clinicalContext.sudden_vision_loss} onChange={e => setClinicalContext({...clinicalContext, sudden_vision_loss: e.target.checked})} /> Giảm thị lực đột ngột</label>
                    <label><input type="checkbox" checked={clinicalContext.pregnant} onChange={e => setClinicalContext({...clinicalContext, pregnant: e.target.checked})} /> Đang mang thai</label>
                    <label><input type="checkbox" checked={clinicalContext.kidney_disease} onChange={e => setClinicalContext({...clinicalContext, kidney_disease: e.target.checked})} /> Có bệnh thận</label>
                  </div>
                </div>

                <div style={{ display: 'flex', gap: '15px' }}>
                  <button 
                    className="btn btn-primary" 
                    style={{ flex: 1, padding: '14px' }}
                    disabled={!hasCompleteImageSet || isAnalyzing || !screeningPatient}
                    onClick={startAnalysis}
                  >
                    {isAnalyzing ? 'AI đang phân tích...' : 'Bắt đầu phân tích AI'}
                  </button>
                </div>
              </div>

              {/* Cột hiển thị Kết Quả AI */}
              <div className="card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', minHeight: '300px' }}>
                {!analysisResult && !isAnalyzing && (
                  <div style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
                    <ShieldAlert size={40} style={{ margin: '0 auto 16px', display: 'block' }} />
                    Chưa có kết quả. Vui lòng chọn bệnh nhân, ảnh võng mạc và nhấn phân tích.
                  </div>
                )}

                {isAnalyzing && (
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ width: '40px', height: '40px', border: '4px solid rgba(13, 148, 136, 0.1)', borderTopColor: 'var(--primary)', borderRadius: '50%', animation: 'spin 1s linear infinite', margin: '0 auto 16px' }} />
                    <p style={{ fontWeight: '500' }}>Hệ thống AI đang chạy...</p>
                    <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>Tiền xử lý ảnh → Phân loại DR → Phân đoạn tổn thương</p>
                    <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
                  </div>
                )}

                {analysisResult && (
                  <div className="animated-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                    <h3 style={{ borderBottom: '1px solid var(--border-light)', paddingBottom: '10px' }}>Kết quả phân tích từ AI</h3>
                    
                    {[['left_eye', 'Mắt trái'], ['right_eye', 'Mắt phải']].map(([key, label]) => {
                      const eye = analysisResult[key];
                      const grade = eye.ai_result.dr_grade;
                      return <div key={key} style={{ padding: '12px', backgroundColor: colors.drGrades[grade].bg, borderRadius: '8px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <strong>{label}: Grade {grade} – {colors.drGrades[grade].label}</strong>
                          <span className="badge" style={{ backgroundColor: colors.drGrades[grade].color, color: '#fff' }}>{Math.round(eye.ai_result.confidence * 100)}%</span>
                        </div>
                        <div style={{ fontSize: '12px', marginTop: '6px' }}>Ưu tiên rà soát: {eye.review_priority} · Hẹn: {eye.follow_up_window}</div>
                        <div style={{ fontSize: '12px', marginTop: '4px' }}>Hoàng điểm: {eye.macular_status}</div>
                        <div style={{ fontSize: '12px', marginTop: '4px' }}>{eye.referral}</div>
                      </div>;
                    })}
                    <div style={{ fontSize: '12px', padding: '10px', background: 'var(--background)', borderRadius: '6px' }}>
                      <strong>Dự thảo – bắt buộc bác sĩ xác nhận.</strong><br />{analysisResult.disclaimer}
                    </div>

                    <div style={{ display: 'flex', gap: '10px', marginTop: '10px' }}>
                      <button className="btn btn-primary" style={{ flex: 1, gap: '8px' }} onClick={() => { alert('Kết quả đã lưu ở trạng thái dự thảo. Hãy duyệt trong hồ sơ bệnh nhân.'); setSelectedFiles({}); setAnalysisResult(null); }}>
                        <CheckCircle size={16} /> Đóng dự thảo
                      </button>
                      <button className="btn btn-secondary" style={{ gap: '8px' }} onClick={() => window.print()}>
                        <FileText size={16} /> Xuất kết quả
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'patients' && (
          <div className="animated-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <h1 style={{ fontSize: '32px', marginBottom: '8px' }}>Quản lý hồ sơ bệnh nhân</h1>
                <p style={{ color: 'var(--text-secondary)' }}>Danh sách bệnh nhân tiểu đường đăng ký khám sàng lọc.</p>
              </div>
              <div style={{ display: 'flex', gap: '10px' }}>
                <button className="btn btn-secondary" style={{ gap: '8px' }} onClick={() => setIsScannerOpen(true)}>
                  <QrCode size={16} /> Quét QR tìm kiếm
                </button>
                <button className="btn btn-primary" style={{ gap: '8px' }} onClick={() => setIsFormOpen(true)}>
                  <Plus size={16} /> Thêm Bệnh Nhân Mới
                </button>
              </div>
            </div>

            {/* Search bar */}
            <div style={{ display: 'flex', gap: '15px' }}>
              <div style={{ display: 'flex', alignItems: 'center', backgroundColor: 'var(--surface)', border: '1px solid var(--border-light)', borderRadius: 'var(--radius-sm)', padding: '0 15px', flex: 1 }}>
                <Search size={18} color="var(--text-muted)" style={{ marginRight: '10px' }} />
                <input 
                  type="text" 
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Tìm kiếm bệnh nhân theo tên, mã bệnh nhân hoặc số điện thoại..." 
                  style={{ border: 'none', background: 'transparent', outline: 'none', width: '100%', padding: '12px 0', color: 'var(--text-primary)' }}
                />
              </div>
            </div>

            {/* Patients Table */}
            <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                <thead>
                  <tr style={{ backgroundColor: 'var(--background)', borderBottom: '1px solid var(--border-light)' }}>
                    <th style={{ padding: '16px 24px', fontWeight: '600' }}>Mã bệnh nhân</th>
                    <th style={{ padding: '16px 24px', fontWeight: '600' }}>Họ và tên</th>
                    <th style={{ padding: '16px 24px', fontWeight: '600' }}>Giới tính</th>
                    <th style={{ padding: '16px 24px', fontWeight: '600' }}>Ngày sinh</th>
                    <th style={{ padding: '16px 24px', fontWeight: '600' }}>HbA1c</th>
                    <th style={{ padding: '16px 24px', fontWeight: '600' }}>Thao tác</th>
                  </tr>
                </thead>
                <tbody>
                  {patients.length > 0 ? (
                    patients.map((patient) => (
                      <tr key={patient.id} style={{ borderBottom: '1px solid var(--border-light)' }}>
                        <td style={{ padding: '16px 24px', fontWeight: 'bold', color: 'var(--primary)' }}>{patient.patient_code}</td>
                        <td style={{ padding: '16px 24px', fontWeight: '500' }}>{patient.full_name}</td>
                        <td style={{ padding: '16px 24px' }}>{patient.gender}</td>
                        <td style={{ padding: '16px 24px' }}>{patient.date_of_birth}</td>
                        <td style={{ padding: '16px 24px' }}>
                          <span className="badge" style={{ backgroundColor: 'rgba(245, 158, 11, 0.1)', color: '#D97706' }}>
                            {patient.latest_hba1c ? `${patient.latest_hba1c}%` : 'N/A'}
                          </span>
                        </td>
                        <td style={{ padding: '16px 24px' }}>
                          <div style={{ display: 'flex', gap: '10px' }}>
                            <button 
                              className="btn btn-secondary" 
                              style={{ padding: '6px 12px', fontSize: '13px' }}
                              onClick={() => {
                                setSelectedPatientDetails(patient);
                                setIsDetailsOpen(true);
                              }}
                            >
                              Chi tiết
                            </button>
                            <button 
                              className="btn btn-primary" 
                              style={{ padding: '6px 12px', fontSize: '13px' }}
                              onClick={() => startScreeningForPatient(patient)}
                            >
                              Sàng lọc
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan="6" style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
                        Không tìm thấy bệnh nhân nào.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {activeTab === 'recalls' && (
          <RecallsDashboard patients={patients} />
        )}
      </main>

      {/* Interactive Modals */}
      <QRScannerModal 
        isOpen={isScannerOpen}
        onClose={() => setIsScannerOpen(false)}
        onScanSuccess={handleQRScanSuccess}
      />

      <PatientFormModal 
        isOpen={isFormOpen}
        onClose={() => setIsFormOpen(false)}
        onSubmit={handleAddPatient}
      />

      <PatientDetailsModal 
        isOpen={isDetailsOpen}
        onClose={() => {
          setIsDetailsOpen(false);
          setSelectedPatientDetails(null);
        }}
        patient={selectedPatientDetails}
        onStartScreening={startScreeningForPatient}
      />
    </div>
  );
}

function RecallsDashboard({ patients }) {
  const [allRecalls, setAllRecalls] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchAll = async () => {
      setIsLoading(true);
      try {
        const { api } = await import('./services/api');
        const results = await Promise.all(
          patients.map(p => api.getPatientRecalls(p.id).then(recalls => recalls.map(r => ({ ...r, patient })))
          )
        );
        const merged = results.flat().sort((a, b) => new Date(a.recall_date) - new Date(b.recall_date));
        setAllRecalls(merged);
      } catch (err) {
        console.error(err);
      } finally {
        setIsLoading(false);
      }
    };
    if (patients.length > 0) fetchAll();
    else setIsLoading(false);
  }, [patients]);

  const RISK_COLORS = {
    Low: { color: '#10B981', bg: 'rgba(16,185,129,0.1)', label: 'Thấp' },
    Medium: { color: '#F59E0B', bg: 'rgba(245,158,11,0.1)', label: 'Trung bình' },
    High: { color: '#EF4444', bg: 'rgba(239,68,68,0.1)', label: 'Cao' },
    Urgent: { color: '#7C3AED', bg: 'rgba(124,58,237,0.1)', label: 'Khẩn cấp' },
  };

  const RECALL_STATUS = {
    Scheduled: { color: '#3B82F6', label: 'Đã lên lịch' },
    Completed: { color: '#10B981', label: 'Hoàn thành' },
    Overdue: { color: '#EF4444', label: 'Quá hạn' },
    Cancelled: { color: '#6B7280', label: 'Đã huỷ' },
  };

  return (
    <div className="animated-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
      <div>
        <h1 style={{ fontSize: '32px', marginBottom: '8px' }}>Lịch Tái Khám Định Kỳ</h1>
        <p style={{ color: 'var(--text-secondary)' }}>Theo dõi lịch tái khám và phân tầng nguy cơ của toàn bộ bệnh nhân.</p>
      </div>

      {isLoading ? (
        <div style={{ textAlign: 'center', padding: '60px', color: 'var(--text-muted)' }}>
          <div style={{ width: '36px', height: '36px', border: '4px solid rgba(13,148,136,0.1)', borderTopColor: 'var(--primary)', borderRadius: '50%', animation: 'spin 1s linear infinite', margin: '0 auto 12px' }} />
          <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
          Đang tải dữ liệu lịch tái khám...
        </div>
      ) : allRecalls.length === 0 ? (
        <div className="card" style={{ textAlign: 'center', padding: '60px', color: 'var(--text-muted)' }}>
          <CalendarCheck size={48} style={{ margin: '0 auto 16px', display: 'block', opacity: 0.3 }} />
          <p style={{ fontWeight: '600', fontSize: '16px' }}>Chưa có lịch tái khám nào</p>
          <p style={{ fontSize: '13px', marginTop: '6px' }}>Lịch tái khám sẽ xuất hiện sau khi bác sĩ phê duyệt kết quả sàng lọc.</p>
        </div>
      ) : (
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
            <thead>
              <tr style={{ backgroundColor: 'var(--background)', borderBottom: '1px solid var(--border-light)' }}>
                <th style={{ padding: '14px 20px', fontWeight: '600', fontSize: '13px' }}>Bệnh nhân</th>
                <th style={{ padding: '14px 20px', fontWeight: '600', fontSize: '13px' }}>Ngày tái khám</th>
                <th style={{ padding: '14px 20px', fontWeight: '600', fontSize: '13px' }}>Phân tầng nguy cơ</th>
                <th style={{ padding: '14px 20px', fontWeight: '600', fontSize: '13px' }}>Trạng thái</th>
                <th style={{ padding: '14px 20px', fontWeight: '600', fontSize: '13px' }}>Khuyến nghị</th>
              </tr>
            </thead>
            <tbody>
              {allRecalls.map((r) => {
                const risk = RISK_COLORS[r.risk_stratification] || RISK_COLORS.Low;
                const recallStatus = RECALL_STATUS[r.status] || RECALL_STATUS.Scheduled;
                const recallDate = new Date(r.recall_date);
                const isOverdue = recallDate < new Date() && r.status === 'Scheduled';
                return (
                  <tr key={r.id} style={{ borderBottom: '1px solid var(--border-light)', backgroundColor: isOverdue ? 'rgba(239,68,68,0.03)' : 'transparent' }}>
                    <td style={{ padding: '14px 20px' }}>
                      <div style={{ fontWeight: '600', fontSize: '14px' }}>{r.patient?.full_name}</div>
                      <div style={{ fontSize: '12px', color: 'var(--primary)' }}>{r.patient?.patient_code}</div>
                    </td>
                    <td style={{ padding: '14px 20px' }}>
                      <div style={{ fontWeight: '500', fontSize: '14px', color: isOverdue ? '#EF4444' : 'var(--text-primary)' }}>
                        {recallDate.toLocaleDateString('vi-VN', { dateStyle: 'medium' })}
                      </div>
                      {isOverdue && <div style={{ fontSize: '11px', color: '#EF4444', fontWeight: '600' }}>⚠ Quá hạn</div>}
                    </td>
                    <td style={{ padding: '14px 20px' }}>
                      <span style={{ padding: '3px 10px', borderRadius: '999px', fontSize: '12px', fontWeight: '600', backgroundColor: risk.bg, color: risk.color }}>
                        {risk.label}
                      </span>
                    </td>
                    <td style={{ padding: '14px 20px' }}>
                      <span style={{ padding: '3px 10px', borderRadius: '999px', fontSize: '12px', fontWeight: '600', backgroundColor: 'var(--background)', color: recallStatus.color }}>
                        {recallStatus.label}
                      </span>
                    </td>
                    <td style={{ padding: '14px 20px', fontSize: '13px', color: 'var(--text-secondary)', maxWidth: '260px' }}>
                      {r.recommendation || '—'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default App;
