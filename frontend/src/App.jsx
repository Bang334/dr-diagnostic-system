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
  Minus,
  CalendarCheck,
  ClipboardList,
  Menu,
  X,
  Award,
  Calculator,
  ChevronDown,
  Gauge,
  Target,
  CalendarPlus,
  BookOpenCheck,
  Eye,
  Maximize2,
  ZoomIn
} from 'lucide-react';
import { colors } from './theme/colors';
import { api } from './services/api';

// Import custom modals
import QRScannerModal from './components/QRScannerModal';
import PatientFormModal from './components/PatientFormModal';
import PatientDetailsModal from './components/PatientDetailsModal';
import SecureImage from './components/SecureImage';
import PatientPortal from './components/PatientPortal';
import { DoctorReviewForm } from './components/ScreeningDetailPanel';
import { useAppDialog } from './components/AppDialogProvider';
import ProjectEvidencePage from './components/ProjectEvidencePage';

const RECALL_MONTH_OPTIONS = [1, 2, 3, 6, 12];
const DEFAULT_GRADING_MODELS = [
  {
    id: 'grading',
    label: 'Grading gốc',
    checkpoint: 'checkpoint-best.pth',
    description: 'Classifier RETFound 5 mức DR, dùng checkpoint grading gốc.',
    ready: true,
  },
  {
    id: 'fewshot',
    label: 'Few-shot DeepDRiD',
    checkpoint: 'best-fewshot.pth',
    description: 'RETFound ProtoNet đã thích nghi trên tập DeepDRiD few-shot.',
    ready: true,
  },
];

const addMonthsToToday = (months) => {
  const today = new Date();
  const originalDay = today.getDate();
  today.setDate(1);
  today.setMonth(today.getMonth() + Number(months));
  const lastDay = new Date(today.getFullYear(), today.getMonth() + 1, 0).getDate();
  today.setDate(Math.min(originalDay, lastDay));
  const year = today.getFullYear();
  const month = String(today.getMonth() + 1).padStart(2, '0');
  const day = String(today.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
};

const suggestedRecallMonths = (followUp) => {
  const suggested = Number(String(followUp || '').match(/\d+/)?.[0]);
  if (RECALL_MONTH_OPTIONS.includes(suggested)) return suggested;
  return 6;
};

const SEMI_SUPERVISED_TEST_RESULT = {
  loss: 0.4953,
  accuracy: 0.8410,
  macro_f1: 0.8429,
  balanced_accuracy: 0.8486,
  qwk: 0.9144,
  per_class_recall: {
    'No DR': 0.8638,
    Mild: 0.8035,
    Moderate: 0.7247,
    Severe: 0.9500,
    Proliferative: 0.9012,
  },
};

const EXPECTED_DISAGREEMENT_BALANCED = 0.25;
const MAX_GRADE_DISTANCE_SQUARED = 16;
const GRADE_ERROR_EXAMPLE = {
  samples: 1000,
  accuracy: SEMI_SUPERVISED_TEST_RESULT.accuracy,
  qwk: SEMI_SUPERVISED_TEST_RESULT.qwk,
};
const GRADE_ERROR_EXAMPLE_RATE = 1 - GRADE_ERROR_EXAMPLE.accuracy;
const GRADE_ERROR_EXAMPLE_CORRECT = GRADE_ERROR_EXAMPLE.samples * GRADE_ERROR_EXAMPLE.accuracy;
const GRADE_ERROR_EXAMPLE_WRONG = GRADE_ERROR_EXAMPLE.samples * GRADE_ERROR_EXAMPLE_RATE;
const GRADE_ERROR_OBSERVED_DISAGREEMENT = (
  (1 - GRADE_ERROR_EXAMPLE.qwk) * EXPECTED_DISAGREEMENT_BALANCED
);
const GRADE_ERROR_SQUARED_SUM = (
  GRADE_ERROR_EXAMPLE.samples
  *
  MAX_GRADE_DISTANCE_SQUARED
  * GRADE_ERROR_OBSERVED_DISAGREEMENT
);
const GRADE_ERROR_WRONG_MSE = GRADE_ERROR_SQUARED_SUM / GRADE_ERROR_EXAMPLE_WRONG;
const GRADE_ERROR_WRONG_RMSE = Math.sqrt(GRADE_ERROR_WRONG_MSE);
const GRADE_ERROR_MAE_MIN = (GRADE_ERROR_WRONG_MSE + 4) / 5;
const GRADE_ERROR_MAE_MAX = (GRADE_ERROR_WRONG_MSE + 2) / 3;
const GRADE_ERROR_MAE_MIDPOINT = (GRADE_ERROR_MAE_MIN + GRADE_ERROR_MAE_MAX) / 2;

const RECALL_LABELS = {
  'No DR': 'Không DR',
  Mild: 'DR nhẹ',
  Moderate: 'DR trung bình',
  Severe: 'DR nặng',
  Proliferative: 'DR tăng sinh',
};

const formatPercent = (value) => `${(value * 100).toFixed(2)}%`;
const formatMetric = (value) => value.toFixed(4);
const formatDecimal = (value) => value.toFixed(2);
const formatInteger = (value) => new Intl.NumberFormat('vi-VN').format(value);

function App() {
  const dialog = useAppDialog();
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
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  // Screening States
  const [screeningPatient, setScreeningPatient] = useState(null);
  const [selectedFiles, setSelectedFiles] = useState({});
  const [gradingModels, setGradingModels] = useState(DEFAULT_GRADING_MODELS);
  const [selectedGradingModel, setSelectedGradingModel] = useState('grading');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [showMasks, setShowMasks] = useState({ left_eye: false, right_eye: false });
  const [lightboxImage, setLightboxImage] = useState(null);
  const [currentScreeningDetail, setCurrentScreeningDetail] = useState(null);
  const [summarySource, setSummarySource] = useState('ai');

  // Quick Recall Modal State
  const [isQuickRecallOpen, setIsQuickRecallOpen] = useState(false);
  const [quickRecallForm, setQuickRecallForm] = useState({ recall_in_months: 6, risk_stratification: 'Low', recommendation: '' });
  const [isSubmittingRecall, setIsSubmittingRecall] = useState(false);

  useEffect(() => {
    if (!isQuickRecallOpen) return undefined;
    const closeOnEscape = (event) => {
      if (event.key === 'Escape') setIsQuickRecallOpen(false);
    };
    window.addEventListener('keydown', closeOnEscape);
    return () => window.removeEventListener('keydown', closeOnEscape);
  }, [isQuickRecallOpen]);

  // Layout States
  const [isUploadCollapsed, setIsUploadCollapsed] = useState(false);

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
    if (currentUser && currentUser.role !== 'patient') {
      fetchPatients();
    }
  }, [currentUser, searchQuery]);

  useEffect(() => {
    if (!currentUser || currentUser.role === 'patient') return;
    let cancelled = false;
    api.getScreeningModels()
      .then((data) => {
        if (cancelled || !Array.isArray(data.models)) return;
        const descriptions = Object.fromEntries(
          DEFAULT_GRADING_MODELS.map((model) => [model.id, model.description]),
        );
        const models = data.models.map((model) => ({
          ...model,
          description: descriptions[model.id] || 'Model phân độ võng mạc cục bộ.',
        }));
        setGradingModels(models);
        const nextModel = models.find(
          (model) => model.id === data.default && model.ready,
        ) || models.find((model) => model.ready);
        if (nextModel) setSelectedGradingModel(nextModel.id);
      })
      .catch((error) => console.warn('Could not load grading model options', error));
    return () => {
      cancelled = true;
    };
  }, [currentUser]);

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

  const navigateTo = (tab) => {
    setActiveTab(tab);
    setIsSidebarOpen(false);
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
        dialog.showWarning(`Không tìm thấy bệnh nhân với mã QR: ${patientCode}`, 'Không tìm thấy hồ sơ');
      }
    } catch (err) {
      console.error(err);
      dialog.showError('Có lỗi xảy ra khi truy vấn dữ liệu từ mã QR.', 'Không thể đọc hồ sơ');
    }
  };

  const startScreeningForPatient = (patient) => {
    setScreeningPatient(patient);
    setSelectedFiles({});
    setAnalysisResult(null);
    setCurrentScreeningDetail(null);
    setSummarySource('ai');
    setIsDetailsOpen(false);
    setSelectedPatientDetails(null);
    setIsUploadCollapsed(false);
    setActiveTab('screening');
  };

  const handleFileChange = (key, e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFiles(current => ({ ...current, [key]: e.target.files[0] }));
      setAnalysisResult(null);
      setCurrentScreeningDetail(null);
      setSummarySource('ai');
    }
  };

  const screeningImageKeys = ['left_fundus_image', 'right_fundus_image'];
  const hasScreeningImage = screeningImageKeys.some(key => selectedFiles[key]);
  const selectedClinicalSummary = analysisResult
    ? (summarySource === 'rules'
      ? analysisResult.rule_summary
      : analysisResult.clinical_summary)
    : null;
  const selectedGradingModelInfo = gradingModels.find(
    (model) => model.id === selectedGradingModel,
  ) || DEFAULT_GRADING_MODELS[0];
  const isSelectedGradingModelReady = Boolean(selectedGradingModelInfo.ready);
  const canReviewScreening = currentUser?.role === 'admin'
    || (
      currentUser?.role === 'doctor'
      && (currentUser.hospital_department || '').toLocaleLowerCase('vi-VN').includes('nhãn')
    );
  const quickRecallDate = addMonthsToToday(quickRecallForm.recall_in_months);

  const stepRecallMonths = (direction) => {
    const currentIndex = RECALL_MONTH_OPTIONS.indexOf(Number(quickRecallForm.recall_in_months));
    const nextIndex = Math.min(
      RECALL_MONTH_OPTIONS.length - 1,
      Math.max(0, currentIndex + direction),
    );
    setQuickRecallForm((current) => ({
      ...current,
      recall_in_months: RECALL_MONTH_OPTIONS[nextIndex],
    }));
  };

  const startAnalysis = async () => {
    if (!hasScreeningImage || !screeningPatient || !isSelectedGradingModelReady) return;
    setIsAnalyzing(true);
    setIsUploadCollapsed(true);
    try {
      const result = await api.uploadScreening(
        screeningPatient.id,
        selectedFiles,
        selectedGradingModel,
      );
      setAnalysisResult(result);
      setSummarySource(result.clinical_summary?.status === 'generated' ? 'ai' : 'rules');
      try {
        setCurrentScreeningDetail(await api.getScreeningDetail(result.screening_id));
      } catch (detailError) {
        console.error('Could not load the persisted screening detail', detailError);
      }
    } catch (error) {
      dialog.showError(error.message, 'Không thể phân tích ảnh');
      setIsUploadCollapsed(false);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const submitCurrentScreeningReview = async (payload) => {
    if (!analysisResult?.screening_id) return;
    await api.reviewScreening(analysisResult.screening_id, payload);
    const detail = await api.getScreeningDetail(analysisResult.screening_id);
    setCurrentScreeningDetail(detail);
    setAnalysisResult((current) => ({ ...current, status: detail.status }));
    dialog.showSuccess(
      'Kết luận bác sĩ và kế hoạch tái khám đã được lưu. Bệnh nhân hiện có thể xem chi tiết lần khám.',
      'Đã duyệt kết quả thành công',
    );
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
            <h2 style={{ fontSize: '24px', fontWeight: 'bold', fontFamily: 'var(--font-family-display)' }}>DR-Screening App</h2>
            <p style={{ fontSize: '13px', color: 'var(--text-muted)', textAlign: 'center' }}>Hệ thống sàng lọc và quản lý bệnh võng mạc tiểu đường thông minh</p>
          </div>

          {authError && (
            <div className="alert alert-error" role="alert">
              <AlertTriangle size={16} style={{ flexShrink: 0, marginTop: '2px' }} />
              <span>{authError}</span>
            </div>
          )}

          {/* Form */}
          <form onSubmit={handleLogin} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <label htmlFor="login-username" style={{ fontSize: '13px', fontWeight: '500', color: 'var(--text-secondary)' }}>Tên đăng nhập</label>
              <div className="input-shell">
                <User size={16} color="var(--text-muted)" style={{ marginRight: '8px' }} />
                <input 
                  type="text" 
                  id="login-username"
                  autoComplete="username"
                  value={loginUsername}
                  onChange={(e) => setLoginUsername(e.target.value)}
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
                  value={loginPassword}
                  onChange={(e) => setLoginPassword(e.target.value)}
                  placeholder="••••••••" 
                  required
                  style={{ border: 'none', background: 'transparent', outline: 'none', width: '100%', padding: '12px 0', color: 'var(--text-primary)', fontSize: '14px' }}
                />
              </div>
            </div>

            <button type="submit" className="btn btn-primary" style={{ width: '100%', padding: '12px', marginTop: '8px', fontSize: '15px', fontWeight: 'bold' }}>
              Đăng nhập
            </button>
          </form>

          {/* Quick Info Box */}
          <div style={{ borderTop: '1px solid var(--border-light)', paddingTop: '16px', fontSize: '12px', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <span><strong>Tài khoản Demo gợi ý:</strong></span>
            <div className="demo-credentials">
              <span>Bác sĩ: <code>dr.nguyen</code> / <code>doctor123</code></span>
              <span>Admin: <code>admin</code> / <code>admin123</code></span>
              <span>Bệnh nhân: <code>BN0001</code> / <code>benhnhan</code></span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (currentUser.role === 'patient') {
    return <PatientPortal currentUser={currentUser} onLogout={handleLogout} />;
  }

  return (
    <div className="app-container">
      <a className="skip-link" href="#main-content">Bỏ qua điều hướng</a>

      <header className="mobile-header">
        <div className="mobile-brand">
          <span className="brand-mark"><Activity size={20} aria-hidden="true" /></span>
          <span>DR-Screening</span>
        </div>
        <button
          type="button"
          className="icon-btn"
          aria-label={isSidebarOpen ? 'Đóng menu điều hướng' : 'Mở menu điều hướng'}
          aria-expanded={isSidebarOpen}
          onClick={() => setIsSidebarOpen((open) => !open)}
        >
          {isSidebarOpen ? <X size={22} /> : <Menu size={22} />}
        </button>
      </header>

      {isSidebarOpen && (
        <button className="sidebar-backdrop" aria-label="Đóng menu điều hướng" onClick={() => setIsSidebarOpen(false)} />
      )}

      {/* Sidebar điều hướng */}
      <aside className={`sidebar ${isSidebarOpen ? 'sidebar-open' : ''}`}>
        <div className="sidebar-brand">
          <span className="brand-mark"><Activity size={22} aria-hidden="true" /></span>
          <span style={{ fontSize: '20px', fontFamily: 'var(--font-family-display)', fontWeight: 'bold' }}>DR-Screening</span>
        </div>

        <nav className="sidebar-nav" aria-label="Điều hướng chính">
          <button 
            className={`nav-item ${activeTab === 'dashboard' ? 'active' : ''}`}
            aria-current={activeTab === 'dashboard' ? 'page' : undefined}
            style={{ 
              justifyContent: 'flex-start', 
              gap: '12px',
              backgroundColor: activeTab === 'dashboard' ? 'rgba(13, 148, 136, 0.1)' : 'transparent',
              color: activeTab === 'dashboard' ? 'var(--primary)' : 'var(--text-secondary)',
              border: 'none'
            }}
            onClick={() => navigateTo('dashboard')}
          >
            <BarChart3 size={18} /> Dashboard
          </button>
          
          <button 
            className={`nav-item ${activeTab === 'screening' ? 'active' : ''}`}
            aria-current={activeTab === 'screening' ? 'page' : undefined}
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
              navigateTo('screening');
            }}
          >
            <UploadCloud size={18} /> Phòng Sàng Lọc
          </button>

          <button 
            className={`nav-item ${activeTab === 'patients' ? 'active' : ''}`}
            aria-current={activeTab === 'patients' ? 'page' : undefined}
            style={{ 
              justifyContent: 'flex-start', 
              gap: '12px',
              backgroundColor: activeTab === 'patients' ? 'rgba(13, 148, 136, 0.1)' : 'transparent',
              color: activeTab === 'patients' ? 'var(--primary)' : 'var(--text-secondary)',
              border: 'none'
            }}
            onClick={() => navigateTo('patients')}
          >
            <Users size={18} /> Hồ Sơ Bệnh Nhân
          </button>

          <button 
            className={`nav-item ${activeTab === 'recalls' ? 'active' : ''}`}
            aria-current={activeTab === 'recalls' ? 'page' : undefined}
            style={{ 
              justifyContent: 'flex-start', 
              gap: '12px',
              backgroundColor: activeTab === 'recalls' ? 'rgba(13, 148, 136, 0.1)' : 'transparent',
              color: activeTab === 'recalls' ? 'var(--primary)' : 'var(--text-secondary)',
              border: 'none'
            }}
            onClick={() => navigateTo('recalls')}
          >
            <CalendarCheck size={18} /> Lịch Tái Khám
          </button>

          <button
            className={`nav-item ${activeTab === 'evidence' ? 'active' : ''}`}
            aria-current={activeTab === 'evidence' ? 'page' : undefined}
            style={{
              justifyContent: 'flex-start',
              gap: '12px',
              backgroundColor: activeTab === 'evidence' ? 'rgba(13, 148, 136, 0.1)' : 'transparent',
              color: activeTab === 'evidence' ? 'var(--primary)' : 'var(--text-secondary)',
              border: 'none'
            }}
            onClick={() => navigateTo('evidence')}
          >
            <BookOpenCheck size={18} /> Pháp Lý & Khoa Học
          </button>
        </nav>

        {/* User profile section & Logout */}
        <div className="sidebar-profile">
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
      <main className="main-content" id="main-content" tabIndex="-1">
        {activeTab === 'dashboard' && (
          <div className="page-stack animated-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
            <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <h1 style={{ fontSize: '32px', marginBottom: '8px' }}>Hiệu năng mô hình phân loại DR</h1>
                <p style={{ color: 'var(--text-secondary)' }}>
                  Kết quả trên tập test sau khi checkpoint grading tốt nhất được huấn luyện semi-supervised.
                </p>
              </div>

            </div>

            <div className="metric-grid model-metric-grid">
              <div className="card metric-card model-metric-card">
                <div className="metric-label"><Target size={17} aria-hidden="true" /> Accuracy</div>
                <strong className="metric-value">{formatPercent(SEMI_SUPERVISED_TEST_RESULT.accuracy)}</strong>
                <span className="metric-caption">Tỉ lệ dự đoán đúng chính xác grade</span>
              </div>
              <div className="card metric-card model-metric-card">
                <div className="metric-label"><Award size={17} aria-hidden="true" /> QWK</div>
                <strong className="metric-value metric-value-primary">{formatMetric(SEMI_SUPERVISED_TEST_RESULT.qwk)}</strong>
                <span className="metric-caption">Mức đồng thuận có xét khoảng cách grade</span>
              </div>
              <div className="card metric-card model-metric-card">
                <div className="metric-label"><Gauge size={17} aria-hidden="true" /> Macro F1</div>
                <strong className="metric-value">{formatMetric(SEMI_SUPERVISED_TEST_RESULT.macro_f1)}</strong>
                <span className="metric-caption">F1 trung bình đồng đều giữa 5 lớp</span>
              </div>
              <div className="card metric-card model-metric-card">
                <div className="metric-label"><Activity size={17} aria-hidden="true" /> Balanced Accuracy</div>
                <strong className="metric-value">{formatMetric(SEMI_SUPERVISED_TEST_RESULT.balanced_accuracy)}</strong>
                <span className="metric-caption">Độ chính xác cân bằng theo lớp</span>
              </div>
            </div>

            <div className="model-analysis-grid">
              {/* Cột trái: Test result + Sensitivity */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <section className="card model-detail-card" aria-labelledby="loss-heading">
                  <div className="section-heading-row">
                    <div>
                      <h3 id="loss-heading">Kết quả đánh giá</h3>
                    </div>
                    <span className="status-chip">Semi-supervised</span>
                  </div>
                  <div className="loss-grid">
                    <div>
                      <span>Test loss</span>
                      <strong>{formatMetric(SEMI_SUPERVISED_TEST_RESULT.loss)}</strong>
                    </div>
                    <div>
                      <span>Tập đánh giá</span>
                      <strong>Test set</strong>
                    </div>
                    <div>
                      <span>Giai đoạn</span>
                      <strong>Sau semi</strong>
                    </div>
                  </div>
                  <p className="model-note">
                    Đây là số liệu từ lần test lại sau khi mô hình grading tốt nhất tiếp tục được huấn luyện semi-supervised.
                  </p>
                </section>

                <section className="card recall-card" aria-labelledby="recall-heading">
                  <div className="section-heading-row">
                    <div>
                      <span className="section-eyebrow">Sensitivity theo lớp</span>
                      <h3 id="recall-heading">Recall trên test set sau semi</h3>
                    </div>
                    <span className="status-chip">5 mức ICDR</span>
                  </div>
                  <div className="recall-list">
                    {Object.entries(SEMI_SUPERVISED_TEST_RESULT.per_class_recall).map(([label, recall], index) => (
                      <div className="recall-row" key={label}>
                        <span className="recall-label">{RECALL_LABELS[label]}</span>
                        <div
                          className="recall-track"
                          role="progressbar"
                          aria-label={`Recall ${RECALL_LABELS[label]}`}
                          aria-valuemin="0"
                          aria-valuemax="100"
                          aria-valuenow={(recall * 100).toFixed(3)}
                        >
                          <div
                            className="recall-fill"
                            style={{
                              width: `${recall * 100}%`,
                              backgroundColor: colors.drGrades[index].color,
                            }}
                          />
                        </div>
                        <strong>{formatPercent(recall)}</strong>
                      </div>
                    ))}
                  </div>
                </section>
              </div>

              {/* Cột phải: Ước lượng độ lệch grade */}
              <section className="card rmse-card" aria-labelledby="rmse-heading">
                <div className="section-heading-row">
                  <div>
                    <span className="section-eyebrow">Minh họa từ kết quả test semi</span>
                    <h3 id="rmse-heading">Ví dụ với 1.000 bệnh nhân</h3>
                  </div>
                  <Calculator size={22} color="var(--primary)" aria-hidden="true" />
                </div>

                <p className="example-intro">
                  Với Accuracy <strong>{formatPercent(GRADE_ERROR_EXAMPLE.accuracy)}</strong> và QWK <strong>{formatMetric(GRADE_ERROR_EXAMPLE.qwk)}</strong>:
                </p>

                <div className="grade-error-summary">
                  <div>
                    <span>Dự đoán đúng</span>
                    <strong>{GRADE_ERROR_EXAMPLE_CORRECT.toFixed(0)} ca</strong>
                  </div>
                  <div>
                    <span>Dự đoán sai</span>
                    <strong>{GRADE_ERROR_EXAMPLE_WRONG.toFixed(0)} ca</strong>
                  </div>
                  <div>
                    <span>RMSE trên ca sai</span>
                    <strong>{formatDecimal(GRADE_ERROR_WRONG_RMSE)} grade</strong>
                  </div>
                  <div>
                    <span>Khoảng MAE có thể có</span>
                    <strong>{formatDecimal(GRADE_ERROR_MAE_MIN)}–{formatDecimal(GRADE_ERROR_MAE_MAX)} grade</strong>
                  </div>
                </div>

                <div className="interpretation-callout">
                  <span>Diễn giải ngắn</span>
                  Trong {formatInteger(GRADE_ERROR_EXAMPLE_WRONG)} ca sai, độ lệch tuyệt đối trung bình chỉ có thể ước lượng trong khoảng <strong>{formatDecimal(GRADE_ERROR_MAE_MIN)}–{formatDecimal(GRADE_ERROR_MAE_MAX)} grade</strong>. Nếu lấy giá trị giữa khoảng để minh họa thì khoảng <strong>{formatDecimal(GRADE_ERROR_MAE_MIDPOINT)} grade/ca sai</strong>.
                </div>

                <details className="calculation-details">
                  <summary>
                    <span>Xem công thức và cách tính chi tiết</span>
                    <ChevronDown size={18} aria-hidden="true" />
                  </summary>
                  <div className="calculation-content">
                    <div className="calculation-assumption" role="note">
                      <strong>Giả định:</strong> 5 grade 0–4 có phân bố thật và dự đoán cân bằng, vì vậy sai lệch kỳ vọng D<sub>e</sub> = 0.250. Không có confusion matrix nên không thể tính MAE chính xác.
                    </div>

                    <ol className="calculation-steps">
                      <li>
                        <strong>Đếm số ca đúng và sai</strong>
                        <div className="equation">Số ca đúng = {formatInteger(GRADE_ERROR_EXAMPLE.samples)} × {formatMetric(GRADE_ERROR_EXAMPLE.accuracy)} = {formatInteger(GRADE_ERROR_EXAMPLE_CORRECT)} ca</div>
                        <div className="equation">Số ca sai = {formatInteger(GRADE_ERROR_EXAMPLE.samples)} × (1 − {formatMetric(GRADE_ERROR_EXAMPLE.accuracy)}) = {formatInteger(GRADE_ERROR_EXAMPLE_WRONG)} ca</div>
                      </li>
                      <li>
                        <strong>Tính sai lệch quan sát từ QWK</strong>
                        <div className="equation">QWK = 1 − D<sub>o</sub> / D<sub>e</sub></div>
                        <div className="equation">D<sub>o</sub> = (1 − {formatMetric(GRADE_ERROR_EXAMPLE.qwk)}) × 0.250 = {GRADE_ERROR_OBSERVED_DISAGREEMENT.toFixed(4)}</div>
                      </li>
                      <li>
                        <strong>Đổi về tổng bình phương khoảng cách grade</strong>
                        <div className="equation">SSE = {formatInteger(GRADE_ERROR_EXAMPLE.samples)} × (5 − 1)² × {GRADE_ERROR_OBSERVED_DISAGREEMENT.toFixed(4)} = {formatDecimal(GRADE_ERROR_SQUARED_SUM)}</div>
                      </li>
                      <li>
                        <strong>Tính MSE và RMSE trên {formatInteger(GRADE_ERROR_EXAMPLE_WRONG)} ca sai</strong>
                        <div className="equation">MSE<sub>sai</sub> = {formatDecimal(GRADE_ERROR_SQUARED_SUM)} / {formatInteger(GRADE_ERROR_EXAMPLE_WRONG)} = {formatDecimal(GRADE_ERROR_WRONG_MSE)}</div>
                        <div className="equation equation-result">RMSE<sub>sai</sub> = √{formatDecimal(GRADE_ERROR_WRONG_MSE)} ≈ {formatDecimal(GRADE_ERROR_WRONG_RMSE)} grade</div>
                      </li>
                    </ol>

                    <div className="weight-table-wrap" role="region" aria-label="Bảng trọng số sai lệch QWK" tabIndex="0">
                      <table className="weight-table">
                        <thead>
                          <tr>
                            <th>Khoảng cách grade</th>
                            <th>Ví dụ</th>
                            <th>Trọng số phạt</th>
                          </tr>
                        </thead>
                        <tbody>
                          <tr><td>0</td><td>0 → 0</td><td>0.000</td></tr>
                          <tr><td>1</td><td>0 → 1</td><td>0.063</td></tr>
                          <tr><td>2</td><td>0 → 2</td><td>0.250</td></tr>
                          <tr><td>3</td><td>0 → 3</td><td>0.563</td></tr>
                          <tr><td>4</td><td>0 → 4</td><td>1.000</td></tr>
                        </tbody>
                      </table>
                    </div>

                    <p className="model-note">
                      Accuracy coi mọi dự đoán sai như nhau. QWK phạt lỗi lệch xa mạnh hơn theo bình phương khoảng cách. Muốn biết MAE thật sự, cần tính trực tiếp mean(|y<sub>true</sub> − y<sub>pred</sub>|) từ toàn bộ dự đoán.
                    </p>
                  </div>
                </details>
              </section>
            </div>
          </div>
        )}

        {activeTab === 'screening' && (
          <div className="page-stack animated-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
            <div>
              <h1 style={{ fontSize: '32px', marginBottom: '8px' }}>Phòng Khám Sàng Lọc AI</h1>
              <p style={{ color: 'var(--text-secondary)' }}>
                Phân loại DR từng mắt bằng mô hình nội bộ, sau đó Gemini tổng hợp hồ sơ thành báo cáo dự thảo để bác sĩ duyệt.
              </p>
            </div>

            {/* Patient selector for screening */}
            <div className="card patient-context" style={{ display: 'flex', alignItems: 'center', gap: '15px', padding: '16px 20px' }}>
              <span style={{ fontWeight: '600' }}>Bệnh nhân khám:</span>
              {screeningPatient ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
                  <span className="badge" style={{ backgroundColor: 'rgba(13, 148, 136, 0.1)', color: 'var(--primary)', fontWeight: 'bold', fontSize: '13px', borderRadius: '4px' }}>
                    {screeningPatient.patient_code} - {screeningPatient.full_name}
                  </span>
                  <span className="status-chip">{screeningPatient.diabetes_type || 'Chưa rõ loại ĐTĐ'}</span>
                  <span className="status-chip">{screeningPatient.diabetes_duration_years ?? '—'} năm</span>
                  <span className="status-chip">HbA1c {screeningPatient.latest_hba1c ?? '—'}%</span>
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

            <section className="card grading-model-selector" aria-labelledby="grading-model-label">
              <div className="grading-model-selector-copy">
                <span className="grading-model-icon" aria-hidden="true">
                  <Gauge size={22} />
                </span>
                <div>
                  <label id="grading-model-label" htmlFor="grading-model">
                    Model nhận diện DR
                  </label>
                  <p>{selectedGradingModelInfo.description}</p>
                </div>
              </div>
              <div className="grading-model-control">
                <select
                  id="grading-model"
                  value={selectedGradingModel}
                  disabled={isAnalyzing}
                  onChange={(event) => {
                    setSelectedGradingModel(event.target.value);
                    setAnalysisResult(null);
                    setCurrentScreeningDetail(null);
                    setSummarySource('ai');
                  }}
                >
                  {gradingModels.map((model) => (
                    <option key={model.id} value={model.id} disabled={!model.ready}>
                      {model.label} · {model.checkpoint}{model.ready ? '' : ' (không tìm thấy tệp)'}
                    </option>
                  ))}
                </select>
                <span
                  className={`model-readiness ${isSelectedGradingModelReady ? 'is-ready' : 'is-missing'}`}
                  role="status"
                >
                  {isSelectedGradingModelReady ? 'Checkpoint sẵn sàng' : 'Checkpoint chưa sẵn sàng'}
                </span>
              </div>
            </section>

            <div className="screening-grid" style={{ display: 'grid', gridTemplateColumns: isUploadCollapsed && (analysisResult || isAnalyzing) ? '250px 1fr' : '1fr 1fr', gap: '30px' }}>
              {/* Cột Upload */}
              {isUploadCollapsed && (analysisResult || isAnalyzing) ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
                  <div className="card" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '15px', border: '1px solid var(--border-light)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: 'var(--primary)' }}>
                      <UploadCloud size={20} />
                      <strong style={{ fontSize: '14px' }}>Ảnh đã chọn</strong>
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '13px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '6px 10px', background: 'var(--background)', borderRadius: '6px' }}>
                        <span>Mắt trái:</span>
                        <strong style={{ color: selectedFiles.left_fundus_image ? 'var(--primary)' : 'var(--text-muted)' }}>
                          {selectedFiles.left_fundus_image ? '✓ Đã tải' : 'Trống'}
                        </strong>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '6px 10px', background: 'var(--background)', borderRadius: '6px' }}>
                        <span>Mắt phải:</span>
                        <strong style={{ color: selectedFiles.right_fundus_image ? 'var(--primary)' : 'var(--text-muted)' }}>
                          {selectedFiles.right_fundus_image ? '✓ Đã tải' : 'Trống'}
                        </strong>
                      </div>
                    </div>
                    <button
                      type="button"
                      className="btn btn-secondary"
                      style={{ fontSize: '12px', padding: '8px', width: '100%', justifyContent: 'center' }}
                      onClick={() => setIsUploadCollapsed(false)}
                    >
                      Thay đổi ảnh võng mạc
                    </button>
                  </div>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                  <div className="card upload-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '300px', borderStyle: 'dashed', borderWidth: '2px', borderColor: 'var(--primary)' }}>
                    <UploadCloud size={48} color="var(--primary)" style={{ marginBottom: '16px' }} />
                    <p style={{ fontWeight: '500', marginBottom: '8px' }}>Chọn ảnh cho một hoặc hai mắt</p>
                    <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '16px', textAlign: 'center' }}>
                      Mỗi mắt chỉ cần 1 ảnh fundus trung tâm rõ võng mạc. Có thể phân tích một mắt hoặc cả hai mắt trong cùng lượt khám.
                    </p>
                    <div className="file-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', width: '100%', gap: '10px' }}>
                      {[
                        ['left_fundus_image', 'Ảnh fundus mắt trái'],
                        ['right_fundus_image', 'Ảnh fundus mắt phải'],
                      ].map(([key, label]) => (
                        <label key={key} className="btn btn-secondary" style={{ cursor: 'pointer', fontSize: '12px' }}>
                          {selectedFiles[key] ? `✓ ${label}` : label}
                          <input type="file" accept="image/png,image/jpeg" onChange={(e) => handleFileChange(key, e)} style={{ display: 'none' }} />
                        </label>
                      ))}
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: '15px' }}>
                    <button
                      className="btn btn-primary"
                      style={{ flex: 1, padding: '14px' }}
                      disabled={!hasScreeningImage || isAnalyzing || !screeningPatient || !isSelectedGradingModelReady}
                      onClick={startAnalysis}
                    >
                      {isAnalyzing ? 'AI đang phân tích...' : 'Bắt đầu phân tích AI'}
                    </button>
                    {(analysisResult || isAnalyzing) && (
                      <button
                        type="button"
                        className="btn btn-secondary"
                        style={{ padding: '0 15px' }}
                        onClick={() => setIsUploadCollapsed(true)}
                      >
                        Thu nhỏ
                      </button>
                    )}
                  </div>
                </div>
              )}

              {/* Cột hiển thị Kết Quả AI */}
              <div className="card result-panel" aria-live="polite" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', minHeight: '300px' }}>
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
                    <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>Tiền xử lý → Phân loại DR → Phân đoạn tổn thương → Tổng hợp báo cáo</p>
                    <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
                  </div>
                )}

                {analysisResult && (
                  <div className="animated-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                    <h3 style={{ borderBottom: '1px solid var(--border-light)', paddingBottom: '10px' }}>Kết quả phân tích từ AI</h3>
                    
                    {[['left_eye', 'Mắt trái'], ['right_eye', 'Mắt phải']].map(([key, label]) => {
                      const eye = analysisResult[key];
                      if (!eye) return null;
                      const grade = eye.ai_result.dr_grade;
                      return <div key={key} style={{ padding: '16px', backgroundColor: colors.drGrades[grade].bg, borderRadius: '8px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <strong>{label}: Grade {grade} – {colors.drGrades[grade].label}</strong>
                          <span className="badge" style={{ backgroundColor: colors.drGrades[grade].color, color: '#fff' }}>{Math.round(eye.ai_result.confidence * 100)}%</span>
                        </div>
                        <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Ưu tiên rà soát: <strong>{eye.review_priority === 'prompt' ? 'Cao (Cần rà soát sớm)' : eye.review_priority}</strong> · Hẹn: <strong>{eye.follow_up_window}</strong></div>
                        <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{eye.referral}</div>

                        {/* Image Preview & AI Overlay Toggle */}
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '6px' }}>
                          <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
                            <button 
                              className="btn btn-secondary" 
                              style={{ padding: '6px 12px', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '6px', border: '1px solid var(--border-light)' }}
                              onClick={() => setShowMasks(prev => ({ ...prev, [key]: !prev[key] }))}
                            >
                              <Eye size={12} /> {showMasks[key] ? 'Xem ảnh võng mạc gốc' : 'Xem bản đồ tổn thương AI'}
                            </button>
                            <button 
                              className="btn btn-secondary" 
                              style={{ padding: '6px 12px', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '6px', border: '1px solid var(--border-light)', color: 'var(--primary)', fontWeight: '600' }}
                              onClick={() => setLightboxImage({
                                src: showMasks[key] && eye.segmentation?.lesion_mask_url ? eye.segmentation.lesion_mask_url : eye.image_url,
                                originalSrc: eye.image_url,
                                maskSrc: eye.segmentation?.lesion_mask_url,
                                title: `${label} - Grade ${grade} (${colors.drGrades[grade].label})`,
                                isMask: showMasks[key]
                              })}
                            >
                              <Maximize2 size={12} /> Phóng to xem siêu rõ
                            </button>
                          </div>
                          
                          <div 
                            onClick={() => setLightboxImage({
                              src: showMasks[key] && eye.segmentation?.lesion_mask_url ? eye.segmentation.lesion_mask_url : eye.image_url,
                              originalSrc: eye.image_url,
                              maskSrc: eye.segmentation?.lesion_mask_url,
                              title: `${label} - Grade ${grade} (${colors.drGrades[grade].label})`,
                              isMask: showMasks[key]
                            })}
                            style={{ position: 'relative', width: '100%', height: '340px', borderRadius: '8px', overflow: 'hidden', border: '1px solid var(--border-light)', backgroundColor: '#000', cursor: 'zoom-in' }}
                            title="Nhấp để phóng to xem ảnh toàn màn hình"
                          >
                            <SecureImage 
                              src={showMasks[key] && eye.segmentation?.lesion_mask_url ? eye.segmentation.lesion_mask_url : eye.image_url} 
                              alt={`${label} image`}
                              style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                            />
                            <div style={{ position: 'absolute', bottom: '8px', right: '8px', backgroundColor: 'rgba(0,0,0,0.7)', color: '#fff', padding: '4px 10px', borderRadius: '6px', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '5px', backdropFilter: 'blur(4px)', fontWeight: '500' }}>
                              <ZoomIn size={12} /> Nhấp để phóng to
                            </div>
                          </div>
                        </div>

                        {/* Lesion Statistics Table */}
                        {eye.segmentation?.lesions && eye.segmentation.lesions.length > 0 && (
                          <div style={{ marginTop: '6px', fontSize: '12px', backgroundColor: 'rgba(255,255,255,0.6)', padding: '10px', borderRadius: '6px', border: '1px solid rgba(0,0,0,0.05)' }}>
                            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                              <thead>
                                <tr style={{ borderBottom: '1px solid rgba(0,0,0,0.1)', color: 'var(--text-muted)' }}>
                                  <th style={{ paddingBottom: '6px', fontWeight: '600' }}>Tổn thương</th>
                                  <th style={{ paddingBottom: '6px', fontWeight: '600', textAlign: 'center' }}>Trạng thái</th>
                                  <th style={{ paddingBottom: '6px', fontWeight: '600', textAlign: 'right' }}>Diện tích (%)</th>
                                </tr>
                              </thead>
                              <tbody>
                                {eye.segmentation.lesions.map(lesion => (
                                  <tr key={lesion.key} style={{ borderBottom: '1px solid rgba(0,0,0,0.04)' }}>
                                    <td style={{ padding: '6px 0', fontWeight: '500' }}>{lesion.label}</td>
                                    <td style={{ padding: '6px 0', textAlign: 'center' }}>
                                      {lesion.detected ? (
                                        <span className="badge" style={{ backgroundColor: 'rgba(239, 68, 68, 0.1)', color: '#EF4444', fontSize: '11px', padding: '2px 8px' }}>Phát hiện</span>
                                      ) : (
                                        <span className="badge" style={{ backgroundColor: 'rgba(16, 185, 129, 0.1)', color: '#10B981', fontSize: '11px', padding: '2px 8px' }}>Không</span>
                                      )}
                                    </td>
                                    <td style={{ padding: '6px 0', textAlign: 'right', fontWeight: 'bold', fontFamily: 'monospace' }}>
                                      {lesion.detected ? `${(lesion.area_pct * 100).toFixed(4)}%` : '0.0000%'}
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        )}
                      </div>;
                    })}
                    {selectedClinicalSummary && (
                      <section style={{ padding: '14px', border: '1px solid var(--border-light)', borderRadius: '10px', background: 'var(--surface)' }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '10px', marginBottom: '10px', flexWrap: 'wrap' }}>
                          <strong>Phân tích hồ sơ lâm sàng</strong>
                          <span className="status-chip">
                            {summarySource === 'ai'
                              ? `Gemini · ${analysisResult.clinical_summary.model}`
                              : 'Quy tắc hệ thống · v1'}
                          </span>
                        </div>
                        <div className="summary-source-switch" role="group" aria-label="Chọn nguồn phân tích hồ sơ">
                          <button
                            type="button"
                            className={`summary-source-button ${summarySource === 'ai' ? 'active' : ''}`}
                            aria-pressed={summarySource === 'ai'}
                            disabled={analysisResult.clinical_summary.status !== 'generated'}
                            onClick={() => setSummarySource('ai')}
                          >
                            Gemini AI
                          </button>
                          <button
                            type="button"
                            className={`summary-source-button ${summarySource === 'rules' ? 'active' : ''}`}
                            aria-pressed={summarySource === 'rules'}
                            onClick={() => setSummarySource('rules')}
                          >
                            Quy tắc hệ thống
                          </button>
                        </div>
                        <p className="summary-source-note">
                          {summarySource === 'ai'
                            ? 'Gemini diễn giải hồ sơ; thời gian tái khám được giữ nguyên theo quy tắc hệ thống.'
                            : 'Bản phân tích xác định từ logic lâm sàng có sẵn trong backend.'}
                        </p>
                        <p style={{ fontSize: '13px', lineHeight: 1.65 }}>{selectedClinicalSummary.overview}</p>
                        <article className="diagnostic-support-card">
                          <h4>Nhận định hỗ trợ chẩn đoán DR</h4>
                          <p>{selectedClinicalSummary.diagnostic_impression}</p>
                          {selectedClinicalSummary.diagnostic_basis?.length > 0 && (
                            <div className="diagnostic-basis">
                              <strong>Cơ sở nhận định từ ảnh và mô hình</strong>
                              <ul>
                                {selectedClinicalSummary.diagnostic_basis.map((item, index) => <li key={index}>{item}</li>)}
                              </ul>
                            </div>
                          )}
                          {selectedClinicalSummary.diagnostic_limitations?.length > 0 && (
                            <details className="diagnostic-limitations">
                              <summary>Giới hạn của nhận định</summary>
                              <ul>
                                {selectedClinicalSummary.diagnostic_limitations.map((item, index) => <li key={index}>{item}</li>)}
                              </ul>
                            </details>
                          )}
                        </article>
                        {selectedClinicalSummary.key_findings.length > 0 && (
                          <ul style={{ margin: '10px 0 0', paddingLeft: '18px', fontSize: '13px', lineHeight: 1.65 }}>
                            {selectedClinicalSummary.key_findings.map((item, index) => <li key={index}>{item}</li>)}
                          </ul>
                        )}
                        <div className="recall-window" style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap' }}>
                          <div>
                            <strong>Gợi ý thời gian tái khám:</strong> {selectedClinicalSummary.follow_up}
                            <p style={{ marginTop: '4px', color: 'var(--text-muted)', fontSize: '12px' }}>
                              Đây là gợi ý từ hệ thống; lịch chính thức được bác sĩ chọn trong bước duyệt bên dưới.
                            </p>
                          </div>
                        </div>

                        {/* Quick Recall Modal */}
                        {isQuickRecallOpen && (
                          <div style={{
                            position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.6)',
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            zIndex: 2000, backdropFilter: 'blur(4px)'
                          }}>
                            <div className="card" role="dialog" aria-modal="true" aria-labelledby="quick-recall-title" style={{
                              width: '100%', maxWidth: '420px', padding: '28px',
                              display: 'flex', flexDirection: 'column', gap: '18px',
                              animation: 'fadeIn 0.2s ease'
                            }}>
                              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <h3 id="quick-recall-title" style={{ fontSize: '17px', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--primary)' }}>
                                  <CalendarPlus size={18} /> Lên lịch tái khám nhanh
                                </h3>
                                <button type="button" aria-label="Đóng hộp thoại đặt lịch tái khám" onClick={() => setIsQuickRecallOpen(false)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}>
                                  <X size={20} />
                                </button>
                              </div>

                              <div style={{ fontSize: '13px', color: 'var(--text-secondary)', padding: '10px', background: 'var(--background)', borderRadius: '8px' }}>
                                <strong>Bệnh nhân:</strong> {screeningPatient?.full_name} ({screeningPatient?.patient_code})
                              </div>

                              <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                                  <label htmlFor="recall-months" style={{ fontSize: '13px', fontWeight: '500' }}>Khoảng thời gian tái khám</label>
                                  <div className="recall-month-control">
                                    <button
                                      type="button"
                                      aria-label="Giảm khoảng thời gian tái khám"
                                      onClick={() => stepRecallMonths(-1)}
                                      disabled={Number(quickRecallForm.recall_in_months) === RECALL_MONTH_OPTIONS[0]}
                                    >
                                      <Minus size={16} aria-hidden="true" />
                                    </button>
                                    <select
                                      id="recall-months"
                                      autoFocus
                                      value={quickRecallForm.recall_in_months}
                                      onChange={(event) => setQuickRecallForm((current) => ({
                                        ...current,
                                        recall_in_months: Number(event.target.value),
                                      }))}
                                    >
                                      {RECALL_MONTH_OPTIONS.map((months) => (
                                        <option key={months} value={months}>{months} tháng</option>
                                      ))}
                                    </select>
                                    <button
                                      type="button"
                                      aria-label="Tăng khoảng thời gian tái khám"
                                      onClick={() => stepRecallMonths(1)}
                                      disabled={Number(quickRecallForm.recall_in_months) === RECALL_MONTH_OPTIONS.at(-1)}
                                    >
                                      <Plus size={16} aria-hidden="true" />
                                    </button>
                                  </div>
                                  <p className="recall-date-preview">
                                    Ngày dự kiến: <strong>{new Date(`${quickRecallDate}T00:00:00`).toLocaleDateString('vi-VN', { dateStyle: 'long' })}</strong>
                                  </p>
                                </div>

                                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                                  <label style={{ fontSize: '13px', fontWeight: '500' }}>Phân tầng nguy cơ</label>
                                  <select
                                    value={quickRecallForm.risk_stratification}
                                    onChange={e => setQuickRecallForm(f => ({ ...f, risk_stratification: e.target.value }))}
                                    style={{ padding: '9px 12px', borderRadius: '8px', border: '1px solid var(--border-light)', backgroundColor: 'var(--surface)', color: 'var(--text-primary)', fontSize: '14px', outline: 'none' }}
                                  >
                                    <option value="Low">Thấp</option>
                                    <option value="Medium">Trung bình</option>
                                    <option value="High">Cao</option>
                                    <option value="Urgent">Khẩn cấp</option>
                                  </select>
                                </div>

                                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                                  <label style={{ fontSize: '13px', fontWeight: '500' }}>Khuyến nghị / ghi chú</label>
                                  <textarea
                                    value={quickRecallForm.recommendation}
                                    onChange={e => setQuickRecallForm(f => ({ ...f, recommendation: e.target.value }))}
                                    rows={3}
                                    style={{ padding: '9px 12px', borderRadius: '8px', border: '1px solid var(--border-light)', backgroundColor: 'var(--surface)', color: 'var(--text-primary)', fontSize: '13px', resize: 'vertical', outline: 'none', lineHeight: 1.6 }}
                                    placeholder="Nhập khuyến nghị hoặc để trống..."
                                  />
                                </div>
                              </div>

                              <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
                                <button type="button" className="btn btn-secondary" onClick={() => setIsQuickRecallOpen(false)}>Huỷ</button>
                                <button
                                  type="button"
                                  className="btn btn-primary"
                                  style={{ gap: '8px' }}
                                  disabled={isSubmittingRecall}
                                  onClick={async () => {
                                    if (!screeningPatient) return;
                                    setIsSubmittingRecall(true);
                                    try {
                                      await api.createQuickRecall(screeningPatient.id, {
                                        recall_date: quickRecallDate,
                                        risk_stratification: quickRecallForm.risk_stratification,
                                        recommendation: quickRecallForm.recommendation,
                                        screening_id: analysisResult?.screening_id || null,
                                      });
                                      setIsQuickRecallOpen(false);
                                      dialog.showSuccess(
                                        `Đã lên lịch tái khám cho ${screeningPatient.full_name} vào ngày ${new Date(`${quickRecallDate}T00:00:00`).toLocaleDateString('vi-VN')}.`,
                                        'Đã lên lịch tái khám',
                                      );
                                    } catch (err) {
                                      dialog.showError(err.message, 'Không thể lên lịch tái khám');
                                    } finally {
                                      setIsSubmittingRecall(false);
                                    }
                                  }}
                                >
                                  <CalendarPlus size={15} />
                                  {isSubmittingRecall ? 'Đang lưu...' : 'Xác nhận lên lịch'}
                                </button>
                              </div>
                            </div>
                          </div>
                        )}
                        <details style={{ marginTop: '10px', fontSize: '13px' }}>
                          <summary style={{ cursor: 'pointer', fontWeight: 600 }}>Xem yếu tố nguy cơ và hành động đề xuất</summary>
                          <div style={{ marginTop: '8px', display: 'grid', gap: '10px' }}>
                            <div>
                              <strong>Yếu tố nguy cơ</strong>
                              {selectedClinicalSummary.risk_factors.length > 0 ? (
                                <ul style={{ margin: '6px 0 0', paddingLeft: '18px', lineHeight: 1.6 }}>
                                  {selectedClinicalSummary.risk_factors.map((item, index) => <li key={index}>{item}</li>)}
                                </ul>
                              ) : <p style={{ marginTop: '4px', color: 'var(--text-muted)' }}>Chưa ghi nhận thêm từ dữ liệu đã nhập.</p>}
                            </div>
                            <div>
                              <strong>Hành động đề xuất</strong>
                              <ul style={{ margin: '6px 0 0', paddingLeft: '18px', lineHeight: 1.6 }}>
                                {selectedClinicalSummary.recommended_actions.map((item, index) => <li key={index}>{item}</li>)}
                              </ul>
                            </div>
                          </div>
                        </details>
                      </section>
                    )}
                    {currentScreeningDetail && canReviewScreening && currentScreeningDetail.status === 'AI_Analyzed' && (
                      <DoctorReviewForm
                        detail={currentScreeningDetail}
                        onSubmit={submitCurrentScreeningReview}
                      />
                    )}
                    {currentScreeningDetail?.status === 'Reviewed' && (
                      <div className="screening-reviewed-notice" role="status">
                        <CheckCircle size={18} aria-hidden="true" />
                        <div>
                          <strong>Bác sĩ đã xác nhận kết quả</strong>
                          <p>Kết luận đã được lưu; bệnh nhân có thể xem chi tiết trong cổng bệnh nhân.</p>
                        </div>
                      </div>
                    )}
                    <div style={{ fontSize: '12px', padding: '10px', background: 'var(--background)', borderRadius: '6px' }}>
                      <strong>Dự thảo – bắt buộc bác sĩ xác nhận.</strong><br />{analysisResult.disclaimer}
                    </div>

                    <div style={{ display: 'flex', gap: '10px', marginTop: '10px' }}>
                      <button className="btn btn-primary" style={{ flex: 1, gap: '8px' }} onClick={() => {
                        if (analysisResult.status === 'Reviewed') {
                          dialog.showSuccess('Lần khám đã hoàn tất và bệnh nhân có thể xem kết quả.', 'Đã hoàn tất lần khám');
                        } else {
                          dialog.showInfo('Kết quả đã lưu ở trạng thái chờ bác sĩ duyệt.', 'Đã lưu lần khám');
                        }
                        setSelectedFiles({});
                        setAnalysisResult(null);
                        setCurrentScreeningDetail(null);
                      }}>
                        <CheckCircle size={16} /> {analysisResult.status === 'Reviewed' ? 'Hoàn tất lần khám' : 'Lưu để duyệt sau'}
                      </button>
                      <button className="btn btn-secondary" style={{ gap: '8px' }} onClick={() => window.print()}>
                        <FileText size={16} /> Xuất PDF
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'patients' && (
          <div className="page-stack animated-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
            <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <h1 style={{ fontSize: '32px', marginBottom: '8px' }}>Quản lý hồ sơ bệnh nhân</h1>
                <p style={{ color: 'var(--text-secondary)' }}>Danh sách bệnh nhân tiểu đường đăng ký khám sàng lọc.</p>
              </div>
              <div className="header-actions" style={{ display: 'flex', gap: '10px' }}>
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
              <div className="search-shell" style={{ display: 'flex', alignItems: 'center', backgroundColor: 'var(--surface)', border: '1px solid var(--border-light)', borderRadius: 'var(--radius-sm)', padding: '0 15px', flex: 1 }}>
                <Search size={18} color="var(--text-muted)" style={{ marginRight: '10px' }} />
                <input 
                  type="text" 
                  aria-label="Tìm kiếm bệnh nhân"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Tìm kiếm bệnh nhân theo tên, mã bệnh nhân hoặc số điện thoại..." 
                  style={{ border: 'none', background: 'transparent', outline: 'none', width: '100%', padding: '12px 0', color: 'var(--text-primary)' }}
                />
              </div>
            </div>

            {/* Patients Table */}
            <div className="card table-card" role="region" aria-label="Danh sách bệnh nhân" tabIndex="0" style={{ padding: 0, overflow: 'hidden' }}>
              <table className="data-table" style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                <thead>
                  <tr style={{ backgroundColor: 'var(--background)', borderBottom: '1px solid var(--border-light)' }}>
                    <th style={{ padding: '16px 24px', fontWeight: '600' }}>Mã bệnh nhân</th>
                    <th style={{ padding: '16px 24px', fontWeight: '600' }}>Họ và tên</th>
                    <th style={{ padding: '16px 24px', fontWeight: '600' }}>Giới tính</th>
                    <th style={{ padding: '16px 24px', fontWeight: '600' }}>Ngày sinh</th>
                    <th style={{ padding: '16px 24px', fontWeight: '600' }}>HbA1c</th>
                    <th style={{ padding: '16px 24px', fontWeight: '600' }}>Tài khoản</th>
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
                          <span className={`account-status ${patient.has_portal_account ? 'account-status-active' : 'account-status-missing'}`}>
                            {patient.has_portal_account ? patient.portal_username : 'Chưa có'}
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
                      <td colSpan="7" style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
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

        {activeTab === 'evidence' && (
          <ProjectEvidencePage />
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
        canReview={
          currentUser.role === 'admin'
          || (
            currentUser.role === 'doctor'
            && (currentUser.hospital_department || '').toLocaleLowerCase('vi-VN').includes('nhãn')
          )
        }
      />

      {/* Fullscreen Image Lightbox Modal */}
      {lightboxImage && (
        <div 
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.92)',
            zIndex: 99999,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '24px',
            backdropFilter: 'blur(8px)',
            animation: 'fadeIn 0.2s ease'
          }}
          onClick={() => setLightboxImage(null)}
        >
          {/* Lightbox Header */}
          <div 
            style={{
              width: '100%',
              maxWidth: '1300px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '16px',
              color: '#fff',
              zIndex: 100000
            }}
            onClick={e => e.stopPropagation()}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '14px', flexWrap: 'wrap' }}>
              <h3 style={{ margin: 0, fontSize: '18px', fontWeight: '600', color: '#fff' }}>
                {lightboxImage.title}
              </h3>
              {lightboxImage.maskSrc && (
                <span className="badge" style={{ backgroundColor: lightboxImage.isMask ? '#EF4444' : '#10B981', color: '#fff', fontSize: '12px', padding: '5px 12px', fontWeight: '600' }}>
                  {lightboxImage.isMask ? '🔴 Bản đồ tổn thương AI (MA / HE / EX)' : '👁️ Ảnh võng mạc gốc'}
                </span>
              )}
            </div>

            <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
              {lightboxImage.maskSrc && (
                <button
                  className="btn btn-secondary"
                  style={{ backgroundColor: 'rgba(255,255,255,0.18)', color: '#fff', border: '1px solid rgba(255,255,255,0.35)', padding: '8px 16px', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '8px', fontWeight: '600', borderRadius: '8px' }}
                  onClick={() => setLightboxImage(prev => ({
                    ...prev,
                    isMask: !prev.isMask,
                    src: !prev.isMask ? prev.maskSrc : prev.originalSrc
                  }))}
                >
                  <Eye size={15} /> {lightboxImage.isMask ? 'Chuyển sang Ảnh gốc' : 'Chuyển sang Mask AI'}
                </button>
              )}
              <button
                style={{ backgroundColor: 'rgba(255,255,255,0.25)', border: 'none', color: '#fff', borderRadius: '50%', width: '40px', height: '40px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'all 0.2s' }}
                onClick={() => setLightboxImage(null)}
                title="Đóng (ESC)"
              >
                <X size={24} />
              </button>
            </div>
          </div>

          {/* Lightbox Main Viewport */}
          <div 
            style={{
              position: 'relative',
              maxWidth: '92vw',
              maxHeight: '80vh',
              width: '100%',
              height: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              overflow: 'hidden',
              backgroundColor: '#050505',
              borderRadius: '12px',
              border: '1px solid rgba(255,255,255,0.1)',
              boxShadow: '0 25px 60px rgba(0,0,0,0.9)'
            }}
            onClick={e => e.stopPropagation()}
          >
            <SecureImage 
              src={lightboxImage.src} 
              alt={lightboxImage.title}
              style={{ maxWidth: '100%', maxHeight: '80vh', width: 'auto', height: 'auto', objectFit: 'contain', display: 'block' }}
            />
          </div>

          {/* Lightbox Sub-Caption & Tip */}
          <div style={{ marginTop: '14px', color: 'rgba(255,255,255,0.75)', fontSize: '13px', textAlign: 'center', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <ZoomIn size={14} /> Bác sĩ có thể nhấp <strong>Chuyển đổi Mask AI</strong> ở góc phải để đối chiếu trực tiếp tổn thương với ảnh võng mạc gốc.
          </div>
        </div>
      )}
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
        const results = await Promise.all(
          patients.map(p => api.getPatientRecalls(p.id).then(recalls => recalls.map(r => ({ ...r, patient: p }))))
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
    <div className="page-stack animated-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
      <div className="page-header">
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
        <div className="card table-card" role="region" aria-label="Lịch tái khám" tabIndex="0" style={{ padding: 0, overflow: 'hidden' }}>
          <table className="data-table" style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
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
                      {isOverdue && <div className="overdue-label"><AlertTriangle size={12} aria-hidden="true" /> Quá hạn</div>}
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
