import React, { useState } from 'react';
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
  AlertTriangle
} from 'lucide-react';
import { colors } from './theme/colors';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [selectedFile, setSelectedFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);

  // Dữ liệu bệnh nhân mẫu
  const mockPatients = [
    { id: 1, code: 'BN0001', name: 'Phạm Văn Đồng', age: 61, duration: '8.5 năm', hba1c: '7.2%' },
    { id: 2, code: 'BN0002', name: 'Lê Thị Mai', age: 48, duration: '4.0 năm', hba1c: '6.5%' },
    { id: 3, code: 'BN0003', name: 'Nguyễn Tiến Dũng', age: 74, duration: '15.0 năm', hba1c: '8.4%' },
  ];

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setAnalysisResult(null);
    }
  };

  const startAnalysis = () => {
    if (!selectedFile) return;
    setIsAnalyzing(true);
    setTimeout(() => {
      setIsAnalyzing(false);
      setAnalysisResult({
        leftEye: { grade: 2, confidence: 0.91 },
        rightEye: { grade: 0, confidence: 0.98 },
        lesions: {
          microaneurysm: 12,
          hemorrhage: 3,
          hardExudate: 0
        }
      });
    }, 2000);
  };

  return (
    <div className="app-container">
      {/* Sidebar điều hướng */}
      <aside className="sidebar">
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '40px' }}>
          <Activity size={28} color={colors.light.primary} />
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
            onClick={() => setActiveTab('screening')}
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
        </nav>

        <div style={{ borderTop: '1px solid var(--border-light)', paddingTop: '20px', fontSize: '13px', color: 'var(--text-muted)' }}>
          Bác sĩ đang trực:<br />
          <strong style={{ color: 'var(--text-primary)' }}>TS. BS. Nguyễn Văn An</strong>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="main-content">
        {activeTab === 'dashboard' && (
          <div className="animated-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
            <div>
              <h1 style={{ fontSize: '32px', marginBottom: '8px' }}>Tổng quan sàng lọc võng mạc</h1>
              <p style={{ color: 'var(--text-secondary)' }}>Báo cáo dịch tễ học và hiệu năng AI thời gian thực.</p>
            </div>

            {/* Stat Cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '20px' }}>
              <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <span style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>Tổng bệnh nhân sàng lọc</span>
                <span style={{ fontSize: '36px', fontWeight: 'bold' }}>1,248</span>
                <span style={{ fontSize: '12px', color: 'var(--primary)' }}>+12% tháng này</span>
              </div>
              <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <span style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>Phát hiện bệnh lý DR (>0)</span>
                <span style={{ fontSize: '36px', fontWeight: 'bold', color: '#F59E0B' }}>384</span>
                <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Chiếm 30.7% số ca khám</span>
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
              <p style={{ color: 'var(--text-secondary)' }}>Upload ảnh đáy mắt để nhận kết quả phân tích mức độ DR và bản đồ tổn thương từ AI.</p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px' }}>
              {/* Cột Upload và Cấu hình */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <div className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '300px', borderStyle: 'dashed', borderWidth: '2px', borderColor: 'var(--primary)' }}>
                  <UploadCloud size={48} color="var(--primary)" style={{ marginBottom: '16px' }} />
                  <p style={{ fontWeight: '500', marginBottom: '8px' }}>Kéo thả hoặc Click để chọn ảnh võng mạc</p>
                  <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '20px' }}>Hỗ trợ định dạng PNG, JPG, JPEG chất lượng cao</p>
                  <input type="file" accept="image/*" onChange={handleFileChange} style={{ display: 'none' }} id="file-uploader" />
                  <label htmlFor="file-uploader" className="btn btn-secondary" style={{ cursor: 'pointer' }}>Chọn ảnh</label>
                  
                  {selectedFile && (
                    <div style={{ marginTop: '20px', fontSize: '14px', color: 'var(--primary)', fontWeight: 'bold' }}>
                      Tệp đã chọn: {selectedFile.name}
                    </div>
                  )}
                </div>

                <div style={{ display: 'flex', gap: '15px' }}>
                  <button 
                    className="btn btn-primary" 
                    style={{ flex: 1, padding: '14px' }}
                    disabled={!selectedFile || isAnalyzing}
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
                    Chưa có kết quả. Vui lòng chọn ảnh võng mạc và nhấn phân tích.
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
                    
                    {/* Mắt trái */}
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px', backgroundColor: colors.drGrades[analysisResult.leftEye.grade].bg, borderRadius: '8px' }}>
                      <div>
                        <strong>Mắt trái (Left Eye):</strong>
                        <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>Mức độ chẩn đoán: {colors.drGrades[analysisResult.leftEye.grade].label}</div>
                      </div>
                      <span className="badge" style={{ backgroundColor: colors.drGrades[analysisResult.leftEye.grade].color, color: '#fff' }}>
                        Grade {analysisResult.leftEye.grade} ({Math.round(analysisResult.leftEye.confidence * 100)}%)
                      </span>
                    </div>

                    {/* Mắt phải */}
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px', backgroundColor: colors.drGrades[analysisResult.rightEye.grade].bg, borderRadius: '8px' }}>
                      <div>
                        <strong>Mắt phải (Right Eye):</strong>
                        <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>Mức độ chẩn đoán: {colors.drGrades[analysisResult.rightEye.grade].label}</div>
                      </div>
                      <span className="badge" style={{ backgroundColor: colors.drGrades[analysisResult.rightEye.grade].color, color: '#fff' }}>
                        Grade {analysisResult.rightEye.grade} ({Math.round(analysisResult.rightEye.confidence * 100)}%)
                      </span>
                    </div>

                    {/* Bản đồ tổn thương */}
                    <div style={{ marginTop: '10px' }}>
                      <h4 style={{ marginBottom: '10px', fontSize: '14px' }}>Chi tiết các tổn thương phát hiện (Mắt trái):</h4>
                      <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        <li style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px' }}>
                          <span>Vi phình mạch (Microaneurysm):</span>
                          <strong style={{ color: 'red' }}>{analysisResult.lesions.microaneurysm} điểm</strong>
                        </li>
                        <li style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px' }}>
                          <span>Xuất huyết (Hemorrhage):</span>
                          <strong style={{ color: 'red' }}>{analysisResult.lesions.hemorrhage} điểm</strong>
                        </li>
                        <li style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px' }}>
                          <span>Rỉ dịch cứng (Hard Exudate):</span>
                          <span>Không phát hiện</span>
                        </li>
                      </ul>
                    </div>

                    <div style={{ display: 'flex', gap: '10px', marginTop: '10px' }}>
                      <button className="btn btn-primary" style={{ flex: 1, gap: '8px' }}>
                        <CheckCircle size={16} /> Xác nhận kết quả
                      </button>
                      <button className="btn btn-secondary" style={{ gap: '8px' }}>
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
          <div className="animated-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <h1 style={{ fontSize: '32px', marginBottom: '8px' }}>Quản lý hồ sơ bệnh nhân</h1>
                <p style={{ color: 'var(--text-secondary)' }}>Danh sách bệnh nhân tiểu đường đăng ký khám sàng lọc.</p>
              </div>
              <button className="btn btn-primary">Thêm Bệnh Nhân Mới</button>
            </div>

            {/* Search bar */}
            <div style={{ display: 'flex', gap: '15px' }}>
              <div style={{ display: 'flex', alignItems: 'center', backgroundColor: 'var(--surface)', border: '1px solid var(--border-light)', borderRadius: 'var(--radius-sm)', padding: '0 15px', flex: 1 }}>
                <Search size={18} color="var(--text-muted)" style={{ marginRight: '10px' }} />
                <input 
                  type="text" 
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
                    <th style={{ padding: '16px 24px', fontWeight: '600' }}>Tuổi</th>
                    <th style={{ padding: '16px 24px', fontWeight: '600' }}>Thời gian mắc ĐTĐ</th>
                    <th style={{ padding: '16px 24px', fontWeight: '600' }}>HbA1c gần nhất</th>
                    <th style={{ padding: '16px 24px', fontWeight: '600' }}>Thao tác</th>
                  </tr>
                </thead>
                <tbody>
                  {mockPatients.map((patient) => (
                    <tr key={patient.id} style={{ borderBottom: '1px solid var(--border-light)' }}>
                      <td style={{ padding: '16px 24px', fontWeight: 'bold', color: 'var(--primary)' }}>{patient.code}</td>
                      <td style={{ padding: '16px 24px', fontWeight: '500' }}>{patient.name}</td>
                      <td style={{ padding: '16px 24px' }}>{patient.age}</td>
                      <td style={{ padding: '16px 24px' }}>{patient.duration}</td>
                      <td style={{ padding: '16px 24px' }}>
                        <span className="badge" style={{ backgroundColor: 'rgba(245, 158, 11, 0.1)', color: '#D97706' }}>
                          {patient.hba1c}
                        </span>
                      </td>
                      <td style={{ padding: '16px 24px', display: 'flex', gap: '10px' }}>
                        <button className="btn btn-secondary" style={{ padding: '6px 12px', fontSize: '13px' }}>Chi tiết</button>
                        <button 
                          className="btn btn-primary" 
                          style={{ padding: '6px 12px', fontSize: '13px' }}
                          onClick={() => setActiveTab('screening')}
                        >
                          Khám sàng lọc
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
