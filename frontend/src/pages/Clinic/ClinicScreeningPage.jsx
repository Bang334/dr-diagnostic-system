import React, { useState, useEffect } from 'react';
import { 
  UploadCloud, 
  ShieldAlert, 
  Eye, 
  Maximize2, 
  ZoomIn, 
  Gauge, 
  CalendarPlus, 
  X, 
  Minus, 
  Plus, 
  CheckCircle 
} from 'lucide-react';
import { colors } from '../../theme/colors';
import SecureImage from '../../components/SecureImage';
import { DoctorReviewForm } from '../../components/ScreeningDetailPanel';
import LightboxModal from '../../components/common/LightboxModal';
import { RECALL_MONTH_OPTIONS, addMonthsToToday } from '../../utils/dateHelpers';
import { DEFAULT_GRADING_MODELS } from '../../utils/clinicalFormatters';
import { api } from '../../services/api';
import { useAuth } from '../../contexts/AuthContext';
import { useAppDialog } from '../../components/AppDialogProvider';

export default function ClinicScreeningPage({ initialPatient, onClearPatient }) {
  const dialog = useAppDialog();
  const { currentUser } = useAuth();

  // Patients Data
  const [patients, setPatients] = useState([]);
  const [screeningPatient, setScreeningPatient] = useState(initialPatient || null);

  // AI Models & Input Files
  const [gradingModels, setGradingModels] = useState(DEFAULT_GRADING_MODELS);
  const [selectedGradingModel, setSelectedGradingModel] = useState('grading');
  const [selectedFiles, setSelectedFiles] = useState({});

  // Analysis State
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [showMasks, setShowMasks] = useState({ left_eye: false, right_eye: false });
  const [lightboxImage, setLightboxImage] = useState(null);
  const [currentScreeningDetail, setCurrentScreeningDetail] = useState(null);
  const [summarySource, setSummarySource] = useState('ai');
  const [isUploadCollapsed, setIsUploadCollapsed] = useState(false);

  // Quick Recall Modal State
  const [isQuickRecallOpen, setIsQuickRecallOpen] = useState(false);
  const [quickRecallForm, setQuickRecallForm] = useState({
    recall_in_months: 6,
    risk_stratification: 'Low',
    recommendation: ''
  });
  const [isSubmittingRecall, setIsSubmittingRecall] = useState(false);

  // Sync initial patient if changed from outside
  useEffect(() => {
    if (initialPatient) {
      setScreeningPatient(initialPatient);
      setSelectedFiles({});
      setAnalysisResult(null);
      setCurrentScreeningDetail(null);
      setIsUploadCollapsed(false);
    }
  }, [initialPatient]);

  // Fetch Patients
  useEffect(() => {
    api.getPatients()
      .then(data => {
        setPatients(data);
        if (!screeningPatient && !initialPatient && data.length > 0) {
          setScreeningPatient(data[0]);
        }
      })
      .catch(err => console.error('Error fetching patients for clinic:', err));
  }, []);

  // Load Models
  useEffect(() => {
    let cancelled = false;
    api.getScreeningModels()
      .then((data) => {
        if (cancelled || !Array.isArray(data.models)) return;
        setGradingModels(data.models);
        if (data.default_model_id) {
          setSelectedGradingModel(data.default_model_id);
        }
      })
      .catch((error) => console.error('Error fetching AI models:', error));
    return () => { cancelled = true; };
  }, []);

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
  const quickRecallDate = addMonthsToToday(quickRecallForm.recall_in_months);

  const canReviewScreening = currentUser?.role === 'admin'
    || (
      currentUser?.role === 'doctor'
      && (currentUser.hospital_department || '').toLocaleLowerCase('vi-VN').includes('nhãn')
    );

  const handleFileChange = (key, e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFiles(current => ({ ...current, [key]: e.target.files[0] }));
      setAnalysisResult(null);
      setCurrentScreeningDetail(null);
      setSummarySource('ai');
    }
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
        console.error('Could not load screening detail:', detailError);
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
    try {
      await api.reviewScreening(analysisResult.screening_id, payload);
      const detail = await api.getScreeningDetail(analysisResult.screening_id);
      setCurrentScreeningDetail(detail);
      setAnalysisResult((current) => ({ ...current, status: detail.status }));
      dialog.showSuccess(
        'Kết luận bác sĩ và kế hoạch tái khám đã được lưu. Bệnh nhân hiện có thể xem chi tiết lần khám.',
        'Đã duyệt kết quả thành công',
      );
    } catch (err) {
      dialog.showError(err.message, 'Lỗi duyệt kết quả');
    }
  };

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

  return (
    <div className="page-stack animated-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
      <div>
        <h1 style={{ fontSize: '32px', marginBottom: '8px', fontWeight: 'bold' }}>Phòng Khám Sàng Lọc AI</h1>
        <p style={{ color: 'var(--text-secondary)', margin: 0 }}>
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
            <button 
              className="btn btn-secondary" 
              style={{ padding: '4px 8px', fontSize: '12px' }} 
              onClick={() => {
                setScreeningPatient(null);
                onClearPatient && onClearPatient();
              }}
            >
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

      {/* Grading Model Selector */}
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

      {/* Screening Work Grid */}
      <div className="screening-grid" style={{ display: 'grid', gridTemplateColumns: isUploadCollapsed && (analysisResult || isAnalyzing) ? '250px 1fr' : '1fr 1fr', gap: '30px' }}>
        {/* Upload Column */}
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

        {/* Results Column */}
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
              <h3 style={{ borderBottom: '1px solid var(--border-light)', paddingBottom: '10px', margin: 0 }}>Kết quả phân tích từ AI</h3>
              
              {[['left_eye', 'Mắt trái'], ['right_eye', 'Mắt phải']].map(([key, label]) => {
                const eye = analysisResult[key];
                if (!eye) return null;
                const grade = eye.ai_result.dr_grade;
                return (
                  <div key={key} style={{ padding: '16px', backgroundColor: colors.drGrades[grade].bg, borderRadius: '8px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
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
                            isMask: showMasks[key],
                            onToggleMask: (nextMask) => {
                              setShowMasks(prev => ({ ...prev, [key]: nextMask }));
                              setLightboxImage(curr => ({
                                ...curr,
                                isMask: nextMask,
                                src: nextMask && eye.segmentation?.lesion_mask_url ? eye.segmentation.lesion_mask_url : eye.image_url
                              }));
                            }
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
                          isMask: showMasks[key],
                          onToggleMask: (nextMask) => {
                            setShowMasks(prev => ({ ...prev, [key]: nextMask }));
                            setLightboxImage(curr => ({
                              ...curr,
                              isMask: nextMask,
                              src: nextMask && eye.segmentation?.lesion_mask_url ? eye.segmentation.lesion_mask_url : eye.image_url
                            }));
                          }
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
                  </div>
                );
              })}

              {/* Gemini Clinical Summary Card */}
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

                  {selectedClinicalSummary.diabetes_assessment && (
                    <article className="diagnostic-support-card">
                      <h4>Đánh giá tình trạng đái tháo đường</h4>
                      <p>{selectedClinicalSummary.diabetes_assessment}</p>
                    </article>
                  )}
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
                          <h3 id="quick-recall-title" style={{ fontSize: '17px', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--primary)', margin: 0 }}>
                            <CalendarPlus size={18} /> Lên lịch tái khám nhanh
                          </h3>
                          <button type="button" aria-label="Đóng" onClick={() => setIsQuickRecallOpen(false)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}>
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
                                aria-label="Giảm"
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
                                aria-label="Tăng"
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
                                  'Đã lên lịch thành công',
                                );
                              } catch (err) {
                                dialog.showError(err.message, 'Không thể lưu lịch tái khám');
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

              {/* Doctor Review Section */}
              {currentScreeningDetail && canReviewScreening && currentScreeningDetail.status === 'AI_Analyzed' && (
                <DoctorReviewForm
                  key={currentScreeningDetail.id}
                  detail={currentScreeningDetail}
                  suggestedFollowUp={selectedClinicalSummary?.follow_up || ''}
                  onSubmit={submitCurrentScreeningReview}
                />
              )}
              {currentScreeningDetail?.status === 'Reviewed' && (
                <div className="screening-reviewed-notice" role="status">
                  <CheckCircle size={18} aria-hidden="true" />
                  <div>
                    <strong>Bác sĩ đã duyệt ca sàng lọc này</strong>
                    <p>
                      Kết quả chuyên môn đã được xác nhận. Bệnh nhân hiện có thể tra cứu chi tiết kết quả qua cổng thông tin.
                    </p>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Embedded Lightbox Modal */}
      <LightboxModal 
        image={lightboxImage}
        onClose={() => setLightboxImage(null)}
      />
    </div>
  );
}
