import React, { useEffect, useMemo, useState } from 'react';
import {
  Activity,
  AlertCircle,
  CalendarCheck,
  CheckCircle2,
  ClipboardList,
  Clock3,
  FileHeart,
  HeartPulse,
  Home,
  LogOut,
  RefreshCw,
  ShieldCheck,
  UserRound,
} from 'lucide-react';

import { api } from '../services/api';
import ScreeningDetailPanel from './ScreeningDetailPanel';


const TABS = [
  { id: 'overview', label: 'Tổng quan', icon: Home },
  { id: 'history', label: 'Lịch sử khám', icon: ClipboardList },
  { id: 'recalls', label: 'Lịch tái khám', icon: CalendarCheck },
  { id: 'profile', label: 'Hồ sơ', icon: UserRound },
];

const DR_LABELS = {
  0: 'Không thấy DR trên ảnh',
  1: 'DR không tăng sinh nhẹ',
  2: 'DR không tăng sinh trung bình',
  3: 'DR không tăng sinh nặng',
  4: 'DR tăng sinh',
};

const RECALL_STATUS = {
  Scheduled: 'Đã lên lịch',
  Completed: 'Đã hoàn thành',
  Overdue: 'Quá hạn',
  Cancelled: 'Đã huỷ',
};

const RISK_LABELS = {
  Low: 'Thấp',
  Medium: 'Trung bình',
  High: 'Cao',
  Urgent: 'Khẩn cấp',
};

const formatDate = (value, withTime = false) => {
  if (!value) return 'Chưa cập nhật';
  const normalized = value.length === 10 ? `${value}T00:00:00` : value;
  return new Intl.DateTimeFormat('vi-VN', withTime
    ? { dateStyle: 'long', timeStyle: 'short' }
    : { dateStyle: 'long' }).format(new Date(normalized));
};

export default function PatientPortal({ currentUser, onLogout }) {
  const [activeTab, setActiveTab] = useState('overview');
  const [overview, setOverview] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');
  const [selectedScreeningId, setSelectedScreeningId] = useState(null);
  const [screeningDetail, setScreeningDetail] = useState(null);
  const [isLoadingDetail, setIsLoadingDetail] = useState(false);
  const [detailError, setDetailError] = useState('');

  const loadOverview = async () => {
    setIsLoading(true);
    setError('');
    try {
      setOverview(await api.getPatientPortalOverview());
    } catch (loadError) {
      setError(loadError.message);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadOverview();
  }, []);

  const openScreeningDetail = async (screeningId) => {
    setSelectedScreeningId(screeningId);
    setScreeningDetail(null);
    setDetailError('');
    setIsLoadingDetail(true);
    try {
      setScreeningDetail(await api.getScreeningDetail(screeningId));
    } catch (loadError) {
      setDetailError(loadError.message);
    } finally {
      setIsLoadingDetail(false);
    }
  };

  const closeScreeningDetail = () => {
    setSelectedScreeningId(null);
    setScreeningDetail(null);
    setDetailError('');
  };

  useEffect(() => {
    if (!selectedScreeningId) return undefined;
    const closeOnEscape = (event) => {
      if (event.key === 'Escape') closeScreeningDetail();
    };
    window.addEventListener('keydown', closeOnEscape);
    return () => window.removeEventListener('keydown', closeOnEscape);
  }, [selectedScreeningId]);

  const nextRecall = useMemo(() => {
    const recalls = overview?.recalls || [];
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    return recalls.find((recall) => (
      recall.status === 'Scheduled'
      && new Date(`${recall.recall_date}T00:00:00`) >= today
    )) || null;
  }, [overview]);

  if (isLoading) {
    return (
      <div className="patient-portal-state" aria-live="polite">
        <div className="loading-spinner" aria-hidden="true" />
        <p>Đang tải hồ sơ của bạn...</p>
      </div>
    );
  }

  if (error || !overview) {
    return (
      <div className="patient-portal-state">
        <AlertCircle size={36} aria-hidden="true" />
        <h1>Chưa thể tải hồ sơ</h1>
        <p role="alert">{error || 'Tài khoản chưa được liên kết với hồ sơ bệnh nhân.'}</p>
        <div className="patient-state-actions">
          <button className="btn btn-primary" type="button" onClick={loadOverview}>
            <RefreshCw size={16} /> Thử lại
          </button>
          <button className="btn btn-secondary" type="button" onClick={onLogout}>Đăng xuất</button>
        </div>
      </div>
    );
  }

  const { patient, screenings, recalls } = overview;
  const reviewedScreenings = screenings.filter((item) => item.status === 'Reviewed').length;

  return (
    <div className="patient-portal">
      <header className="patient-portal-header">
        <div className="patient-portal-brand">
          <span className="brand-mark"><Activity size={21} aria-hidden="true" /></span>
          <div>
            <strong>DR-Screening</strong>
            <span>Cổng thông tin bệnh nhân</span>
          </div>
        </div>
        <div className="patient-header-user">
          <div>
            <span>Xin chào</span>
            <strong>{currentUser.full_name}</strong>
          </div>
          <button className="btn btn-secondary" type="button" onClick={onLogout}>
            <LogOut size={16} /> Đăng xuất
          </button>
        </div>
      </header>

      <nav className="patient-portal-nav" aria-label="Điều hướng cổng bệnh nhân">
        {TABS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            type="button"
            className={activeTab === id ? 'active' : ''}
            aria-current={activeTab === id ? 'page' : undefined}
            onClick={() => setActiveTab(id)}
          >
            <Icon size={18} aria-hidden="true" />
            <span>{label}</span>
          </button>
        ))}
      </nav>

      <main className="patient-portal-main" id="patient-main">
        {activeTab === 'overview' && (
          <div className="patient-page-stack animated-fade-in">
            <section className="patient-welcome-card">
              <div>
                <span className="patient-eyebrow">Hồ sơ {patient.patient_code}</span>
                <h1>Chào {patient.full_name}</h1>
                <p>Theo dõi kết quả khám mắt và lịch hẹn tái khám của bạn tại một nơi.</p>
              </div>
              <ShieldCheck size={48} aria-hidden="true" />
            </section>

            <section className="patient-summary-grid" aria-label="Tóm tắt hồ sơ">
              <article className="card patient-summary-card">
                <FileHeart size={21} aria-hidden="true" />
                <span>Lần khám đã ghi nhận</span>
                <strong>{screenings.length}</strong>
                <small>{reviewedScreenings} lần đã được bác sĩ duyệt</small>
              </article>
              <article className="card patient-summary-card">
                <HeartPulse size={21} aria-hidden="true" />
                <span>HbA1c gần nhất</span>
                <strong>{patient.latest_hba1c ? `${patient.latest_hba1c}%` : '—'}</strong>
                <small>{patient.latest_hba1c ? 'Theo hồ sơ hiện tại' : 'Chưa có dữ liệu'}</small>
              </article>
              <article className="card patient-summary-card patient-next-visit">
                <CalendarCheck size={21} aria-hidden="true" />
                <span>Lịch hẹn tiếp theo</span>
                <strong>{nextRecall ? formatDate(nextRecall.recall_date) : 'Chưa có lịch'}</strong>
                <small>{nextRecall?.recommendation || 'Bác sĩ sẽ cập nhật sau khi duyệt kết quả.'}</small>
              </article>
            </section>

            <section className="patient-overview-grid">
              <article className="card patient-section-card">
                <div className="patient-section-heading">
                  <div><span className="patient-eyebrow">Gần đây</span><h2>Lịch sử khám</h2></div>
                  <button type="button" onClick={() => setActiveTab('history')}>Xem tất cả</button>
                </div>
                <ScreeningList screenings={screenings.slice(0, 3)} compact onSelect={openScreeningDetail} />
              </article>
              <article className="card patient-section-card patient-guidance-card">
                <CheckCircle2 size={24} aria-hidden="true" />
                <h2>Lưu ý dành cho bạn</h2>
                <p>Kết quả chỉ hiển thị sau khi bác sĩ xác nhận. Hãy đến đúng lịch tái khám và liên hệ cơ sở y tế nếu thị lực thay đổi đột ngột.</p>
              </article>
            </section>
          </div>
        )}

        {activeTab === 'history' && (
          <PatientSection title="Lịch sử khám" subtitle={`${screenings.length} lần khám đã được ghi nhận`}>
            <ScreeningList screenings={screenings} onSelect={openScreeningDetail} />
          </PatientSection>
        )}

        {activeTab === 'recalls' && (
          <PatientSection title="Lịch tái khám" subtitle="Theo dõi lịch hẹn và hướng dẫn của bác sĩ">
            {recalls.length ? (
              <div className="patient-record-list">
                {recalls.map((recall) => (
                  <article className="card patient-recall-item" key={recall.id}>
                    <div className="patient-record-icon"><CalendarCheck size={20} aria-hidden="true" /></div>
                    <div className="patient-record-body">
                      <div className="patient-record-title">
                        <h2>{formatDate(recall.recall_date)}</h2>
                        <span className={`patient-status patient-status-${recall.status.toLowerCase()}`}>
                          {RECALL_STATUS[recall.status] || recall.status}
                        </span>
                      </div>
                      <p>Phân tầng nguy cơ: <strong>{RISK_LABELS[recall.risk_stratification] || recall.risk_stratification}</strong></p>
                      {recall.recommendation && <p className="patient-recommendation">{recall.recommendation}</p>}
                    </div>
                  </article>
                ))}
              </div>
            ) : <EmptyState icon={CalendarCheck} title="Chưa có lịch tái khám" description="Lịch hẹn sẽ xuất hiện sau khi bác sĩ duyệt kết quả." />}
          </PatientSection>
        )}

        {activeTab === 'profile' && (
          <PatientSection title="Hồ sơ cá nhân" subtitle="Thông tin đang được cơ sở y tế lưu trữ">
            <div className="card patient-profile-card">
              <ProfileField label="Mã bệnh nhân" value={patient.patient_code} />
              <ProfileField label="Họ và tên" value={patient.full_name} />
              <ProfileField label="Ngày sinh" value={formatDate(patient.date_of_birth)} />
              <ProfileField label="Giới tính" value={patient.gender} />
              <ProfileField label="Số điện thoại" value={patient.phone_number || 'Chưa cập nhật'} />
              <ProfileField label="Địa chỉ" value={patient.address || 'Chưa cập nhật'} wide />
              <ProfileField label="Loại đái tháo đường" value={patient.diabetes_type || 'Chưa cập nhật'} />
              <ProfileField label="Thời gian mắc bệnh" value={patient.diabetes_duration_years ? `${patient.diabetes_duration_years} năm` : 'Chưa cập nhật'} />
            </div>
          </PatientSection>
        )}
      </main>

      {selectedScreeningId && (
        <div
          className="patient-screening-modal-overlay"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) closeScreeningDetail();
          }}
        >
          <div
            className="card patient-screening-modal"
            role="dialog"
            aria-modal="true"
            aria-label="Chi tiết lần khám"
          >
            <ScreeningDetailPanel
              detail={screeningDetail}
              isLoading={isLoadingDetail}
              error={detailError}
              onBack={closeScreeningDetail}
              backLabel="Đóng chi tiết"
            />
          </div>
        </div>
      )}
    </div>
  );
}

function PatientSection({ title, subtitle, children }) {
  return (
    <section className="patient-page-stack animated-fade-in">
      <div className="patient-page-heading"><h1>{title}</h1><p>{subtitle}</p></div>
      {children}
    </section>
  );
}

function ScreeningList({ screenings, compact = false, onSelect }) {
  if (!screenings.length) {
    return <EmptyState icon={Clock3} title="Chưa có lịch sử khám" description="Các lần khám của bạn sẽ được hiển thị tại đây." />;
  }
  return (
    <div className={`patient-record-list ${compact ? 'compact' : ''}`}>
      {screenings.map((screening) => {
        const reviewed = screening.status === 'Reviewed';
        return (
          <button
            type="button"
            className={`${compact ? 'patient-screening-compact' : 'card patient-screening-item'} patient-screening-button`}
            key={screening.id}
            onClick={() => onSelect?.(screening.id)}
            aria-label={`Xem chi tiết lần khám ngày ${formatDate(screening.screening_date, true)}`}
          >
            <div className="patient-record-icon"><ClipboardList size={20} aria-hidden="true" /></div>
            <div className="patient-record-body">
              <div className="patient-record-title">
                <h2>Khám ngày {formatDate(screening.screening_date, true)}</h2>
                <span className={`patient-status ${reviewed ? 'patient-status-reviewed' : 'patient-status-pending'}`}>
                  {reviewed ? 'Đã được bác sĩ duyệt' : 'Đang chờ bác sĩ duyệt'}
                </span>
              </div>
              <p>{screening.doctor_name ? `Bác sĩ phụ trách: ${screening.doctor_name}` : 'Chưa phân công bác sĩ phụ trách'}</p>
              {reviewed && !compact && (
                <div className="patient-eye-results">
                  <EyeResult label="Mắt trái" grade={screening.left_eye_grade} notes={screening.left_eye_notes} />
                  <EyeResult label="Mắt phải" grade={screening.right_eye_grade} notes={screening.right_eye_notes} />
                </div>
              )}
              <span className="btn btn-secondary patient-detail-button">
                  Xem ảnh và chi tiết kết quả
              </span>
            </div>
          </button>
        );
      })}
    </div>
  );
}

function EyeResult({ label, grade, notes }) {
  return (
    <div>
      <span>{label}</span>
      <strong>{grade === null || grade === undefined ? 'Chưa có kết luận' : DR_LABELS[grade]}</strong>
      {notes && <small>{notes}</small>}
    </div>
  );
}

function EmptyState({ icon: Icon, title, description }) {
  return (
    <div className="patient-empty-state">
      <Icon size={34} aria-hidden="true" />
      <strong>{title}</strong>
      <p>{description}</p>
    </div>
  );
}

function ProfileField({ label, value, wide = false }) {
  return <div className={wide ? 'wide' : ''}><span>{label}</span><strong>{value}</strong></div>;
}
