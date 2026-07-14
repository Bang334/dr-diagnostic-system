import React, { useState, useEffect } from 'react';
import {
  X, Calendar, Phone, MapPin, Activity, FileText, Printer,
  Clock, ClipboardList, CalendarCheck, ChevronRight, AlertTriangle, CheckCircle,
  Eye, Stethoscope
} from 'lucide-react';
import { api } from '../services/api';

const DR_LABELS = {
  0: 'No DR',
  1: 'Mild NPDR',
  2: 'Moderate NPDR',
  3: 'Severe NPDR',
  4: 'Proliferative DR',
};

const DR_COLORS = {
  0: { color: '#10B981', bg: 'rgba(16,185,129,0.08)' },
  1: { color: '#3B82F6', bg: 'rgba(59,130,246,0.08)' },
  2: { color: '#F59E0B', bg: 'rgba(245,158,11,0.08)' },
  3: { color: '#EF4444', bg: 'rgba(239,68,68,0.08)' },
  4: { color: '#7C3AED', bg: 'rgba(124,58,237,0.08)' },
};

const RISK_COLORS = {
  Low: { color: '#10B981', bg: 'rgba(16,185,129,0.1)', label: 'Thấp' },
  Medium: { color: '#F59E0B', bg: 'rgba(245,158,11,0.1)', label: 'Trung bình' },
  High: { color: '#EF4444', bg: 'rgba(239,68,68,0.1)', label: 'Cao' },
  Urgent: { color: '#7C3AED', bg: 'rgba(124,58,237,0.1)', label: 'Khẩn cấp' },
};

const RECALL_STATUS_COLORS = {
  Scheduled: { color: '#3B82F6', label: 'Đã lên lịch' },
  Completed: { color: '#10B981', label: 'Đã hoàn thành' },
  Overdue: { color: '#EF4444', label: 'Quá hạn' },
  Cancelled: { color: '#6B7280', label: 'Đã huỷ' },
};

export default function PatientDetailsModal({ isOpen, onClose, patient, onStartScreening }) {
  const [activeTab, setActiveTab] = useState('profile');
  const [screenings, setScreenings] = useState([]);
  const [recalls, setRecalls] = useState([]);
  const [isLoadingScreenings, setIsLoadingScreenings] = useState(false);
  const [isLoadingRecalls, setIsLoadingRecalls] = useState(false);

  const qrCodeUrl = patient
    ? `https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=${encodeURIComponent(patient.patient_code)}`
    : '';

  useEffect(() => {
    if (isOpen && patient) {
      setActiveTab('profile');
      setScreenings([]);
      setRecalls([]);
    }
  }, [isOpen, patient]);

  useEffect(() => {
    if (activeTab === 'history' && patient && screenings.length === 0) {
      setIsLoadingScreenings(true);
      api.getPatientScreenings(patient.id)
        .then(data => setScreenings(data))
        .catch(err => console.error(err))
        .finally(() => setIsLoadingScreenings(false));
    }
    if (activeTab === 'recalls' && patient && recalls.length === 0) {
      setIsLoadingRecalls(true);
      api.getPatientRecalls(patient.id)
        .then(data => setRecalls(data))
        .catch(err => console.error(err))
        .finally(() => setIsLoadingRecalls(false));
    }
  }, [activeTab, patient]);

  const handlePrint = () => {
    const printWindow = window.open('', '_blank');
    printWindow.document.write(`
      <html>
        <head>
          <title>Mã QR Bệnh Nhân - ${patient.full_name}</title>
          <style>
            body { font-family: sans-serif; text-align: center; padding: 40px; }
            .card { border: 2px solid #ccc; padding: 20px; border-radius: 12px; display: inline-block; }
            h2 { margin: 10px 0; }
            p { color: #555; font-size: 16px; margin: 5px 0; }
            .qr { margin: 20px 0; }
          </style>
        </head>
        <body>
          <div class="card">
            <p>HỆ THỐNG SÀNG LỌC VÕNG MẠC DR</p>
            <div class="qr">
              <img src="${qrCodeUrl}" width="200" height="200" />
            </div>
            <h2>${patient.full_name}</h2>
            <p><strong>Mã BN:</strong> ${patient.patient_code}</p>
            <p><strong>Ngày sinh:</strong> ${patient.date_of_birth}</p>
            <p><strong>Số điện thoại:</strong> ${patient.phone_number || 'N/A'}</p>
          </div>
          <script>
            window.onload = function() { window.print(); window.close(); }
          </script>
        </body>
      </html>
    `);
    printWindow.document.close();
  };

  if (!isOpen || !patient) return null;

  const tabStyle = (tab) => ({
    padding: '8px 16px',
    fontSize: '13px',
    fontWeight: '600',
    cursor: 'pointer',
    border: 'none',
    background: 'none',
    borderBottom: activeTab === tab ? '2px solid var(--primary)' : '2px solid transparent',
    color: activeTab === tab ? 'var(--primary)' : 'var(--text-muted)',
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    transition: 'color 0.2s',
  });

  return (
    <div style={{
      position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
      backgroundColor: 'rgba(0,0,0,0.65)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      zIndex: 1000, backdropFilter: 'blur(4px)'
    }}>
      <div className="card" style={{
        width: '92%', maxWidth: '780px',
        maxHeight: '92vh', overflowY: 'auto',
        position: 'relative', display: 'flex', flexDirection: 'column', gap: 0,
        animation: 'fadeIn 0.2s ease-out', padding: 0
      }}>
        {/* Header */}
        <div style={{ padding: '20px 24px 0', borderBottom: '1px solid var(--border-light)' }}>
          <button onClick={onClose} style={{
            position: 'absolute', top: '16px', right: '16px',
            background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)'
          }}>
            <X size={22} />
          </button>

          <h3 style={{ fontSize: '19px', color: 'var(--primary)', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Stethoscope size={20} /> Hồ sơ bệnh nhân: <strong>{patient.full_name}</strong>
            &nbsp;<span style={{ fontSize: '14px', color: 'var(--text-muted)', fontWeight: 'normal' }}>({patient.patient_code})</span>
          </h3>

          {/* Tab Navigation */}
          <div style={{ display: 'flex', gap: '4px', borderBottom: 'none' }}>
            <button style={tabStyle('profile')} onClick={() => setActiveTab('profile')}>
              <ClipboardList size={14} /> Hồ sơ
            </button>
            <button style={tabStyle('history')} onClick={() => setActiveTab('history')}>
              <Clock size={14} /> Lịch sử khám
            </button>
            <button style={tabStyle('recalls')} onClick={() => setActiveTab('recalls')}>
              <CalendarCheck size={14} /> Lịch tái khám
            </button>
          </div>
        </div>

        {/* Tab: Profile */}
        {activeTab === 'profile' && (
          <div style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px', animation: 'fadeIn 0.15s ease' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: '30px' }}>
              {/* Thông tin chi tiết */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <div>
                  <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Họ và tên</span>
                  <p style={{ fontSize: '18px', fontWeight: 'bold' }}>{patient.full_name}</p>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                  <div>
                    <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Giới tính</span>
                    <p style={{ fontWeight: '500' }}>{patient.gender}</p>
                  </div>
                  <div>
                    <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Ngày sinh</span>
                    <p style={{ fontWeight: '500', display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <Calendar size={13} /> {patient.date_of_birth}
                    </p>
                  </div>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Liên hệ & Địa chỉ</span>
                  <p style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px' }}>
                    <Phone size={13} /> {patient.phone_number || 'Chưa cập nhật'}
                  </p>
                  <p style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px' }}>
                    <MapPin size={13} /> {patient.address || 'Chưa cập nhật'}
                  </p>
                </div>

                <div style={{ borderTop: '1px solid var(--border-light)', paddingTop: '16px' }}>
                  <h4 style={{ fontSize: '13px', marginBottom: '10px', color: 'var(--primary)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Activity size={15} /> Chỉ số lâm sàng
                  </h4>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '10px', backgroundColor: 'var(--background)', padding: '12px', borderRadius: '8px' }}>
                    <div>
                      <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Loại tiểu đường</span>
                      <p style={{ fontWeight: 'bold', fontSize: '14px' }}>{patient.diabetes_type || 'N/A'}</p>
                    </div>
                    <div>
                      <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Thời gian mắc</span>
                      <p style={{ fontWeight: 'bold', fontSize: '14px' }}>{patient.diabetes_duration_years ? `${patient.diabetes_duration_years} năm` : 'N/A'}</p>
                    </div>
                    <div>
                      <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>HbA1c</span>
                      <p style={{ fontWeight: 'bold', fontSize: '14px', color: '#D97706' }}>{patient.latest_hba1c ? `${patient.latest_hba1c}%` : 'N/A'}</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* QR Code */}
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '14px', borderLeft: '1px solid var(--border-light)', paddingLeft: '20px' }}>
                <div style={{ padding: '10px', border: '1px solid var(--border-light)', borderRadius: '12px', backgroundColor: '#fff' }}>
                  <img src={qrCodeUrl} alt="Patient QR Code" style={{ width: '160px', height: '160px', display: 'block' }} />
                </div>
                <p style={{ fontSize: '11px', color: 'var(--text-muted)', textAlign: 'center' }}>
                  Quét mã QR để tìm kiếm nhanh bệnh nhân
                </p>
                <button className="btn btn-secondary" style={{ gap: '8px', width: '100%', fontSize: '13px' }} onClick={handlePrint}>
                  <Printer size={14} /> In thẻ bệnh nhân / Mã QR
                </button>
              </div>
            </div>

            {/* Footer actions */}
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px', borderTop: '1px solid var(--border-light)', paddingTop: '16px' }}>
              <button className="btn btn-secondary" onClick={onClose}>Đóng</button>
              <button className="btn btn-primary" style={{ gap: '8px' }} onClick={() => { onStartScreening(patient); onClose(); }}>
                <FileText size={15} /> Khám sàng lọc mới
              </button>
            </div>
          </div>
        )}

        {/* Tab: Lịch sử khám */}
        {activeTab === 'history' && (
          <div style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px', animation: 'fadeIn 0.15s ease' }}>
            {isLoadingScreenings ? (
              <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}>
                <div style={{ width: '32px', height: '32px', border: '3px solid rgba(13,148,136,0.15)', borderTopColor: 'var(--primary)', borderRadius: '50%', animation: 'spin 1s linear infinite', margin: '0 auto 12px' }} />
                <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
                Đang tải lịch sử khám...
              </div>
            ) : screenings.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '48px 20px', color: 'var(--text-muted)' }}>
                <Clock size={40} style={{ margin: '0 auto 12px', display: 'block', opacity: 0.4 }} />
                <p style={{ fontWeight: '500' }}>Chưa có lần khám nào</p>
                <p style={{ fontSize: '13px', marginTop: '4px' }}>Bệnh nhân chưa thực hiện sàng lọc lần nào.</p>
                <button className="btn btn-primary" style={{ marginTop: '16px', gap: '8px' }} onClick={() => { onStartScreening(patient); onClose(); }}>
                  <FileText size={14} /> Bắt đầu khám sàng lọc
                </button>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Tổng <strong>{screenings.length}</strong> lần khám</p>
                {screenings.map((s, idx) => (
                  <div key={s.id} className="card" style={{
                    padding: '16px 20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                    borderLeft: '3px solid var(--primary)', gap: '12px'
                  }}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', flex: 1 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                        <span style={{ fontWeight: 'bold', fontSize: '14px' }}>
                          Lần khám #{screenings.length - idx}
                        </span>
                        <span style={{
                          padding: '2px 10px', borderRadius: '999px', fontSize: '11px', fontWeight: '600',
                          backgroundColor: s.status === 'Reviewed' ? 'rgba(16,185,129,0.1)' : s.status === 'AI_Analyzed' ? 'rgba(59,130,246,0.1)' : 'rgba(245,158,11,0.1)',
                          color: s.status === 'Reviewed' ? '#10B981' : s.status === 'AI_Analyzed' ? '#3B82F6' : '#F59E0B'
                        }}>
                          {s.status === 'Reviewed' ? 'Đã duyệt' : s.status === 'AI_Analyzed' ? 'AI đã phân tích' : 'Đang chờ'}
                        </span>
                      </div>
                      <span style={{ fontSize: '13px', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <Calendar size={12} />
                        {new Date(s.screening_date).toLocaleString('vi-VN', { dateStyle: 'medium', timeStyle: 'short' })}
                      </span>
                      {s.doctor_name && (
                        <span style={{ fontSize: '12px', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '5px' }}>
                          <Eye size={11} /> Bác sĩ: {s.doctor_name}
                        </span>
                      )}
                    </div>
                    <ChevronRight size={18} color="var(--text-muted)" />
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Tab: Lịch tái khám */}
        {activeTab === 'recalls' && (
          <div style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px', animation: 'fadeIn 0.15s ease' }}>
            {isLoadingRecalls ? (
              <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}>
                <div style={{ width: '32px', height: '32px', border: '3px solid rgba(13,148,136,0.15)', borderTopColor: 'var(--primary)', borderRadius: '50%', animation: 'spin 1s linear infinite', margin: '0 auto 12px' }} />
                Đang tải lịch tái khám...
              </div>
            ) : recalls.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '48px 20px', color: 'var(--text-muted)' }}>
                <CalendarCheck size={40} style={{ margin: '0 auto 12px', display: 'block', opacity: 0.4 }} />
                <p style={{ fontWeight: '500' }}>Chưa có lịch tái khám</p>
                <p style={{ fontSize: '13px', marginTop: '4px' }}>Lịch tái khám sẽ được tạo sau khi bác sĩ phê duyệt kết quả sàng lọc.</p>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Tổng <strong>{recalls.length}</strong> lịch tái khám</p>
                {recalls.map((r) => {
                  const risk = RISK_COLORS[r.risk_stratification] || RISK_COLORS.Low;
                  const recallStatus = RECALL_STATUS_COLORS[r.status] || RECALL_STATUS_COLORS.Scheduled;
                  const recallDate = new Date(r.recall_date);
                  const isOverdue = recallDate < new Date() && r.status === 'Scheduled';
                  return (
                    <div key={r.id} className="card" style={{
                      padding: '16px 20px', display: 'flex', flexDirection: 'column', gap: '10px',
                      borderLeft: `3px solid ${risk.color}`
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                          <span style={{ fontWeight: 'bold', fontSize: '15px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                            <CalendarCheck size={15} color={risk.color} />
                            {recallDate.toLocaleDateString('vi-VN', { dateStyle: 'long' })}
                          </span>
                          {isOverdue && (
                            <span style={{ padding: '2px 8px', borderRadius: '999px', backgroundColor: 'rgba(239,68,68,0.1)', color: '#EF4444', fontSize: '11px', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '3px' }}>
                              <AlertTriangle size={10} /> Quá hạn
                            </span>
                          )}
                        </div>
                        <div style={{ display: 'flex', gap: '8px' }}>
                          <span style={{
                            padding: '2px 10px', borderRadius: '999px', fontSize: '11px', fontWeight: '600',
                            backgroundColor: risk.bg, color: risk.color
                          }}>
                            Nguy cơ: {risk.label}
                          </span>
                          <span style={{
                            padding: '2px 10px', borderRadius: '999px', fontSize: '11px', fontWeight: '600',
                            backgroundColor: 'var(--background)', color: recallStatus.color
                          }}>
                            {recallStatus.label}
                          </span>
                        </div>
                      </div>
                      {r.recommendation && (
                        <p style={{ fontSize: '13px', color: 'var(--text-secondary)', display: 'flex', alignItems: 'flex-start', gap: '6px', margin: 0 }}>
                          <CheckCircle size={13} style={{ flexShrink: 0, marginTop: '2px', color: risk.color }} />
                          {r.recommendation}
                        </p>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
