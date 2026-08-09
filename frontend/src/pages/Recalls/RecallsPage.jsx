import React, { useState, useEffect } from 'react';
import { CalendarCheck, AlertTriangle } from 'lucide-react';
import { api } from '../../services/api';
import { RISK_COLORS, RECALL_STATUS } from '../../utils/clinicalFormatters';

export default function RecallsPage({ patients: initialPatients }) {
  const [allRecalls, setAllRecalls] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchAll = async () => {
      setIsLoading(true);
      try {
        let patientList = initialPatients;
        if (!patientList || patientList.length === 0) {
          patientList = await api.getPatients();
        }
        if (patientList && patientList.length > 0) {
          const results = await Promise.all(
            patientList.map(p => api.getPatientRecalls(p.id).then(recalls => recalls.map(r => ({ ...r, patient: p }))))
          );
          const merged = results.flat().sort((a, b) => new Date(a.recall_date) - new Date(b.recall_date));
          setAllRecalls(merged);
        } else {
          setAllRecalls([]);
        }
      } catch (err) {
        console.error('Error loading recalls:', err);
      } finally {
        setIsLoading(false);
      }
    };
    fetchAll();
  }, [initialPatients]);

  return (
    <div className="page-stack animated-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
      <div className="page-header">
        <h1 style={{ fontSize: '32px', marginBottom: '8px', fontWeight: 'bold' }}>Lịch Tái Khám Định Kỳ</h1>
        <p style={{ color: 'var(--text-secondary)', margin: 0 }}>Theo dõi lịch tái khám và phân tầng nguy cơ của toàn bộ bệnh nhân.</p>
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
