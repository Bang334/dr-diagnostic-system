import React, { useState, useEffect, useCallback } from 'react';
import { QrCode, Plus, Search } from 'lucide-react';
import { api } from '../../services/api';
import { useAuth } from '../../contexts/AuthContext';
import { useAppDialog } from '../../components/AppDialogProvider';
import { resolvePatientFromQr } from './patientQrLookup';

// Modals
import QRScannerModal from '../../components/QRScannerModal';
import PatientFormModal from '../../components/PatientFormModal';
import PatientDetailsModal from '../../components/PatientDetailsModal';

export default function PatientsPage({ onStartScreening }) {
  const dialog = useAppDialog();
  const { currentUser } = useAuth();

  const [patients, setPatients] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [isLoading, setIsLoading] = useState(true);

  // Modals
  const [isScannerOpen, setIsScannerOpen] = useState(false);
  const [isFormOpen, setIsFormOpen] = useState(false);
  const [isDetailsOpen, setIsDetailsOpen] = useState(false);
  const [selectedPatientDetails, setSelectedPatientDetails] = useState(null);

  const fetchPatients = useCallback(async () => {
    try {
      setIsLoading(true);
      const data = await api.getPatients(searchQuery);
      setPatients(data);
    } catch (err) {
      console.error('Error fetching patients:', err);
    } finally {
      setIsLoading(false);
    }
  }, [searchQuery]);

  useEffect(() => {
    fetchPatients();
  }, [fetchPatients]);

  const handleAddPatient = async (patientData) => {
    try {
      await api.createPatient(patientData);
      setIsFormOpen(false);
      fetchPatients();
      dialog.showSuccess('Đã thêm bệnh nhân mới thành công!', 'Thành công');
    } catch (err) {
      dialog.showError(err.message, 'Lỗi thêm bệnh nhân');
    }
  };

  const handleQRScanSuccess = async (decodedText) => {
    setIsScannerOpen(false);

    try {
      const patient = await resolvePatientFromQr(decodedText, api.getPatients);
      if (!patient) {
        dialog.showError(
          'Không tìm thấy hồ sơ khớp với mã QR vừa quét.',
          'Không tìm thấy bệnh nhân',
        );
        return;
      }

      setSelectedPatientDetails(patient);
      setIsDetailsOpen(true);
    } catch (err) {
      console.error('Error resolving patient QR code:', err);
      dialog.showError(
        'Không thể tra cứu hồ sơ bệnh nhân. Vui lòng thử lại.',
        'Lỗi quét mã QR',
      );
    }
  };

  const canReview = currentUser?.role === 'admin'
    || (
      currentUser?.role === 'doctor'
      && (currentUser.hospital_department || '').toLocaleLowerCase('vi-VN').includes('nhãn')
    );

  return (
    <div className="page-stack animated-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Header & Actions */}
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <h1 style={{ fontSize: '30px', margin: '0 0 6px', fontWeight: 'bold' }}>Quản Lý Hồ Sơ Bệnh Nhân</h1>
          <p style={{ color: 'var(--text-secondary)', margin: 0, fontSize: '14px' }}>Tra cứu, đăng ký mới và theo dõi lịch sử khám võng mạc tiểu đường.</p>
        </div>
        <div style={{ display: 'flex', gap: '10px' }}>
          <button 
            className="btn btn-secondary" 
            style={{ gap: '8px' }} 
            onClick={() => setIsScannerOpen(true)}
          >
            <QrCode size={16} /> Quét QR tìm kiếm
          </button>
          <button 
            className="btn btn-primary" 
            style={{ gap: '8px' }} 
            onClick={() => setIsFormOpen(true)}
          >
            <Plus size={16} /> Thêm Bệnh Nhân Mới
          </button>
        </div>
      </div>

      {/* Search Bar */}
      <div style={{ display: 'flex', gap: '15px' }}>
        <div className="search-shell" style={{ display: 'flex', alignItems: 'center', backgroundColor: 'var(--surface)', border: '1px solid var(--border-light)', borderRadius: 'var(--radius-sm)', padding: '0 15px', flex: 1 }}>
          <Search size={18} color="var(--text-muted)" style={{ marginRight: '10px' }} />
          <input 
            type="text" 
            aria-label="Tìm kiếm bệnh nhân"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Tìm kiếm bệnh nhân theo tên, mã bệnh nhân hoặc số điện thoại..." 
            style={{ border: 'none', background: 'transparent', outline: 'none', width: '100%', padding: '12px 0', color: 'var(--text-primary)', fontSize: '14px' }}
          />
        </div>
      </div>

      {/* Patients Table */}
      <div className="card table-card" role="region" aria-label="Danh sách bệnh nhân" tabIndex="0" style={{ padding: 0, overflow: 'hidden' }}>
        {isLoading ? (
          <div style={{ textAlign: 'center', padding: '50px', color: 'var(--text-muted)' }}>
            <div style={{ width: '32px', height: '32px', border: '3px solid rgba(13,148,136,0.1)', borderTopColor: 'var(--primary)', borderRadius: '50%', animation: 'spin 1s linear infinite', margin: '0 auto 12px' }} />
            Đang tải danh sách bệnh nhân...
          </div>
        ) : (
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
              {patients && patients.length > 0 ? (
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
                          onClick={() => onStartScreening && onStartScreening(patient)}
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
        )}
      </div>

      {/* Internal Modals */}
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
        onStartScreening={(patient) => {
          setIsDetailsOpen(false);
          onStartScreening && onStartScreening(patient);
        }}
        canReview={canReview}
      />
    </div>
  );
}
