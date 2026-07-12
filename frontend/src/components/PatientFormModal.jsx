import React, { useState } from 'react';
import { X, UserPlus } from 'lucide-react';

export default function PatientFormModal({ isOpen, onClose, onSubmit }) {
  const [formData, setFormData] = useState({
    patient_code: `BN${Math.floor(1000 + Math.random() * 9000)}`, // Auto generate code
    full_name: '',
    gender: 'Nam',
    date_of_birth: '',
    phone_number: '',
    address: '',
    diabetes_type: 'Type 2',
    diabetes_duration_years: '',
    latest_hba1c: ''
  });
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setIsSubmitting(true);

    if (!formData.full_name || !formData.date_of_birth) {
      setError('Vui lòng nhập đầy đủ Họ tên và Ngày sinh.');
      setIsSubmitting(false);
      return;
    }

    try {
      const formattedData = {
        ...formData,
        diabetes_duration_years: formData.diabetes_duration_years ? parseFloat(formData.diabetes_duration_years) : null,
        latest_hba1c: formData.latest_hba1c ? parseFloat(formData.latest_hba1c) : null,
      };
      await onSubmit(formattedData);
      onClose();
    } catch (err) {
      console.error(err);
      setError(err.message || 'Có lỗi xảy ra khi thêm bệnh nhân.');
    } finally {
      setIsSubmitting(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0, 0, 0, 0.65)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000,
      backdropFilter: 'blur(4px)'
    }}>
      <div className="card" style={{
        width: '90%',
        maxWidth: '600px',
        maxHeight: '90vh',
        overflowY: 'auto',
        position: 'relative',
        display: 'flex',
        flexDirection: 'column',
        gap: '20px',
        animation: 'fadeIn 0.2s ease-out'
      }}>
        <button 
          onClick={onClose}
          style={{
            position: 'absolute',
            top: '16px',
            right: '16px',
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            color: 'var(--text-muted)'
          }}
        >
          <X size={24} />
        </button>

        <h3 style={{ fontSize: '20px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <UserPlus size={22} color="var(--primary)" /> Đăng ký bệnh nhân mới
        </h3>

        {error && (
          <div style={{ padding: '12px', backgroundColor: 'rgba(239, 68, 68, 0.1)', color: '#ef4444', borderRadius: '6px', fontSize: '14px' }}>
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <label style={{ fontSize: '13px', fontWeight: '500' }}>Mã bệnh nhân</label>
              <input 
                type="text" 
                name="patient_code"
                value={formData.patient_code}
                onChange={handleChange}
                required
                style={{ padding: '10px', borderRadius: '6px', border: '1px solid var(--border-light)', backgroundColor: 'var(--background)', color: 'var(--text-primary)' }}
              />
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <label style={{ fontSize: '13px', fontWeight: '500' }}>Họ và tên</label>
              <input 
                type="text" 
                name="full_name"
                value={formData.full_name}
                onChange={handleChange}
                required
                placeholder="Nguyễn Văn A"
                style={{ padding: '10px', borderRadius: '6px', border: '1px solid var(--border-light)', backgroundColor: 'var(--background)', color: 'var(--text-primary)' }}
              />
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <label style={{ fontSize: '13px', fontWeight: '500' }}>Giới tính</label>
              <select 
                name="gender"
                value={formData.gender}
                onChange={handleChange}
                style={{ padding: '10px', borderRadius: '6px', border: '1px solid var(--border-light)', backgroundColor: 'var(--background)', color: 'var(--text-primary)' }}
              >
                <option value="Nam">Nam</option>
                <option value="Nữ">Nữ</option>
                <option value="Khác">Khác</option>
              </select>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <label style={{ fontSize: '13px', fontWeight: '500' }}>Ngày sinh</label>
              <input 
                type="date" 
                name="date_of_birth"
                value={formData.date_of_birth}
                onChange={handleChange}
                required
                style={{ padding: '10px', borderRadius: '6px', border: '1px solid var(--border-light)', backgroundColor: 'var(--background)', color: 'var(--text-primary)' }}
              />
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <label style={{ fontSize: '13px', fontWeight: '500' }}>Số điện thoại</label>
              <input 
                type="tel" 
                name="phone_number"
                value={formData.phone_number}
                onChange={handleChange}
                placeholder="09XXXXXXXX"
                style={{ padding: '10px', borderRadius: '6px', border: '1px solid var(--border-light)', backgroundColor: 'var(--background)', color: 'var(--text-primary)' }}
              />
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <label style={{ fontSize: '13px', fontWeight: '500' }}>Địa chỉ</label>
              <input 
                type="text" 
                name="address"
                value={formData.address}
                onChange={handleChange}
                placeholder="Hà Nội"
                style={{ padding: '10px', borderRadius: '6px', border: '1px solid var(--border-light)', backgroundColor: 'var(--background)', color: 'var(--text-primary)' }}
              />
            </div>
          </div>

          <div style={{ borderTop: '1px solid var(--border-light)', paddingTop: '16px', marginTop: '8px' }}>
            <h4 style={{ fontSize: '14px', marginBottom: '12px', color: 'var(--primary)' }}>Thông tin lâm sàng (Tiểu đường)</h4>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <label style={{ fontSize: '12px', fontWeight: '500' }}>Loại tiểu đường</label>
                <select 
                  name="diabetes_type"
                  value={formData.diabetes_type}
                  onChange={handleChange}
                  style={{ padding: '10px', borderRadius: '6px', border: '1px solid var(--border-light)', backgroundColor: 'var(--background)', color: 'var(--text-primary)' }}
                >
                  <option value="Type 1">Type 1</option>
                  <option value="Type 2">Type 2</option>
                  <option value="LADA">LADA</option>
                  <option value="Thai kỳ">Thai kỳ</option>
                  <option value="Khác">Khác</option>
                </select>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <label style={{ fontSize: '12px', fontWeight: '500' }}>Số năm mắc bệnh</label>
                <input 
                  type="number" 
                  name="diabetes_duration_years"
                  value={formData.diabetes_duration_years}
                  onChange={handleChange}
                  step="0.1"
                  placeholder="Ví dụ: 5"
                  style={{ padding: '10px', borderRadius: '6px', border: '1px solid var(--border-light)', backgroundColor: 'var(--background)', color: 'var(--text-primary)' }}
                />
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <label style={{ fontSize: '12px', fontWeight: '500' }}>HbA1c gần nhất (%)</label>
                <input 
                  type="number" 
                  name="latest_hba1c"
                  value={formData.latest_hba1c}
                  onChange={handleChange}
                  step="0.01"
                  placeholder="Ví dụ: 7.2"
                  style={{ padding: '10px', borderRadius: '6px', border: '1px solid var(--border-light)', backgroundColor: 'var(--background)', color: 'var(--text-primary)' }}
                />
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px', marginTop: '16px' }}>
            <button type="button" className="btn btn-secondary" onClick={onClose}>Hủy</button>
            <button type="submit" className="btn btn-primary" disabled={isSubmitting}>
              {isSubmitting ? 'Đang lưu...' : 'Lưu bệnh nhân'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
