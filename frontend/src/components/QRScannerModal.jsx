import React, { useEffect, useRef, useState } from 'react';
import { Html5Qrcode } from 'html5-qrcode';
import { X, Camera, AlertCircle, Upload } from 'lucide-react';

export default function QRScannerModal({ isOpen, onClose, onScanSuccess }) {
  const [error, setError] = useState('');
  const [isScanning, setIsScanning] = useState(false);
  const qrRegionId = 'qr-reader-element';
  const html5QrCodeRef = useRef(null);

  useEffect(() => {
    if (isOpen) {
      setError('');
      setIsScanning(true);
      // Wait for DOM element to render
      setTimeout(() => {
        startScanner();
      }, 300);
    } else {
      stopScanner();
    }

    return () => {
      stopScanner();
    };
  }, [isOpen]);

  const startScanner = async () => {
    try {
      const html5QrCode = new Html5Qrcode(qrRegionId);
      html5QrCodeRef.current = html5QrCode;

      const config = { fps: 10, qrbox: { width: 250, height: 250 } };
      
      await html5QrCode.start(
        { facingMode: 'environment' },
        config,
        (decodedText) => {
          // Success callback
          onScanSuccess(decodedText);
          stopScanner();
          onClose();
        },
        (errorMessage) => {
          // Silent failure for frame-by-frame scan fails
        }
      );
    } catch (err) {
      console.error('Failed to start scanner:', err);
      setError('Không thể truy cập camera. Vui lòng kiểm tra quyền hoặc sử dụng trình duyệt hỗ trợ HTTPS.');
      setIsScanning(false);
    }
  };

  const stopScanner = async () => {
    if (html5QrCodeRef.current && html5QrCodeRef.current.isScanning) {
      try {
        await html5QrCodeRef.current.stop();
        html5QrCodeRef.current = null;
      } catch (err) {
        console.error('Failed to stop scanner:', err);
      }
    }
    setIsScanning(false);
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    try {
      setError('');
      // Dừng quét camera trước khi quét file
      await stopScanner();

      const html5QrCode = new Html5Qrcode(qrRegionId);
      const decodedText = await html5QrCode.scanFile(file, false);
      
      onScanSuccess(decodedText);
      onClose();
    } catch (err) {
      console.error('Failed to scan file:', err);
      setError('Không tìm thấy mã QR hợp lệ trong ảnh này. Vui lòng chọn ảnh rõ nét hơn.');
      // Bắt đầu lại camera để người dùng tiếp tục
      startScanner();
    }
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" style={{
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
      <div className="card modal-card qr-modal" role="dialog" aria-modal="true" aria-labelledby="qr-scanner-title" style={{
        width: '90%',
        maxWidth: '500px',
        position: 'relative',
        display: 'flex',
        flexDirection: 'column',
        gap: '20px',
        animation: 'fadeIn 0.2s ease-out'
      }}>
        <button 
          type="button"
          aria-label="Đóng trình quét QR"
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

        <h3 id="qr-scanner-title" style={{ fontSize: '20px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Camera size={22} color="var(--primary)" /> Quét mã QR Bệnh nhân
        </h3>

        <div style={{
          width: '100%',
          backgroundColor: '#000',
          borderRadius: 'var(--radius-md)',
          overflow: 'hidden',
          aspectRatio: '1',
          position: 'relative',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <div id={qrRegionId} style={{ width: '100%', height: '100%' }}></div>
          {error && (
            <div style={{
              position: 'absolute',
              padding: '20px',
              color: '#ef4444',
              textAlign: 'center',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '10px',
              zIndex: 10
            }}>
              <AlertCircle size={32} />
              <p style={{ fontSize: '14px' }}>{error}</p>
            </div>
          )}
        </div>

        <p style={{ fontSize: '13px', color: 'var(--text-secondary)', textAlign: 'center' }}>
          Đặt mã QR của bệnh nhân vào giữa khung camera để tự động quét
        </p>

        <div style={{ 
          display: 'flex', 
          flexDirection: 'column', 
          alignItems: 'center', 
          gap: '10px', 
          padding: '15px 0',
          borderTop: '1px solid var(--border-color, #e2e8f0)',
          borderBottom: '1px solid var(--border-color, #e2e8f0)'
        }}>
          <span style={{ fontSize: '13px', color: 'var(--text-muted, #64748b)' }}>Hoặc sử dụng ảnh chụp mã QR:</span>
          <label className="btn btn-secondary" style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', margin: 0 }}>
            <Upload size={16} /> Tải ảnh mã QR lên
            <input 
              type="file" 
              accept="image/*" 
              onChange={handleFileUpload} 
              style={{ display: 'none' }} 
            />
          </label>
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
          <button className="btn btn-secondary" onClick={onClose}>Hủy bỏ</button>
        </div>
      </div>
    </div>
  );
}
