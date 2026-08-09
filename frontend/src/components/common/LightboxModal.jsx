import React, { useEffect } from 'react';
import { Eye, X, ZoomIn } from 'lucide-react';
import SecureImage from '../SecureImage';

/**
 * Modal xem ảnh võng mạc & bản đồ tổn thương AI phóng to toàn màn hình
 */
export default function LightboxModal({ image, onClose }) {
  useEffect(() => {
    if (!image) return undefined;
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [image, onClose]);

  if (!image) return null;

  return (
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
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label={image.title || 'Xem ảnh võng mạc'}
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
            {image.title}
          </h3>
          {image.maskSrc && (
            <span className="badge" style={{ backgroundColor: image.isMask ? '#EF4444' : '#10B981', color: '#fff', fontSize: '12px', padding: '5px 12px', fontWeight: '600' }}>
              {image.isMask ? '🔴 Bản đồ tổn thương AI (MA / HE / EX)' : '👁️ Ảnh võng mạc gốc'}
            </span>
          )}
        </div>

        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          {image.maskSrc && (
            <button
              className="btn btn-secondary"
              style={{ backgroundColor: 'rgba(255,255,255,0.18)', color: '#fff', border: '1px solid rgba(255,255,255,0.35)', padding: '8px 16px', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '8px', fontWeight: '600', borderRadius: '8px' }}
              onClick={() => {
                const nextIsMask = !image.isMask;
                image.onToggleMask && image.onToggleMask(nextIsMask);
              }}
            >
              <Eye size={15} /> {image.isMask ? 'Chuyển sang Ảnh gốc' : 'Chuyển sang Mask AI'}
            </button>
          )}
          <button
            style={{ backgroundColor: 'rgba(255,255,255,0.25)', border: 'none', color: '#fff', borderRadius: '50%', width: '40px', height: '40px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'all 0.2s' }}
            onClick={onClose}
            title="Đóng (ESC)"
            aria-label="Đóng"
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
          src={image.src} 
          alt={image.title || 'Ảnh võng mạc'}
          style={{ maxWidth: '100%', maxHeight: '80vh', width: 'auto', height: 'auto', objectFit: 'contain', display: 'block' }}
        />
      </div>

      {/* Lightbox Sub-Caption & Tip */}
      <div style={{ marginTop: '14px', color: 'rgba(255,255,255,0.75)', fontSize: '13px', textAlign: 'center', display: 'flex', alignItems: 'center', gap: '8px' }}>
        <ZoomIn size={14} /> Bác sĩ có thể nhấp <strong>Chuyển đổi Mask AI</strong> ở góc phải để đối chiếu trực tiếp tổn thương với ảnh võng mạc gốc.
      </div>
    </div>
  );
}
