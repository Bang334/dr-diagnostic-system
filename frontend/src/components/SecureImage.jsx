import React, { useState, useEffect } from 'react';

/**
 * Component hiển thị ảnh an toàn từ các route yêu cầu xác thực JWT (như /uploads/...)
 */
export default function SecureImage({ src, alt, style, className }) {
  const [blobUrl, setBlobUrl] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!src) {
      setBlobUrl('');
      return;
    }

    let active = true;
    setIsLoading(true);
    setError(false);

    const token = localStorage.getItem('token');
    const headers = token ? { 'Authorization': token } : {};

    fetch(src, { headers })
      .then(res => {
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        return res.blob();
      })
      .then(blob => {
        if (active) {
          const url = URL.createObjectURL(blob);
          setBlobUrl(url);
          setIsLoading(false);
        }
      })
      .catch(err => {
        console.error('Error fetching secure image:', err);
        if (active) {
          setError(true);
          setIsLoading(false);
        }
      });

    return () => {
      active = false;
      if (blobUrl) {
        URL.revokeObjectURL(blobUrl);
      }
    };
  }, [src]);

  if (isLoading) {
    return (
      <div style={{ 
        ...style, 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center', 
        backgroundColor: 'rgba(0,0,0,0.05)', 
        color: 'var(--text-muted)', 
        fontSize: '12px',
        borderRadius: '8px'
      }}>
        Đang tải ảnh...
      </div>
    );
  }

  if (error || !src) {
    return (
      <div style={{ 
        ...style, 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center', 
        backgroundColor: 'rgba(239, 68, 68, 0.05)', 
        color: '#EF4444', 
        fontSize: '12px',
        borderRadius: '8px',
        border: '1px dashed rgba(239, 68, 68, 0.2)'
      }}>
        Không tải được ảnh
      </div>
    );
  }

  return (
    <img 
      src={blobUrl} 
      alt={alt} 
      style={{ ...style, objectFit: 'cover' }} 
      className={className} 
    />
  );
}
