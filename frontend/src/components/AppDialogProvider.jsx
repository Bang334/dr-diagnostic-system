import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Info,
  X,
  XCircle,
} from 'lucide-react';

const AppDialogContext = createContext(null);

const DIALOG_APPEARANCE = {
  success: { icon: CheckCircle2, defaultTitle: 'Thành công' },
  error: { icon: XCircle, defaultTitle: 'Không thể thực hiện' },
  warning: { icon: AlertTriangle, defaultTitle: 'Cần xác nhận' },
  info: { icon: Info, defaultTitle: 'Thông báo' },
};

export function AppDialogProvider({ children }) {
  const [dialog, setDialog] = useState(null);

  const openDialog = useCallback((options) => new Promise((resolve) => {
    const type = options.type || 'info';
    setDialog({
      mode: 'alert',
      confirmLabel: 'Đóng',
      cancelLabel: 'Huỷ',
      ...options,
      type,
      title: options.title || DIALOG_APPEARANCE[type].defaultTitle,
      resolve,
    });
  }), []);

  const closeDialog = useCallback((result) => {
    setDialog((current) => {
      current?.resolve(result);
      return null;
    });
  }, []);

  useEffect(() => {
    if (!dialog) return undefined;
    const closeOnEscape = (event) => {
      if (event.key === 'Escape') closeDialog(false);
    };
    window.addEventListener('keydown', closeOnEscape);
    return () => window.removeEventListener('keydown', closeOnEscape);
  }, [dialog, closeDialog]);

  const value = useMemo(() => ({
    showSuccess: (message, title) => openDialog({ type: 'success', message, title }),
    showError: (message, title) => openDialog({ type: 'error', message, title }),
    showWarning: (message, title) => openDialog({ type: 'warning', message, title }),
    showInfo: (message, title) => openDialog({ type: 'info', message, title }),
    showConfirm: ({
      message,
      title,
      confirmLabel = 'Xác nhận',
      cancelLabel = 'Huỷ',
      danger = false,
    }) => openDialog({
      type: danger ? 'error' : 'warning',
      mode: 'confirm',
      message,
      title,
      confirmLabel,
      cancelLabel,
      danger,
    }),
  }), [openDialog]);

  const appearance = dialog ? DIALOG_APPEARANCE[dialog.type] : null;
  const DialogIcon = appearance?.icon;
  const titleId = dialog ? 'app-dialog-title' : undefined;
  const descriptionId = dialog ? 'app-dialog-description' : undefined;

  return (
    <AppDialogContext.Provider value={value}>
      {children}
      {dialog && (
        <div className="app-dialog-overlay">
          <section
            className={`app-dialog app-dialog-${dialog.type}`}
            role={dialog.type === 'error' ? 'alertdialog' : 'dialog'}
            aria-modal="true"
            aria-labelledby={titleId}
            aria-describedby={descriptionId}
          >
            <button
              type="button"
              className="app-dialog-close"
              aria-label="Đóng thông báo"
              onClick={() => closeDialog(false)}
            >
              <X size={19} aria-hidden="true" />
            </button>
            <div className="app-dialog-content">
              <span className="app-dialog-icon" aria-hidden="true">
                <DialogIcon size={28} strokeWidth={2.2} />
              </span>
              <div>
                <span className="app-dialog-eyebrow">Hệ thống sàng lọc DR</span>
                <h2 id={titleId}>{dialog.title}</h2>
                <p id={descriptionId}>{dialog.message}</p>
              </div>
            </div>
            <footer className="app-dialog-actions">
              {dialog.mode === 'confirm' && (
                <button type="button" className="btn btn-secondary" onClick={() => closeDialog(false)}>
                  {dialog.cancelLabel}
                </button>
              )}
              <button
                type="button"
                className={`btn ${dialog.danger ? 'btn-danger' : 'btn-primary'}`}
                onClick={() => closeDialog(true)}
                autoFocus
              >
                {dialog.confirmLabel}
              </button>
            </footer>
          </section>
        </div>
      )}
    </AppDialogContext.Provider>
  );
}

export function useAppDialog() {
  const context = useContext(AppDialogContext);
  if (!context) {
    throw new Error('useAppDialog must be used inside AppDialogProvider.');
  }
  return context;
}
