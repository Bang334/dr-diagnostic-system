import React, { useState } from 'react';
import {
  Activity,
  AlertCircle,
  Bot,
  CalendarCheck,
  CheckCircle2,
  ImageOff,
  Minus,
  Plus,
  Stethoscope,
  X,
} from 'lucide-react';
import { useAppDialog } from './AppDialogProvider';

const RECALL_MONTH_OPTIONS = [1, 2, 3, 6, 12];

const STATUS_LABELS = {
  Pending: 'Đang chờ phân tích',
  AI_Analyzed: 'AI đã phân tích',
  Reviewed: 'Bác sĩ đã duyệt',
  Archived: 'Đã lưu trữ',
};

const RISK_LABELS = {
  Low: 'Thấp',
  Medium: 'Trung bình',
  High: 'Cao',
  Urgent: 'Khẩn cấp',
};

const formatDate = (value, withTime = false) => {
  if (!value) return 'Chưa cập nhật';
  return new Intl.DateTimeFormat('vi-VN', withTime
    ? { dateStyle: 'long', timeStyle: 'short' }
    : { dateStyle: 'long' }).format(new Date(value));
};

export default function ScreeningDetailPanel({
  detail,
  isLoading,
  error,
  onBack,
  backLabel = 'Quay lại lịch sử',
  canReview = false,
  onSubmitReview,
}) {
  if (isLoading) {
    return (
      <div className="screening-detail-state" aria-live="polite">
        <div className="loading-spinner" aria-hidden="true" />
        <p>Đang tải kết quả lần khám...</p>
      </div>
    );
  }

  if (error || !detail) {
    return (
      <div className="screening-detail-state" role="alert">
        <AlertCircle size={30} aria-hidden="true" />
        <strong>Không thể tải chi tiết lần khám</strong>
        <p>{error || 'Dữ liệu lần khám không tồn tại.'}</p>
        <button type="button" className="btn btn-secondary" onClick={onBack}>Quay lại</button>
      </div>
    );
  }

  return (
    <section className="screening-detail-panel" aria-labelledby={`screening-detail-${detail.id}`}>
      <div className="screening-detail-header">
        <div>
          <span className="patient-eyebrow">Lần khám #{detail.id}</span>
          <h2 id={`screening-detail-${detail.id}`}>{formatDate(detail.screening_date, true)}</h2>
          <p>{detail.doctor_name ? `Bác sĩ phụ trách: ${detail.doctor_name}` : 'Chưa có bác sĩ xác nhận'}</p>
        </div>
        <span className={`screening-detail-status status-${detail.status.toLowerCase()}`}>
          {STATUS_LABELS[detail.status] || detail.status}
        </span>
      </div>

      <section className="screening-diabetes-context" aria-labelledby={`diabetes-context-${detail.id}`}>
        <div className="screening-diabetes-heading">
          <span className="screening-diabetes-icon" aria-hidden="true">
            <Activity size={21} />
          </span>
          <div>
            <span className="patient-eyebrow">Bệnh nền liên quan</span>
            <h3 id={`diabetes-context-${detail.id}`}>Bối cảnh đái tháo đường và nguy cơ DR</h3>
          </div>
        </div>
        <dl className="screening-diabetes-metrics">
          <div>
            <dt>Loại đái tháo đường</dt>
            <dd>{detail.diabetes_type || 'Chưa cập nhật'}</dd>
          </div>
          <div>
            <dt>Thời gian mắc bệnh</dt>
            <dd>{detail.diabetes_duration_years != null ? `${detail.diabetes_duration_years} năm` : 'Chưa cập nhật'}</dd>
          </div>
          <div>
            <dt>HbA1c gần nhất</dt>
            <dd>{detail.latest_hba1c != null ? `${detail.latest_hba1c}%` : 'Chưa cập nhật'}</dd>
          </div>
        </dl>
        <p>
          Bệnh võng mạc đái tháo đường (DR) là biến chứng tại võng mạc của đái tháo đường.
          Các chỉ số trên giúp bác sĩ diễn giải nguy cơ; ảnh đáy mắt không dùng để kết luận người bệnh có mắc đái tháo đường hay không.
        </p>
      </section>

      {!detail.results_visible && (
        <div className="screening-results-pending" role="status">
          <AlertCircle size={20} aria-hidden="true" />
          <div>
            <strong>Chi tiết chuyên môn đang chờ bác sĩ duyệt</strong>
            <p>{detail.visibility_message || 'Nhận định AI và kết luận bác sĩ sẽ hiển thị sau khi được xác nhận.'}</p>
          </div>
        </div>
      )}

      <div className="screening-eye-grid">
        <EyeDetailCard label="Mắt trái" eye={detail.left_eye} />
        <EyeDetailCard label="Mắt phải" eye={detail.right_eye} />
      </div>

      {canReview && detail.status === 'AI_Analyzed' && (
        <DoctorReviewForm detail={detail} onSubmit={onSubmitReview} />
      )}

      {detail.recall && (
        <section className="screening-recall-card" aria-label="Kế hoạch tái khám">
          <CalendarCheck size={22} aria-hidden="true" />
          <div>
            <span>Kế hoạch tái khám</span>
            <strong>{formatDate(detail.recall.recall_date)}</strong>
            <p>
              Nguy cơ: {RISK_LABELS[detail.recall.risk_stratification] || detail.recall.risk_stratification}
              {detail.recall.recommendation ? ` · ${detail.recall.recommendation}` : ''}
            </p>
          </div>
        </section>
      )}

      <p className="screening-detail-disclaimer">
        Nhận định AI chỉ hỗ trợ phân độ DR trên ảnh đáy mắt. Kết luận của bác sĩ là kết quả được sử dụng cho theo dõi lâm sàng.
      </p>

      <footer className="screening-detail-footer">
        <button type="button" className="btn btn-secondary" onClick={onBack}>
          <X size={17} aria-hidden="true" /> {backLabel}
        </button>
      </footer>
    </section>
  );
}

export function DoctorReviewForm({ detail, onSubmit }) {
  const dialog = useAppDialog();
  const initialEyeReview = (eye) => ({
    final_dr_grade: eye?.ai_result?.dr_grade ?? 0,
    is_agree_with_ai: true,
    clinical_notes: '',
  });
  const [form, setForm] = useState({
    left_eye_review: detail.left_eye ? initialEyeReview(detail.left_eye) : null,
    right_eye_review: detail.right_eye ? initialEyeReview(detail.right_eye) : null,
    recall_in_months: 12,
    risk_stratification: 'Low',
    recommendation: '',
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState('');

  const updateEye = (key, field, value) => {
    setForm((current) => ({
      ...current,
      [key]: { ...current[key], [field]: value },
    }));
  };

  const stepRecallMonths = (direction) => {
    const currentIndex = RECALL_MONTH_OPTIONS.indexOf(Number(form.recall_in_months));
    const nextIndex = Math.min(
      RECALL_MONTH_OPTIONS.length - 1,
      Math.max(0, currentIndex + direction),
    );
    setForm((current) => ({
      ...current,
      recall_in_months: RECALL_MONTH_OPTIONS[nextIndex],
    }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setIsSubmitting(true);
    setSubmitError('');
    try {
      await onSubmit({
        left_eye_review: form.left_eye_review,
        right_eye_review: form.right_eye_review,
        recall_settings: {
          recall_in_months: Number(form.recall_in_months),
          risk_stratification: form.risk_stratification,
          recommendation: form.recommendation || null,
        },
      });
    } catch (error) {
      setSubmitError(error.message);
      dialog.showError(error.message, 'Không thể duyệt kết quả');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <form className="doctor-review-form" onSubmit={handleSubmit}>
      <div className="doctor-review-heading">
        <div>
          <span className="patient-eyebrow">Bước duyệt chuyên môn</span>
          <h3>Duyệt và ghi kết luận bác sĩ</h3>
        </div>
        <Stethoscope size={25} aria-hidden="true" />
      </div>

      <div className="doctor-review-eye-grid">
        {form.left_eye_review && (
          <EyeReviewFields
            label="Mắt trái"
            value={form.left_eye_review}
            onChange={(field, value) => updateEye('left_eye_review', field, value)}
          />
        )}
        {form.right_eye_review && (
          <EyeReviewFields
            label="Mắt phải"
            value={form.right_eye_review}
            onChange={(field, value) => updateEye('right_eye_review', field, value)}
          />
        )}
      </div>

      <fieldset className="doctor-recall-fields">
        <legend>Kế hoạch tái khám</legend>
        <div className="doctor-review-month-field">
          <label htmlFor="doctor-recall-months">Tái khám sau</label>
          <div className="recall-month-control">
            <button
              type="button"
              aria-label="Giảm khoảng thời gian tái khám"
              onClick={() => stepRecallMonths(-1)}
              disabled={Number(form.recall_in_months) === RECALL_MONTH_OPTIONS[0]}
            >
              <Minus size={16} aria-hidden="true" />
            </button>
            <select
              id="doctor-recall-months"
              value={form.recall_in_months}
              onChange={(event) => setForm((current) => ({
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
              aria-label="Tăng khoảng thời gian tái khám"
              onClick={() => stepRecallMonths(1)}
              disabled={Number(form.recall_in_months) === RECALL_MONTH_OPTIONS.at(-1)}
            >
              <Plus size={16} aria-hidden="true" />
            </button>
          </div>
        </div>
        <label>
          Mức nguy cơ
          <select
            value={form.risk_stratification}
            onChange={(event) => setForm((current) => ({ ...current, risk_stratification: event.target.value }))}
          >
            <option value="Low">Thấp</option>
            <option value="Medium">Trung bình</option>
            <option value="High">Cao</option>
            <option value="Urgent">Khẩn cấp</option>
          </select>
        </label>
        <label className="wide">
          Khuyến nghị tái khám
          <textarea
            rows="2"
            value={form.recommendation}
            onChange={(event) => setForm((current) => ({ ...current, recommendation: event.target.value }))}
            placeholder="Ví dụ: Theo dõi HbA1c và tái khám đúng hẹn."
          />
        </label>
      </fieldset>

      {submitError && <p className="doctor-review-error" role="alert">{submitError}</p>}
      <button type="submit" className="btn btn-primary doctor-review-submit" disabled={isSubmitting}>
        <CheckCircle2 size={16} aria-hidden="true" />
        {isSubmitting ? 'Đang lưu kết luận...' : 'Xác nhận kết quả và chuyển sang Đã duyệt'}
      </button>
    </form>
  );
}

function EyeReviewFields({ label, value, onChange }) {
  return (
    <fieldset className="doctor-eye-review-fields">
      <legend>{label}</legend>
      <label>
        Kết luận DR
        <select
          value={value.final_dr_grade}
          onChange={(event) => onChange('final_dr_grade', Number(event.target.value))}
        >
          <option value="0">Grade 0 — Không thấy DR</option>
          <option value="1">Grade 1 — NPDR nhẹ</option>
          <option value="2">Grade 2 — NPDR trung bình</option>
          <option value="3">Grade 3 — NPDR nặng</option>
          <option value="4">Grade 4 — DR tăng sinh</option>
        </select>
      </label>
      <label className="doctor-agreement-field">
        <input
          type="checkbox"
          checked={value.is_agree_with_ai}
          onChange={(event) => onChange('is_agree_with_ai', event.target.checked)}
        />
        Đồng ý với nhận định AI
      </label>
      <label>
        Ghi chú lâm sàng
        <textarea
          rows="3"
          value={value.clinical_notes}
          onChange={(event) => onChange('clinical_notes', event.target.value)}
          placeholder="Nhập nhận xét của bác sĩ..."
        />
      </label>
    </fieldset>
  );
}

function EyeDetailCard({ label, eye }) {
  if (!eye) {
    return (
      <article className="screening-eye-card screening-eye-empty">
        <ImageOff size={30} aria-hidden="true" />
        <h3>{label}</h3>
        <p>Không có ảnh hoặc kết quả cho mắt này.</p>
      </article>
    );
  }

  const ai = eye.ai_result;
  const review = eye.doctor_review;
  const detectedLesions = eye.segmentation
    ? [
      eye.segmentation.microaneurysm_detected && 'Vi phình mạch',
      eye.segmentation.hemorrhage_detected && 'Xuất huyết',
      eye.segmentation.hard_exudate_detected && 'Xuất tiết cứng',
    ].filter(Boolean)
    : [];

  return (
    <article className="screening-eye-card">
      <div className="screening-eye-heading">
        <h3>{label}</h3>
        <span>{eye.eye === 'L' ? 'L' : 'R'}</span>
      </div>

      {eye.image_url ? (
        <a href={eye.image_url} target="_blank" rel="noreferrer" className="screening-fundus-link">
          <img src={eye.image_url} alt={`Ảnh đáy ${label.toLowerCase()}`} />
          <span>Mở ảnh kích thước đầy đủ</span>
        </a>
      ) : (
        <div className="screening-image-placeholder"><ImageOff size={26} /> Không có ảnh</div>
      )}

      <section className="screening-result-block ai-result-block">
        <div className="screening-result-title"><Bot size={18} aria-hidden="true" /><strong>Nhận định AI</strong></div>
        {ai ? (
          <>
            <div className="screening-grade-row">
              <strong>{ai.dr_label}</strong>
              <span>Grade {ai.dr_grade}</span>
            </div>
            <p>Độ tin cậy: <strong>{Math.round(ai.confidence * 100)}%</strong></p>
            {detectedLesions.length > 0
              ? <p>Tổn thương phát hiện: {detectedLesions.join(', ')}.</p>
              : <p>Chưa phát hiện nhóm tổn thương được mô hình phân đoạn ghi nhận.</p>}
            <small>Mô hình: {ai.model_version}</small>
          </>
        ) : <p>Chưa có kết quả phân tích AI.</p>}
      </section>

      <section className="screening-result-block doctor-result-block">
        <div className="screening-result-title"><Stethoscope size={18} aria-hidden="true" /><strong>Kết luận bác sĩ</strong></div>
        {review ? (
          <>
            <div className="screening-grade-row">
              <strong>{review.final_dr_label}</strong>
              <span>Grade {review.final_dr_grade}</span>
            </div>
            <p className="screening-agreement">
              <CheckCircle2 size={14} aria-hidden="true" />
              {review.is_agree_with_ai ? 'Bác sĩ đồng ý với nhận định AI' : 'Bác sĩ đã điều chỉnh nhận định AI'}
            </p>
            {review.clinical_notes && <p className="screening-clinical-notes">{review.clinical_notes}</p>}
            <small>
              {review.doctor_name || 'Bác sĩ phụ trách'}
              {review.confirmed_at ? ` · ${formatDate(review.confirmed_at, true)}` : ''}
            </small>
          </>
        ) : <p>Đang chờ bác sĩ xác nhận kết quả.</p>}
      </section>
    </article>
  );
}
