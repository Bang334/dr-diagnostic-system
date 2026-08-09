import React from 'react';
import { Target, Award, Gauge, Activity } from 'lucide-react';
import { colors } from '../../theme/colors';
import { 
  SEMI_SUPERVISED_TEST_RESULT, 
  GRADE_DISTANCE_TEST_RESULT, 
  RECALL_LABELS, 
  formatPercent, 
  formatMetric, 
  formatInteger 
} from '../../utils/clinicalFormatters';

export default function DashboardPage() {
  return (
    <div className="page-stack animated-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
      <div className="page-header">
        <h1 style={{ fontSize: '32px', marginBottom: '8px', fontWeight: 'bold' }}>Tổng Quan & Báo Cáo Nghiên Cứu</h1>
        <p style={{ color: 'var(--text-secondary)', margin: 0 }}>
          Đánh giá định lượng hiệu năng mô hình phân loại DR và phân đoạn tổn thương trên tập dữ liệu chuẩn.
        </p>
      </div>

      {/* Top 4 Metrics */}
      <div className="metric-grid model-metric-grid">
        <div className="card metric-card model-metric-card">
          <div className="metric-label"><Target size={17} aria-hidden="true" /> Accuracy</div>
          <strong className="metric-value">{formatPercent(SEMI_SUPERVISED_TEST_RESULT.accuracy)}</strong>
          <span className="metric-caption">Tỉ lệ dự đoán đúng chính xác grade</span>
        </div>
        <div className="card metric-card model-metric-card">
          <div className="metric-label"><Award size={17} aria-hidden="true" /> QWK</div>
          <strong className="metric-value metric-value-primary">{formatMetric(SEMI_SUPERVISED_TEST_RESULT.qwk)}</strong>
          <span className="metric-caption">Mức đồng thuận có xét khoảng cách grade</span>
        </div>
        <div className="card metric-card model-metric-card">
          <div className="metric-label"><Gauge size={17} aria-hidden="true" /> Macro F1</div>
          <strong className="metric-value">{formatMetric(SEMI_SUPERVISED_TEST_RESULT.macro_f1)}</strong>
          <span className="metric-caption">F1 trung bình đồng đều giữa 5 lớp</span>
        </div>
        <div className="card metric-card model-metric-card">
          <div className="metric-label"><Activity size={17} aria-hidden="true" /> Balanced Accuracy</div>
          <strong className="metric-value">{formatMetric(SEMI_SUPERVISED_TEST_RESULT.balanced_accuracy)}</strong>
          <span className="metric-caption">Độ chính xác cân bằng theo lớp</span>
        </div>
      </div>

      {/* Main Analysis Grid */}
      <div className="model-analysis-grid">
        {/* Left Column: Test result + Sensitivity */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <section className="card model-detail-card" aria-labelledby="loss-heading">
            <div className="section-heading-row">
              <div>
                <h3 id="loss-heading" style={{ margin: 0 }}>Kết quả đánh giá</h3>
              </div>
              <span className="status-chip">Semi-supervised</span>
            </div>
            <div className="loss-grid">
              <div>
                <span>Test loss</span>
                <strong>{formatMetric(SEMI_SUPERVISED_TEST_RESULT.loss)}</strong>
              </div>
              <div>
                <span>Tập đánh giá</span>
                <strong>Test set</strong>
              </div>
              <div>
                <span>Giai đoạn</span>
                <strong>Sau semi</strong>
              </div>
            </div>
            <p className="model-note">
              Đây là số liệu từ lần test lại sau khi mô hình grading tốt nhất tiếp tục được huấn luyện semi-supervised.
            </p>
          </section>

          <section className="card recall-card" aria-labelledby="recall-heading">
            <div className="section-heading-row">
              <div>
                <span className="section-eyebrow">Sensitivity theo lớp</span>
                <h3 id="recall-heading" style={{ margin: 0 }}>Recall trên test set sau semi</h3>
              </div>
              <span className="status-chip">5 mức ICDR</span>
            </div>
            <div className="recall-list">
              {Object.entries(SEMI_SUPERVISED_TEST_RESULT.per_class_recall).map(([label, recall], index) => (
                <div className="recall-row" key={label}>
                  <span className="recall-label">{RECALL_LABELS[label]}</span>
                  <div
                    className="recall-track"
                    role="progressbar"
                    aria-label={`Recall ${RECALL_LABELS[label]}`}
                    aria-valuemin="0"
                    aria-valuemax="100"
                    aria-valuenow={(recall * 100).toFixed(3)}
                  >
                    <div
                      className="recall-fill"
                      style={{
                        width: `${recall * 100}%`,
                        backgroundColor: colors.drGrades[index].color,
                      }}
                    />
                  </div>
                  <strong>{formatPercent(recall)}</strong>
                </div>
              ))}
            </div>
          </section>
        </div>

        {/* Right Column: Grade distance from test predictions */}
        <section className="card rmse-card grade-distance-card" aria-labelledby="grade-distance-heading">
          <div className="section-heading-row">
            <div>
              <span className="section-eyebrow">Sai lệch đo trực tiếp trên test set</span>
              <h3 id="grade-distance-heading" style={{ margin: 0 }}>Khoảng cách giữa grade thật và dự đoán</h3>
            </div>
            <span className="status-chip">
              {formatInteger(GRADE_DISTANCE_TEST_RESULT.total_predictions)} ảnh
            </span>
          </div>

          <p className="example-intro">
            Thống kê trực tiếp từ từng dự đoán sau semi-supervised, không suy ra từ Accuracy hoặc QWK.
          </p>

          <div className="grade-error-summary grade-distance-summary">
            <div className="grade-distance-primary">
              <span>Trung bình trên ca sai</span>
              <strong>{formatMetric(GRADE_DISTANCE_TEST_RESULT.mean_grade_distance_wrong)} grade</strong>
            </div>
            <div>
              <span>Trung bình toàn test</span>
              <strong>{formatMetric(GRADE_DISTANCE_TEST_RESULT.mean_grade_distance_all)} grade</strong>
            </div>
            <div>
              <span>Dự đoán sai</span>
              <strong>
                {formatInteger(GRADE_DISTANCE_TEST_RESULT.wrong_predictions)} ca · {' '}
                {formatPercent(
                  GRADE_DISTANCE_TEST_RESULT.wrong_predictions
                  / GRADE_DISTANCE_TEST_RESULT.total_predictions
                )}
              </strong>
            </div>
            <div>
              <span>Lệch lớn nhất</span>
              <strong>{GRADE_DISTANCE_TEST_RESULT.max_grade_distance} grade</strong>
            </div>
          </div>

          <div className="grade-distance-distribution" aria-labelledby="grade-distance-distribution-heading">
            <div className="grade-distance-distribution-heading">
              <span id="grade-distance-distribution-heading">Phân bố các ca dự đoán sai</span>
              <span>% trên {formatInteger(GRADE_DISTANCE_TEST_RESULT.wrong_predictions)} ca sai</span>
            </div>
            {Object.entries(GRADE_DISTANCE_TEST_RESULT.wrong_grade_distance_counts).map(([distance, count], index) => {
              const ratio = count / GRADE_DISTANCE_TEST_RESULT.wrong_predictions;
              return (
                <div className="grade-distance-row" key={distance}>
                  <span className="grade-distance-label">Lệch {distance} bậc</span>
                  <div
                    className="grade-distance-track"
                    role="progressbar"
                    aria-label={`Số ca dự đoán sai lệch ${distance} bậc`}
                    aria-valuemin="0"
                    aria-valuemax="100"
                    aria-valuenow={(ratio * 100).toFixed(2)}
                  >
                    <div
                      className="grade-distance-fill"
                      style={{
                        width: `${ratio * 100}%`,
                        backgroundColor: colors.drGrades[index + 1].color,
                      }}
                    />
                  </div>
                  <strong>{formatInteger(count)} ca</strong>
                  <span>{formatPercent(ratio)}</span>
                </div>
              );
            })}
          </div>

          <p className="grade-distance-note">
            {formatInteger(GRADE_DISTANCE_TEST_RESULT.correct_predictions)} dự đoán đúng có khoảng cách bằng 0.
            Trong các ca sai, phần lớn lệch 1 bậc; chỉ {' '}
            {formatInteger(
              GRADE_DISTANCE_TEST_RESULT.wrong_grade_distance_counts[3]
              + GRADE_DISTANCE_TEST_RESULT.wrong_grade_distance_counts[4]
            )} ca lệch từ 3 bậc trở lên.
          </p>
        </section>
      </div>
    </div>
  );
}
