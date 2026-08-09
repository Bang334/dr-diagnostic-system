export const DEFAULT_GRADING_MODELS = [
  {
    id: 'grading',
    name: 'RETFound / EfficientNetB3',
    tag: 'Mặc định',
    description: 'Pipeline phân loại 5 mức độ bệnh võng mạc tiểu đường (No DR - PDR).',
  },
  {
    id: 'semi_supervised',
    name: 'Semi-supervised (FixMatch/Teacher-Student)',
    tag: 'Nghiên cứu',
    description: 'Mô hình học bán giám sát tận dụng tập dữ liệu chưa gán nhãn.',
  },
  {
    id: 'few_shot',
    name: 'Few-shot Learning (ProtoNet)',
    tag: 'Thử nghiệm',
    description: 'Thích nghi nhanh với miền dữ liệu ít nhãn (K-shot adaptation).',
  },
];

export const SEMI_SUPERVISED_TEST_RESULT = {
  loss: 0.4953,
  accuracy: 0.8410,
  macro_f1: 0.8429,
  balanced_accuracy: 0.8486,
  qwk: 0.9144,
  per_class_recall: {
    'No DR': 0.8638,
    Mild: 0.8035,
    Moderate: 0.7247,
    Severe: 0.9500,
    Proliferative: 0.9012,
  },
};

export const GRADE_DISTANCE_TEST_RESULT = {
  total_predictions: 8691,
  correct_predictions: 7308,
  wrong_predictions: 1383,
  mean_grade_distance_all: 0.2184,
  mean_grade_distance_wrong: 1.3724,
  max_grade_distance: 4,
  wrong_grade_distance_counts: {
    1: 930,
    2: 398,
    3: 48,
    4: 7,
  },
};

export const RECALL_LABELS = {
  'No DR': 'Không DR',
  Mild: 'DR nhẹ',
  Moderate: 'DR trung bình',
  Severe: 'DR nặng',
  Proliferative: 'DR tăng sinh',
};

export const RISK_COLORS = {
  Low: { color: '#10B981', bg: 'rgba(16,185,129,0.1)', label: 'Thấp' },
  Medium: { color: '#F59E0B', bg: 'rgba(245,158,11,0.1)', label: 'Trung bình' },
  High: { color: '#EF4444', bg: 'rgba(239,68,68,0.1)', label: 'Cao' },
  Urgent: { color: '#7C3AED', bg: 'rgba(124,58,237,0.1)', label: 'Khẩn cấp' },
};

export const RECALL_STATUS = {
  Scheduled: { color: '#3B82F6', label: 'Đã lên lịch' },
  Completed: { color: '#10B981', label: 'Hoàn thành' },
  Overdue: { color: '#EF4444', label: 'Quá hạn' },
  Cancelled: { color: '#6B7280', label: 'Đã huỷ' },
};

export const formatPercent = (value) => `${(value * 100).toFixed(2)}%`;
export const formatMetric = (value) => value.toFixed(4);
export const formatInteger = (value) => new Intl.NumberFormat('vi-VN').format(value);
