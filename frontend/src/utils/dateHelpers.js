export const RECALL_MONTH_OPTIONS = [1, 2, 3, 6, 12];

export const addMonthsToToday = (months) => {
  const today = new Date();
  const originalDay = today.getDate();
  today.setDate(1);
  today.setMonth(today.getMonth() + Number(months));
  const lastDay = new Date(today.getFullYear(), today.getMonth() + 1, 0).getDate();
  today.setDate(Math.min(originalDay, lastDay));
  const year = today.getFullYear();
  const month = String(today.getMonth() + 1).padStart(2, '0');
  const day = String(today.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
};

export const suggestedRecallMonths = (followUp, grades = []) => {
  const followUpMonths = [...String(followUp || '').matchAll(/(\d+)\s*(?:[–-]\s*\d+\s*)?tháng/gi)]
    .map((match) => Number(match[1]))
    .filter((months) => RECALL_MONTH_OPTIONS.includes(months));
  if (followUpMonths.length > 0) return Math.min(...followUpMonths);

  const gradeMonths = grades
    .filter((grade) => Number.isInteger(grade))
    .map((grade) => ({ 0: 12, 1: 6, 2: 3, 3: 3, 4: 1 }[grade]))
    .filter(Boolean);
  if (gradeMonths.length > 0) return Math.min(...gradeMonths);

  return 6;
};

export const formatDate = (value, withTime = false) => {
  if (!value) return 'Chưa cập nhật';
  return new Intl.DateTimeFormat('vi-VN', withTime
    ? { dateStyle: 'long', timeStyle: 'short' }
    : { dateStyle: 'long' }).format(new Date(value));
};
