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

export const suggestedRecallMonths = (followUp) => {
  const suggested = Number(String(followUp || '').match(/\d+/)?.[0]);
  if (RECALL_MONTH_OPTIONS.includes(suggested)) return suggested;
  return 6;
};

export const formatDate = (value, withTime = false) => {
  if (!value) return 'Chưa cập nhật';
  return new Intl.DateTimeFormat('vi-VN', withTime
    ? { dateStyle: 'long', timeStyle: 'short' }
    : { dateStyle: 'long' }).format(new Date(value));
};
