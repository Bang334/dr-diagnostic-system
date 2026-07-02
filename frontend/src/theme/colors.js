/**
 * Premium HSL/Hex Color System
 * Dành cho Hệ thống Chẩn đoán Võng mạc Tiểu đường
 * Phong cách: Sleek Clinical Dark & Light Mode
 */

export const colors = {
  // Chế độ sáng (Premium Light Mode)
  light: {
    primary: '#0D9488',       // Teal / Emerald Deep (Màu y tế hiện đại)
    primaryHover: '#0F766E',
    secondary: '#4F46E5',     // Indigo (Cho các phần thống kê, AI)
    background: '#F8FAFC',    // Slate Light (Nền chính)
    surface: '#FFFFFF',       // Card Background
    border: '#E2E8F0',
    text: {
      primary: '#0F172A',     // Slate 900
      secondary: '#475569',   // Slate 600
      muted: '#94A3B8'        // Slate 400
    }
  },
  
  // Chế độ tối (Sleek Dark Mode)
  dark: {
    primary: '#14B8A6',       // Bright Teal
    primaryHover: '#2DD4BF',
    secondary: '#6366F1',     // Indigo Bright
    background: '#0F172A',    // Deep Slate 900
    surface: '#1E293B',       // Slate 800
    border: '#334155',
    text: {
      primary: '#F8FAFC',     // Slate 50
      secondary: '#CBD5E1',   // Slate 300
      muted: '#64748B'        // Slate 500
    }
  },

  // Màu sắc đại diện cho các mức độ bệnh võng mạc tiểu đường (DR Grades)
  drGrades: {
    0: { label: 'No DR', color: '#10B981', bg: 'rgba(16, 185, 129, 0.1)', desc: 'Bình thường, không phát hiện tổn thương võng mạc.' },
    1: { label: 'Mild NPDR', color: '#3B82F6', bg: 'rgba(59, 130, 246, 0.1)', desc: 'Bệnh lý võng mạc không tăng sinh mức độ nhẹ (Vi phình mạch).' },
    2: { label: 'Moderate NPDR', color: '#F59E0B', bg: 'rgba(245, 158, 11, 0.1)', desc: 'Bệnh lý võng mạc không tăng sinh mức độ trung bình (Xuất huyết, rỉ dịch).' },
    3: { label: 'Severe NPDR', color: '#EF4444', bg: 'rgba(239, 68, 68, 0.1)', desc: 'Bệnh lý võng mạc không tăng sinh mức độ nặng (Nguy cơ cao mất thị lực).' },
    4: { label: 'Proliferative DR', color: '#8B5CF6', bg: 'rgba(139, 92, 246, 0.1)', desc: 'Bệnh lý võng mạc tăng sinh (Có tân mạch, nguy cơ xuất huyết dịch kính).' }
  },

  // Loại tổn thương võng mạc (Lesions)
  lesions: {
    microaneurysm: { label: 'Vi phình mạch (Microaneurysm)', color: '#EF4444' },
    hemorrhage: { label: 'Xuất huyết (Hemorrhage)', color: '#DC2626' },
    hardExudate: { label: 'Rỉ dịch cứng (Hard Exudate)', color: '#F59E0B' },
    softExudate: { label: 'Rỉ dịch mềm (Soft Exudate)', color: '#EAB308' }
  }
}
