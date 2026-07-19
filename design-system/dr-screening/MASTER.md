# DR Screening — Design System

> Trang cụ thể có thể ghi đè bằng `pages/[page-name].md`. Nếu không có, dùng các quy tắc trong tài liệu này.

**Product:** Clinical AI dashboard for diabetic-retinopathy screening  
**Audience:** Bác sĩ nhãn khoa và nhân viên sàng lọc  
**Design dials:** Variance 4/10 · Motion 3/10 · Density 7/10

## Design direction

- Trust-first, clinical, clear, data-dense.
- Light and dark mode both supported.
- Teal is the single interaction accent; diagnostic severity colors are reserved for clinical meaning.
- Prefer whitespace, dividers and hierarchy over decorative glass effects.
- Motion is subtle (160–300 ms) and must respect `prefers-reduced-motion`.

## Color tokens

| Role | Light | Dark |
|---|---:|---:|
| Primary | `#0F766E` | `#5EEAD4` |
| Primary hover | `#115E59` | `#99F6E4` |
| Background | `#F3F7F8` | `#0B171C` |
| Surface | `#FFFFFF` | `#122229` |
| Muted surface | `#EEF3F5` | `#1B3038` |
| Foreground | `#102A33` | `#F3F8F9` |
| Secondary text | `#405B65` | `#C2D1D6` |
| Muted text | `#607984` | `#9AAFB7` |
| Border | `#D8E2E6` | `#2C424B` |
| Destructive | `#B42318` | `#FDA29B` |
| Warning | `#B54708` | `#FEC84B` |
| Success | `#067647` | `#6CE9A6` |

## Typography

- **UI and headings:** Be Vietnam Pro, with Noto Sans and Segoe UI fallbacks.
- Base size: 15–16 px, line height at least 1.5.
- Headings use 600–700 weight and tight tracking; body copy remains 400–500.
- Numerical metrics use tabular numerals.

## Shape, spacing and elevation

- Radius: inputs/buttons 8 px, cards 12 px, large dialogs 18 px.
- Minimum touch target: 44×44 px for primary navigation and icon controls.
- Core spacing scale: 4, 8, 12, 16, 24, 32, 48 px.
- Cards use a 1 px border and subtle tinted shadow. Static cards do not translate on hover.

## Component rules

- Primary buttons use deep teal with white text; dark mode uses dark text on bright teal.
- Inputs always have visible labels or an accessible name, a 44 px minimum height and a focus ring.
- Navigation exposes `aria-current="page"`; mobile navigation becomes a dismissible drawer.
- Tables live in a labelled, keyboard-focusable horizontal scroll region on narrow screens.
- Async result panels use `aria-live="polite"`; errors use `role="alert"`.
- Severity colors must never be the only signal—always pair them with a label or number.

## Responsive rules

- Validate at 375, 768, 1024 and 1440 px.
- At ≤900 px, replace the fixed sidebar with a mobile header and drawer.
- At ≤1100 px, screening and diagnosis split views stack to one column.
- At ≤680 px, page actions, upload grids and clinical fields stack vertically.

## Pre-delivery checklist

- [ ] Text contrast is at least 4.5:1.
- [ ] Every interactive element is keyboard reachable with a visible focus state.
- [ ] Icon-only buttons have accessible names.
- [ ] Loading, empty, error and success states are present.
- [ ] No emoji is used as an interface icon.
- [ ] Reduced-motion preference is respected.
- [ ] No page-level horizontal overflow at 375 px.
- [ ] Tables remain usable through labelled horizontal scrolling.
