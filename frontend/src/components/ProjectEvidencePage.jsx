import React from 'react';
import {
  AlertTriangle,
  ArrowRight,
  BookOpenCheck,
  BrainCircuit,
  CheckCircle2,
  Cloud,
  Code2,
  Database,
  ExternalLink,
  FileText,
  FlaskConical,
  Layers3,
  LockKeyhole,
  Microscope,
  Scale,
  ShieldCheck,
  Stethoscope,
  UserCheck,
  Workflow,
} from 'lucide-react';

const CLINICAL_SOURCES = [
  {
    title: 'QĐ 2558/QĐ-BYT và QĐ 2557/QĐ-BYT',
    publisher: 'Cục Quản lý Khám, chữa bệnh — Bộ Y tế',
    description: 'Hướng dẫn quản lý bệnh võng mạc đái tháo đường và quy trình chụp ảnh đáy mắt không huỳnh quang.',
    href: 'https://kcb.vn/tai-lieu/huong-dan-chan-doan-dieu-tri-va-quan-ly-benh-vong-mac-dai-thao-duong-quyet-dinh-2558-qd-byt-va-quyet-dinh-2557-qd-byt-ng.html',
    tag: 'Việt Nam',
  },
  {
    title: 'International Clinical DR Severity Scale',
    publisher: 'Wilkinson và cộng sự, 2003',
    description: 'Cơ sở cho thang phân độ DR năm mức từ không thấy DR đến DR tăng sinh.',
    href: 'https://pubmed.ncbi.nlm.nih.gov/13129861/',
    tag: 'ICDR',
  },
  {
    title: 'IDRiD Dataset & Attention U-Net Lesion Segmentation',
    publisher: 'IEEE Transactions / Medical Image Analysis',
    description: 'Cơ sở khoa học phân đoạn vi phình mạch (MA), xuất huyết (HE) và tiết cứng (EX) định lượng tỷ lệ % diện tích.',
    href: 'https://idrid.grand-challenge.org/',
    tag: 'Lesion AI',
  },
  {
    title: 'Standards of Care in Diabetes — 2026',
    publisher: 'American Diabetes Association',
    description: 'Cơ sở tham khảo về sàng lọc võng mạc, quản lý yếu tố nguy cơ và chuyển chuyên khoa.',
    href: 'https://diabetesjournals.org/docm-care/article/doi/10.2337/doc26-a012/164621/Section-12-Retinopathy-Neuropathy-and-Foot-Care',
    tag: 'ADA 2026',
  },
  {
    title: 'RETFound',
    publisher: 'Nature, 2023',
    description: 'Cơ sở khoa học của backbone nền tảng ảnh võng mạc được dùng trong pipeline grading nghiên cứu.',
    href: 'https://www.nature.com/articles/s41586-023-06555-x',
    tag: 'AI nền tảng',
  },
];

const LEGAL_SOURCES = [
  {
    title: 'Luật Khám bệnh, chữa bệnh 15/2023/QH15',
    description: 'Khung trách nhiệm chuyên môn, quyền người bệnh và điều kiện hoạt động khám chữa bệnh; có hiệu lực từ 01/01/2024.',
    href: 'https://vanban.chinhphu.vn/?docid=207396&pageid=27160',
  },
  {
    title: 'Luật Bảo vệ dữ liệu cá nhân 91/2025/QH15',
    description: 'Khung pháp lý hiện hành về xử lý dữ liệu cá nhân từ 01/01/2026; dữ liệu sức khỏe cần được quản trị với mức bảo vệ phù hợp.',
    href: 'https://vanban.chinhphu.vn/?classid=1&docid=214590&pageid=27160&typegroup=',
  },
  {
    title: 'Nghị định 13/2023/NĐ-CP',
    description: 'Khung quy định được hồ sơ dự án viện dẫn về bảo vệ dữ liệu cá nhân; khi triển khai phải đối chiếu với Luật 91/2025/QH15 và văn bản hướng dẫn hiện hành.',
    href: 'https://vanban.chinhphu.vn/?classid=0&docid=207759&pageid=27160',
  },
  {
    title: 'Nghị định 98/2021/NĐ-CP và 07/2023/NĐ-CP',
    description: 'Khung quản lý trang thiết bị y tế. Dự án chưa tự kết luận phần mềm thuộc loại thiết bị hay mức phân loại nào.',
    href: 'https://vanban.chinhphu.vn/default.aspx?docid=204442&pageid=27160',
    secondaryHref: 'https://vanban.chinhphu.vn/?classid=1&docid=207543&orggroupid=2&pageid=27160',
  },
];

const RULES = [
  {
    trigger: 'Grade 4 — DR tăng sinh',
    output: 'Ưu tiên khẩn; chuyển chuyên khoa mắt tuyến tỉnh/trung ương trong vòng dưới 1 tháng.',
    basis: 'QĐ 2558/QĐ-BYT; ICDR',
    kind: 'clinical',
    code: 'clinical/analysis.py:71–74',
  },
  {
    trigger: 'Grade 3 — NPDR nặng',
    output: 'Ưu tiên khẩn; đánh giá không quá 3 tháng và sớm hơn theo bệnh cảnh.',
    basis: 'QĐ 2558/QĐ-BYT; ICDR',
    kind: 'clinical',
    code: 'clinical/analysis.py:75–78',
  },
  {
    trigger: 'Grade 2 — NPDR trung bình',
    output: 'Gợi ý 3–6 tháng; bác sĩ phải cá thể hóa và đánh giá giảm thị lực.',
    basis: 'Mốc bảo thủ của hệ thống',
    kind: 'operational',
    code: 'clinical/analysis.py:79–82',
  },
  {
    trigger: 'Grade 1 — NPDR nhẹ',
    output: 'Gợi ý 6–12 tháng; không phải lịch hẹn cứng.',
    basis: 'Mốc bảo thủ của hệ thống',
    kind: 'operational',
    code: 'clinical/analysis.py:83–86',
  },
  {
    trigger: 'Grade 0 — không thấy DR trên ảnh',
    output: 'Gợi ý khoảng 12 tháng; không được diễn giải thành không mắc đái tháo đường.',
    basis: 'ICDR; ADA 2026; quy tắc an toàn',
    kind: 'safety',
    code: 'clinical/analysis.py:87–90',
  },
  {
    trigger: 'Confidence < 0,70',
    output: 'Tăng ưu tiên bác sĩ đọc/chụp lại; tuyệt đối không xem là ngưỡng y khoa.',
    basis: 'Heuristic vận hành, cần calibration theo cơ sở',
    kind: 'operational',
    code: 'clinical/analysis.py:97–102',
  },
  {
    trigger: 'HbA1c > 8%',
    output: 'Chỉ gắn cờ để bác sĩ lưu ý; không cộng điểm nguy cơ hay tự đổi điều trị.',
    basis: 'Heuristic vận hành',
    kind: 'operational',
    code: 'clinical/analysis.py:108–110',
  },
  {
    trigger: 'Ảnh hỏng, < 512×512, quá sáng/tối hoặc tương phản rất thấp',
    output: 'Từ chối xử lý; ảnh qua cổng kỹ thuật vẫn cần người đọc xác nhận khả năng phân loại.',
    basis: 'Ngưỡng kỹ thuật của model, không phải QĐ 2557',
    kind: 'technical',
    code: 'clinical/quality.py:9–55',
  },
];

const PIPELINE = [
  { icon: FileText, title: 'Tiếp nhận', text: 'Nhân viên chọn hồ sơ bệnh nhân và tải ít nhất một ảnh mắt trái hoặc mắt phải.' },
  { icon: ShieldCheck, title: 'Làm sạch & quality gate', text: 'Backend giải mã, tái mã hóa PNG để bỏ metadata và kiểm tra kỹ thuật tối thiểu.' },
  { icon: BrainCircuit, title: 'AI theo từng mắt', text: 'Grading chạy qua checkpoint RETFound cục bộ; segmentation dùng adapter riêng khi được cấu hình.' },
  { icon: Code2, title: 'Quy tắc có kiểm soát', text: 'Backend tạo mức ưu tiên, khoảng theo dõi tham khảo, cờ an toàn và nguồn guideline.' },
  { icon: Cloud, title: 'Lưu ảnh', text: 'Ảnh đã làm sạch được tải lên Cloudinary; database chỉ lưu URL HTTPS và kết quả cấu trúc.' },
  { icon: UserCheck, title: 'Bác sĩ duyệt', text: 'Bác sĩ xác nhận hoặc sửa grade từng mắt, ghi nhận định và kế hoạch tái khám.' },
  { icon: LockKeyhole, title: 'Bệnh nhân xem', text: 'Kết quả chuyên môn chỉ mở cho bệnh nhân sau khi trạng thái lần khám là Reviewed.' },
];

const EVIDENCE_STATUS = [
  {
    status: 'implemented',
    title: 'Phân độ DR 0–4',
    text: 'Đã có inference cục bộ, lưu version/confidence/probabilities và bước bác sĩ duyệt.',
  },
  {
    status: 'implemented',
    title: 'Phân đoạn tổn thương (Lesion Segmentation)',
    text: 'Đã tích hợp hoàn chỉnh 3 mô hình Attention U-Net (MA, HE, EX) nạp cục bộ, tính % diện tích tổn thương và vẽ đè bản đồ Mask.',
  },
  {
    status: 'implemented',
    title: 'Gemini soạn báo cáo',
    text: 'Chỉ nhận trường lâm sàng đã whitelist, không nhận tên/mã/liên hệ; lỗi dịch vụ sẽ dùng rule summary cục bộ.',
  },
  {
    status: 'research',
    title: 'Semi-supervised & few-shot',
    text: 'Chỉ là pipeline nghiên cứu; chưa có đủ artifact, kiểm định nhiều domain/seed và chưa được phép đưa vào production.',
  },
  {
    status: 'gap',
    title: 'Bộ ảnh theo quy trình',
    text: 'Tài liệu mục tiêu yêu cầu hai trường ảnh/mắt, nhưng API hiện chỉ lưu một ảnh fundus cho mỗi mắt.',
  },
  {
    status: 'gap',
    title: 'Sẵn sàng triển khai thật',
    text: 'Chưa có validation đa trung tâm, calibration theo máy/cơ sở, hồ sơ phân loại thiết bị y tế và quy trình giám sát drift/sự cố hoàn chỉnh.',
  },
];

const DOCUMENT_GROUPS = [
  {
    title: 'Nền tảng dự án & thiết kế',
    files: ['CONTEXT.md', 'README.md', 'backend/README.md', 'frontend/README.md', 'database/README.md', 'design-system/dr-screening/MASTER.md'],
  },
  {
    title: 'Lâm sàng, API & vận hành',
    files: ['docs/api_contract.md', 'docs/api_contract_ai.md', 'docs/clinical_guidelines.md', 'docs/clinical_rules_traceability.md', 'docs/image_standards.md', 'docs/system_flow.md', 'docs/tv3_integration_status.md'],
  },
  {
    title: 'AI, nghiên cứu & lịch sử tích hợp',
    files: ['ai/README.md', 'ai/grading/README.md', 'ai/semi_supervised/README.md', 'docs/semi_supervised_research.md', 'docs/dr_screening_migration.md'],
  },
];

const LEGAL_CHECKLIST = [
  'Xác định vai trò xử lý dữ liệu, căn cứ xử lý và nội dung thông báo/đồng ý phù hợp.',
  'Phân quyền tối thiểu, nhật ký truy cập/chỉnh sửa/xuất dữ liệu và rà soát quyền định kỳ.',
  'Mã hóa khi truyền/lưu, quản lý khóa/bí mật và hợp đồng với Cloudinary, Gemini hoặc nhà xử lý khác.',
  'Chính sách thời hạn lưu, xóa, sao lưu, phục hồi và đáp ứng quyền của chủ thể dữ liệu.',
  'Đánh giá tác động xử lý dữ liệu, quy trình ứng phó vi phạm và thông báo sự cố theo yêu cầu áp dụng.',
  'Đánh giá mục đích sử dụng và phân loại phần mềm/trang thiết bị y tế trước triển khai thương mại hoặc lâm sàng.',
];

const STATUS_META = {
  implemented: { label: 'Đã tích hợp', icon: CheckCircle2 },
  partial: { label: 'Tích hợp một phần', icon: Layers3 },
  research: { label: 'Chỉ nghiên cứu', icon: FlaskConical },
  gap: { label: 'Còn thiếu', icon: AlertTriangle },
};

function ExternalSource({ source, legal = false }) {
  return (
    <article className={`evidence-source-card ${legal ? 'legal-source-card' : ''}`}>
      <div className="evidence-source-heading">
        <span>{source.tag || 'Văn bản chính thức'}</span>
        <a href={source.href} target="_blank" rel="noreferrer" aria-label={`Mở nguồn: ${source.title}`}>
          <ExternalLink size={16} aria-hidden="true" />
        </a>
      </div>
      <h3>{source.title}</h3>
      {source.publisher && <strong>{source.publisher}</strong>}
      <p>{source.description}</p>
      {source.secondaryHref && (
        <a className="evidence-inline-link" href={source.secondaryHref} target="_blank" rel="noreferrer">
          Xem văn bản sửa đổi 07/2023/NĐ-CP <ExternalLink size={13} aria-hidden="true" />
        </a>
      )}
    </article>
  );
}

export default function ProjectEvidencePage() {
  return (
    <div className="governance-page animated-fade-in">
      <header className="governance-hero">
        <div className="governance-hero-copy">
          <span className="governance-kicker"><BookOpenCheck size={15} aria-hidden="true" /> Hồ sơ minh bạch dự án</span>
          <h1>Pháp lý, cơ sở khoa học & truy vết quy tắc</h1>
          <p>
            Trang này giải thích hệ thống dựa vào đâu, backend đang áp dụng quy tắc nào,
            dữ liệu đi qua những dịch vụ nào và các điều kiện còn thiếu trước khi có thể triển khai lâm sàng thực tế.
          </p>
          <div className="governance-hero-tags" aria-label="Phạm vi hệ thống">
            <span>Hỗ trợ sàng lọc DR</span>
            <span>Bác sĩ quyết định cuối cùng</span>
            <span>Không chẩn đoán đái tháo đường từ ảnh</span>
          </div>
        </div>
        <aside className="governance-hero-note" aria-label="Tuyên bố sử dụng">
          <ShieldCheck size={25} aria-hidden="true" />
          <div>
            <strong>Prototype nghiên cứu — chưa phải chứng nhận tuân thủ</strong>
            <p>Các nguồn pháp lý được tổng hợp để thiết kế và rà soát. Trang này không thay thế tư vấn pháp lý, hồ sơ thiết bị y tế hoặc phê duyệt của cơ sở khám chữa bệnh.</p>
            <small>Đối chiếu nguồn chính thức ngày 18/07/2026</small>
          </div>
        </aside>
      </header>

      <nav className="governance-anchor-nav" aria-label="Mục lục trang minh bạch">
        <a href="#system-boundary">Phạm vi</a>
        <a href="#backend-rules">Quy tắc backend</a>
        <a href="#scientific-basis">Cơ sở khoa học</a>
        <a href="#legal-basis">Pháp lý</a>
        <a href="#data-flow">Luồng dữ liệu</a>
        <a href="#evidence-status">Trạng thái bằng chứng</a>
        <a href="#document-map">Tài liệu dự án</a>
      </nav>

      <section id="system-boundary" className="governance-section">
        <div className="governance-section-heading">
          <div><span>01 · Ranh giới sản phẩm</span><h2>Hệ thống làm gì — và không làm gì</h2></div>
          <Stethoscope size={24} aria-hidden="true" />
        </div>
        <div className="governance-boundary-grid">
          <article className="governance-boundary-card does">
            <h3><CheckCircle2 size={19} aria-hidden="true" /> Được thiết kế để</h3>
            <ul>
              <li>Phân độ DR 0–4 riêng từng mắt và hiển thị confidence/version model.</li>
              <li>Đưa bằng chứng tổn thương, cờ an toàn và mức ưu tiên cho bác sĩ rà soát.</li>
              <li>Cho bác sĩ sửa grade, ghi nhận định và quyết định lịch tái khám.</li>
              <li>Cho bệnh nhân xem kết quả sau khi bác sĩ xác nhận.</li>
            </ul>
          </article>
          <article className="governance-boundary-card does-not">
            <h3><AlertTriangle size={19} aria-hidden="true" /> Không được dùng để</h3>
            <ul>
              <li>Chẩn đoán hoặc loại trừ đái tháo đường từ ảnh fundus.</li>
              <li>Tự chỉ định anti-VEGF, laser, PRP, phẫu thuật hoặc thuốc.</li>
              <li>Thay thế khám trực tiếp, xét nghiệm hoặc quyết định của bác sĩ.</li>
            </ul>
          </article>
        </div>
      </section>

      <section id="backend-rules" className="governance-section">
        <div className="governance-section-heading">
          <div><span>02 · Truy vết triển khai</span><h2>Quy tắc đang chạy trong backend</h2></div>
          <Code2 size={24} aria-hidden="true" />
        </div>
        <p className="governance-section-intro">
          Bảng dưới phản ánh trực tiếp <code>backend/app/clinical/analysis.py</code> và <code>quality.py</code>.
          Nhãn “vận hành” hoặc “kỹ thuật” có chủ ý để không biến heuristic thành hướng dẫn y khoa.
        </p>
        <div className="governance-table-wrap" role="region" aria-label="Bảng truy vết quy tắc backend" tabIndex="0">
          <table className="governance-rules-table">
            <thead><tr><th>Điều kiện</th><th>Đầu ra hiện tại</th><th>Căn cứ / loại</th><th>Vị trí code</th></tr></thead>
            <tbody>
              {RULES.map((rule) => (
                <tr key={rule.trigger}>
                  <td><strong>{rule.trigger}</strong></td>
                  <td>{rule.output}</td>
                  <td><span className={`rule-kind rule-kind-${rule.kind}`}>{rule.kind}</span><p>{rule.basis}</p></td>
                  <td><code>{rule.code}</code></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section id="scientific-basis" className="governance-section">
        <div className="governance-section-heading">
          <div><span>03 · Nguồn lâm sàng & AI</span><h2>Cơ sở khoa học chính</h2></div>
          <Microscope size={24} aria-hidden="true" />
        </div>
        <div className="evidence-source-grid">
          {CLINICAL_SOURCES.map((source) => <ExternalSource key={source.title} source={source} />)}
        </div>
        <div className="governance-method-note">
          <BrainCircuit size={21} aria-hidden="true" />
          <div><strong>Đánh giá model phải tách khỏi căn cứ lâm sàng</strong><p>Việc dùng ICDR làm nhãn không chứng minh model đủ an toàn. Trước triển khai cần test set giữ kín theo bệnh nhân, QWK, macro-F1, độ nhạy từng lớp, calibration, confidence interval và đánh giá ngoài theo máy/cơ sở.</p></div>
        </div>
      </section>

      <section id="legal-basis" className="governance-section">
        <div className="governance-section-heading">
          <div><span>04 · Quản trị & trách nhiệm</span><h2>Khung pháp lý cần đối chiếu</h2></div>
          <Scale size={24} aria-hidden="true" />
        </div>
        <div className="evidence-source-grid legal-source-grid">
          {LEGAL_SOURCES.map((source) => <ExternalSource key={source.title} source={source} legal />)}
        </div>
        <article className="legal-checklist-card">
          <div className="legal-checklist-heading"><LockKeyhole size={22} aria-hidden="true" /><div><span>Checklist trước triển khai thật</span><h3>Các tài liệu hiện có chưa chứng minh đã tuân thủ</h3></div></div>
          <ul>{LEGAL_CHECKLIST.map((item) => <li key={item}><CheckCircle2 size={16} aria-hidden="true" />{item}</li>)}</ul>
        </article>
      </section>

      <section id="data-flow" className="governance-section">
        <div className="governance-section-heading">
          <div><span>05 · Kiến trúc đang chạy</span><h2>Luồng dữ liệu và điểm kiểm soát</h2></div>
          <Workflow size={24} aria-hidden="true" />
        </div>
        <div className="governance-pipeline">
          {PIPELINE.map((step, index) => {
            const Icon = step.icon;
            return (
              <article key={step.title} className="governance-pipeline-step">
                <div className="pipeline-step-number">{String(index + 1).padStart(2, '0')}</div>
                <span className="pipeline-step-icon"><Icon size={20} aria-hidden="true" /></span>
                <h3>{step.title}</h3><p>{step.text}</p>
                {index < PIPELINE.length - 1 && <ArrowRight className="pipeline-arrow" size={17} aria-hidden="true" />}
              </article>
            );
          })}
        </div>
        <div className="governance-data-cards">
          <article><Database size={20} aria-hidden="true" /><div><strong>Database</strong><p>Lưu hồ sơ, URL ảnh, kết quả AI, kết luận bác sĩ và lịch tái khám; không lưu binary ảnh.</p></div></article>
          <article><Cloud size={20} aria-hidden="true" /><div><strong>Dịch vụ ngoài</strong><p>Cloudinary nhận ảnh đã tái mã hóa; Gemini chỉ nhận trường lâm sàng whitelist để soạn dự thảo, không nhận tên/mã/liên hệ.</p></div></article>
          <article><UserCheck size={20} aria-hidden="true" /><div><strong>Human oversight</strong><p>Trạng thái AI_Analyzed chưa mở kết quả chuyên môn cho bệnh nhân; Reviewed mới là mốc công bố.</p></div></article>
        </div>
      </section>

      <section id="evidence-status" className="governance-section">
        <div className="governance-section-heading">
          <div><span>06 · Độ trưởng thành</span><h2>Trạng thái bằng chứng và khoảng trống</h2></div>
          <FlaskConical size={24} aria-hidden="true" />
        </div>
        <div className="evidence-status-grid">
          {EVIDENCE_STATUS.map((item) => {
            const meta = STATUS_META[item.status];
            const Icon = meta.icon;
            return (
              <article key={item.title} className={`evidence-status-card status-${item.status}`}>
                <div><span><Icon size={15} aria-hidden="true" />{meta.label}</span></div>
                <h3>{item.title}</h3><p>{item.text}</p>
              </article>
            );
          })}
        </div>
      </section>

      <section id="document-map" className="governance-section">
        <div className="governance-section-heading">
          <div><span>07 · Hồ sơ nguồn nội bộ</span><h2>Bản đồ 18 tài liệu đã tổng hợp</h2></div>
          <BookOpenCheck size={24} aria-hidden="true" />
        </div>
        <div className="document-group-grid">
          {DOCUMENT_GROUPS.map((group) => (
            <article key={group.title} className="document-group-card">
              <div><FileText size={19} aria-hidden="true" /><h3>{group.title}</h3><span>{group.files.length} tệp</span></div>
              <ul>{group.files.map((file) => <li key={file}><code>{file}</code></li>)}</ul>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}
