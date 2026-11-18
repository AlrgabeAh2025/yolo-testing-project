"""Generate an Arabic Word report: accuracy results first, then model comparison."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "reports" / "مقارنة_نتائج_نماذج_YOLO.docx"


def set_rtl(paragraph) -> None:
    pPr = paragraph._p.get_or_add_pPr()
    bidi = OxmlElement("w:bidi")
    bidi.set(qn("w:val"), "1")
    pPr.append(bidi)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT


def add_para(doc: Document, text: str, *, bold: bool = False, size: int = 12, space_after: int = 8) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    run.font.name = "Arial"
    run._element.rPr.rFonts.set(qn("w:cs"), "Arial")
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    set_rtl(p)


def add_heading_ar(doc: Document, text: str, level: int = 1) -> None:
    h = doc.add_heading(level=level)
    run = h.add_run(text)
    run.font.name = "Arial"
    run._element.rPr.rFonts.set(qn("w:cs"), "Arial")
    if level == 1:
        run.font.size = Pt(18)
        run.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
    else:
        run.font.size = Pt(14)
        run.font.color.rgb = RGBColor(0x2E, 0x75, 0xB6)
    set_rtl(h)


def fill_table(table, rows: list[list[str]]) -> None:
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = ""
            p = cell.paragraphs[0]
            run = p.add_run(str(val))
            run.font.name = "Arial"
            run._element.rPr.rFonts.set(qn("w:cs"), "Arial")
            run.font.size = Pt(10)
            if r_idx == 0:
                run.bold = True
            set_rtl(p)


def main() -> None:
    doc = Document()
    for section in doc.sections:
        section.top_margin = Cm(2)
        section.bottom_margin = Cm(2)
        section.left_margin = Cm(2)
        section.right_margin = Cm(2)

    add_heading_ar(doc, "نتائج ومقارنة نماذج YOLO لتمييز الآفات الجلدية", 1)
    add_para(doc, "النماذج: YOLOv5su و YOLOv8s و YOLOv3-tinyu", size=12)
    add_para(doc, "البيانات: HAM10000 (حميد / خبيث) | مجموعة الاختبار: 1485 صورة", size=11)
    add_para(doc, "ملاحظة: النماذج الثلاثة دُرّبت وقُيّمت على نفس البيانات ونفس إعدادات التدريب لضمان عدالة المقارنة.", size=11)

    # =========================
    # PART 1: Accuracy results
    # =========================
    add_heading_ar(doc, "الجزء الأول: شرح نتائج الدقة", 1)

    add_heading_ar(doc, "1) ماذا تعني مقاييس الدقة المستخدمة؟", 2)
    add_para(doc, "• الدقة الكلية: نسبة الصور التي صُنّف صنفها النهائي بشكل صحيح من أصل كل صور الاختبار.")
    add_para(doc, "• دقة الحميد: نسبة صور الحميد التي عُرفت بشكل صحيح كحميد.")
    add_para(doc, "• دقة الخبيث: نسبة صور الخبيث التي عُرفت بشكل صحيح كخبيث.")
    add_para(doc, "• Precision للخبيث: من بين ما توقّعه النموذج خبيثًا، كم منها كان خبيثًا فعليًا.")
    add_para(doc, "• Recall للخبيث: من بين الحالات الخبيثة الحقيقية، كم نجح النموذج في اكتشافها.")

    add_heading_ar(doc, "2) نتائج الدقة على مستوى الصورة", 2)
    rows = [
        ["المقياس", "YOLOv5su", "YOLOv8s", "YOLOv3-tiny"],
        ["الدقة الكلية", "88.1%", "88.1%", "87.7%"],
        ["دقة الحميد", "92.1%", "92.7%", "95.5%"],
        ["دقة الخبيث", "71.7%", "68.9%", "54.9%"],
        ["Precision للخبيث", "68.6%", "70.1%", "74.4%"],
        ["Recall للخبيث", "71.7%", "68.9%", "54.9%"],
    ]
    t1 = doc.add_table(rows=len(rows), cols=4)
    t1.style = "Table Grid"
    fill_table(t1, rows)
    doc.add_paragraph()

    add_heading_ar(doc, "3) قراءة نتائج الدقة", 2)
    add_para(doc, "أولًا — الدقة الكلية:", bold=True)
    add_para(doc, "حققت النماذج الثلاثة دقة كلية متقاربة جدًا حول 88%. هذا يعني أن التصنيف النهائي للصورة كان صحيحًا في معظم الحالات على مجموعة الاختبار.")

    add_para(doc, "ثانيًا — دقة الحميد:", bold=True)
    add_para(doc, "دقة الحميد مرتفعة لدى الجميع: 92.1% في YOLOv5su، و92.7% في YOLOv8s، و95.5% في YOLOv3-tiny. أي أن النماذج تتعرف على الحالات الحميدة بنسبة عالية.")

    add_para(doc, "ثالثًا — دقة الخبيث:", bold=True)
    add_para(doc, "هنا يظهر التفاوت الأوضح بين النماذج: YOLOv5su بلغ 71.7%، وYOLOv8s بلغ 68.9%، بينما بلغ YOLOv3-tiny 54.9%. إذن الأداء على الصنف الخبيث هو الذي يميّز النتائج أكثر من الدقة الكلية.")

    add_para(doc, "رابعًا — Precision و Recall للخبيث:", bold=True)
    add_para(doc, "YOLOv5su يجمع بين Recall جيد نسبيًا (71.7%) وPrecision مقبول (68.6%). YOLOv8s قريب منه مع Recall أقل قليلًا. أما YOLOv3-tiny فيرفع Precision إلى 74.4% لكن Recall ينخفض إلى 54.9%، أي أنه يكتشف نسبة أقل من الحالات الخبيثة رغم ارتفاع نسبة الصحة ضمن ما يصنّفه خبيثًا.")

    add_heading_ar(doc, "4) نتائج الدقة من منظور الكشف (mAP)", 2)
    rows_map = [
        ["المقياس", "YOLOv5su", "YOLOv8s", "YOLOv3-tiny"],
        ["mAP50 على التحقق", "0.836", "0.831", "0.768"],
        ["mAP50 على الاختبار", "0.814", "0.813", "0.753"],
        ["استدعاء الخبيث (اختبار)", "0.748", "0.678", "0.587"],
        ["استدعاء الحميد (اختبار)", "0.948", "0.944", "0.956"],
    ]
    t2 = doc.add_table(rows=len(rows_map), cols=4)
    t2.style = "Table Grid"
    fill_table(t2, rows_map)
    doc.add_paragraph()
    add_para(
        doc,
        "من حيث mAP50، يتقارب YOLOv5su و YOLOv8s على الاختبار (0.814 و 0.813)، ويتقدم كلاهما على YOLOv3-tiny (0.753). كما يظهر أن استدعاء الخبيث أعلى عند YOLOv5su، بينما استدعاء الحميد مرتفع ومتقارب لدى النماذج الثلاثة.",
    )

    # =========================
    # PART 2: Comparison
    # =========================
    add_heading_ar(doc, "الجزء الثاني: المقارنة بين النماذج", 1)

    add_heading_ar(doc, "1) مقارنة سريعة شاملة", 2)
    rows_cmp = [
        ["جانب المقارنة", "YOLOv5su", "YOLOv8s", "YOLOv3-tiny", "النتيجة"],
        ["الدقة الكلية", "88.1%", "88.1%", "87.7%", "تعادل تقريبي بين v5 و v8"],
        ["دقة الحميد", "92.1%", "92.7%", "95.5%", "الأعلى: YOLOv3-tiny"],
        ["دقة الخبيث", "71.7%", "68.9%", "54.9%", "الأعلى: YOLOv5su"],
        ["Recall الخبيث", "71.7%", "68.9%", "54.9%", "الأعلى: YOLOv5su"],
        ["Precision الخبيث", "68.6%", "70.1%", "74.4%", "الأعلى: YOLOv3-tiny"],
        ["mAP50 اختبار", "0.814", "0.813", "0.753", "الأعلى: YOLOv5su"],
    ]
    t3 = doc.add_table(rows=len(rows_cmp), cols=5)
    t3.style = "Table Grid"
    fill_table(t3, rows_cmp)
    doc.add_paragraph()

    add_heading_ar(doc, "2) مقارنة حسب كل جانب", 2)

    add_para(doc, "أ) من حيث الدقة الكلية", bold=True)
    add_para(doc, "YOLOv5su و YOLOv8s متساويان عند 88.1%، وYOLOv3-tiny قريب جدًا عند 87.7%. الفرق بين النماذج في هذا الجانب ضئيل.")

    add_para(doc, "ب) من حيث دقة الحميد", bold=True)
    add_para(doc, "يتقدّم YOLOv3-tiny (95.5%) على YOLOv8s (92.7%) ثم YOLOv5su (92.1%). الفروقات موجودة لكنها أقل أهمية من فروقات الخبيث.")

    add_para(doc, "ج) من حيث دقة الخبيث وRecall", bold=True)
    add_para(doc, "يتصدر YOLOv5su بوضوح، يليه YOLOv8s بفارق بسيط، ثم YOLOv3-tiny بفارق أكبر. هذا الجانب هو الأكثر تمييزًا بين النماذج في نتائج المشروع.")

    add_para(doc, "د) من حيث Precision للخبيث", bold=True)
    add_para(doc, "الترتيب معكوس نسبيًا: YOLOv3-tiny الأعلى (74.4%)، ثم YOLOv8s (70.1%)، ثم YOLOv5su (68.6%).")

    add_para(doc, "هـ) من حيث mAP50", bold=True)
    add_para(doc, "YOLOv5su و YOLOv8s شبه متطابقين ويتفوقان على YOLOv3-tiny في كل من التحقق والاختبار.")

    add_heading_ar(doc, "3) خلاصة المقارنة", 2)
    add_para(doc, "• إذا اعتمدنا الدقة الكلية فقط: النتائج متقاربة بين النماذج الثلاثة.")
    add_para(doc, "• إذا اعتمدنا اكتشاف الخبيث (دقة الخبيث وRecall): الأفضل YOLOv5su ثم YOLOv8s ثم YOLOv3-tiny.")
    add_para(doc, "• إذا اعتمدنا Precision للخبيث: الأعلى YOLOv3-tiny.")
    add_para(doc, "• إذا اعتمدنا جودة الكشف العامة (mAP50): الأفضل YOLOv5su (مع تقارب كبير من YOLOv8s).")
    add_para(
        doc,
        "الترتيب النهائي حسب مجمل النتائج في هذا المشروع: YOLOv5su أولًا، ثم YOLOv8s، ثم YOLOv3-tiny.",
        bold=True,
    )

    add_heading_ar(doc, "الملفات المرجعية", 1)
    add_para(doc, "• أفضل وزن: runs/detect/yolov5su_balanced/weights/best.pt")
    add_para(doc, "• تقرير التعميم: runs/generalization/generalization_report.json")
    add_para(doc, "• ملخص المقاييس: runs/compare_summary.json")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
