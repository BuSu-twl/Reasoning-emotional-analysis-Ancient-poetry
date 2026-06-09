"""
将完整文档.md转换为Word文档(.docx)
需要安装: pip install python-docx
"""
import re
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

def convert_md_to_docx(md_file: str, docx_file: str):
    """
    将Markdown文件转换为Word文档
    
    Args:
        md_file: Markdown文件路径
        docx_file: 输出的Word文档路径
    """
    # 读取Markdown文件
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # 创建Word文档
    doc = Document()
    
    # 设置中文字体
    def set_chinese_font(run):
        """设置中文字体"""
        run.font.name = '宋体'
        run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
        run.font.name = 'Times New Roman'
    
    # 分割内容为行
    lines = md_content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        
        # 跳过空行（除非是分隔符）
        if not line.strip():
            i += 1
            continue
        
        # 检查是否是分隔线
        if line.strip() == '---':
            paragraph = doc.add_paragraph()
            paragraph_format = paragraph.paragraph_format
            paragraph_format.space_after = Pt(12)
            i += 1
            continue
        
        # 检查标题（# 开头）
        if line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            title_text = line.lstrip('#').strip()
            
            if level == 1:
                # 一级标题
                heading = doc.add_heading(title_text, level=1)
                for run in heading.runs:
                    set_chinese_font(run)
                    run.font.size = Pt(22)
                    run.bold = True
            elif level == 2:
                # 二级标题
                heading = doc.add_heading(title_text, level=2)
                for run in heading.runs:
                    set_chinese_font(run)
                    run.font.size = Pt(18)
                    run.bold = True
            elif level == 3:
                # 三级标题
                heading = doc.add_heading(title_text, level=3)
                for run in heading.runs:
                    set_chinese_font(run)
                    run.font.size = Pt(16)
                    run.bold = True
            else:
                # 四级及以下标题
                heading = doc.add_heading(title_text, level=min(level, 9))
                for run in heading.runs:
                    set_chinese_font(run)
                    run.bold = True
            
            i += 1
            continue
        
        # 检查是否是列表项（- 或 * 开头）
        if re.match(r'^[\-\*\+]\s+', line):
            list_text = re.sub(r'^[\-\*\+]\s+', '', line)
            paragraph = doc.add_paragraph(list_text, style='List Bullet')
            for run in paragraph.runs:
                set_chinese_font(run)
                run.font.size = Pt(12)
            i += 1
            continue
        
        # 检查是否是数字列表
        if re.match(r'^\d+\.\s+', line):
            list_text = re.sub(r'^\d+\.\s+', '', line)
            paragraph = doc.add_paragraph(list_text, style='List Number')
            for run in paragraph.runs:
                set_chinese_font(run)
                run.font.size = Pt(12)
            i += 1
            continue
        
        # 检查是否是代码块（```）
        if line.strip().startswith('```'):
            code_language = line.strip()[3:].strip()
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            i += 1  # 跳过结束的```
            
            # 添加代码块
            code_text = '\n'.join(code_lines)
            paragraph = doc.add_paragraph(code_text)
            paragraph.style = 'No Spacing'
            for run in paragraph.runs:
                set_chinese_font(run)
                run.font.name = 'Consolas'
                run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Consolas')
                run.font.size = Pt(10)
                run.font.color.rgb = RGBColor(0, 0, 0)
            
            # 设置代码块背景色（灰色）
            paragraph_format = paragraph.paragraph_format
            paragraph_format.left_indent = Inches(0.25)
            paragraph_format.right_indent = Inches(0.25)
            paragraph_format.space_before = Pt(6)
            paragraph_format.space_after = Pt(6)
            
            continue
        
        # 检查是否是表格（| 开头）
        if '|' in line and line.strip().startswith('|'):
            # 收集表格行
            table_rows = []
            while i < len(lines) and '|' in lines[i] and lines[i].strip().startswith('|'):
                row_text = lines[i].strip()
                # 跳过分隔行（|---|---|）
                if not re.match(r'^\|\s*[\-:]+\s*\|', row_text):
                    cells = [cell.strip() for cell in row_text.split('|')[1:-1]]
                    table_rows.append(cells)
                i += 1
            
            # 创建表格
            if table_rows:
                table = doc.add_table(rows=len(table_rows), cols=len(table_rows[0]))
                table.style = 'Light Grid Accent 1'
                
                for row_idx, row_data in enumerate(table_rows):
                    for col_idx, cell_data in enumerate(row_data):
                        if row_idx < len(table.rows) and col_idx < len(table.rows[row_idx].cells):
                            cell = table.rows[row_idx].cells[col_idx]
                            cell.text = cell_data
                            # 设置单元格字体
                            for paragraph in cell.paragraphs:
                                for run in paragraph.runs:
                                    set_chinese_font(run)
                                    run.font.size = Pt(10)
            continue
        
        # 检查是否是行内代码（`代码`）
        if '`' in line:
            paragraph = doc.add_paragraph()
            parts = re.split(r'(`[^`]+`)', line)
            for part in parts:
                if part.startswith('`') and part.endswith('`'):
                    # 代码
                    code_text = part[1:-1]
                    run = paragraph.add_run(code_text)
                    set_chinese_font(run)
                    run.font.name = 'Consolas'
                    run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Consolas')
                    run.font.size = Pt(11)
                    run.font.color.rgb = RGBColor(200, 0, 0)
                else:
                    # 普通文本
                    run = paragraph.add_run(part)
                    set_chinese_font(run)
                    run.font.size = Pt(12)
            
            i += 1
            continue
        
        # 检查是否是加粗文本（**文本**）
        if '**' in line:
            paragraph = doc.add_paragraph()
            parts = re.split(r'(\*\*[^\*]+\*\*)', line)
            for part in parts:
                if part.startswith('**') and part.endswith('**'):
                    # 加粗文本
                    bold_text = part[2:-2]
                    run = paragraph.add_run(bold_text)
                    set_chinese_font(run)
                    run.font.size = Pt(12)
                    run.bold = True
                else:
                    # 普通文本
                    run = paragraph.add_run(part)
                    set_chinese_font(run)
                    run.font.size = Pt(12)
            
            i += 1
            continue
        
        # 普通段落
        paragraph = doc.add_paragraph(line)
        for run in paragraph.runs:
            set_chinese_font(run)
            run.font.size = Pt(12)
        
        i += 1
    
    # 保存文档
    doc.save(docx_file)
    print(f"转换完成！Word文档已保存到: {docx_file}")


if __name__ == "__main__":
    # 输入和输出文件路径
    md_file = "完整文档.md"
    docx_file = "项目说明.docx"
    
    # 检查输入文件是否存在
    if not Path(md_file).exists():
        print(f"错误: 找不到文件 {md_file}")
        exit(1)
    
    try:
        convert_md_to_docx(md_file, docx_file)
    except ImportError:
        print("错误: 缺少 python-docx 库")
        print("请运行: pip install python-docx")
        exit(1)
    except Exception as e:
        print(f"转换失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

