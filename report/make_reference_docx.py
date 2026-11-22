"""Build reference.docx: the Word style template Pandoc uses for the report.

It takes Pandoc's default reference.docx and restyles it to match the original Google Doc:
Ubuntu 11pt, line spacing 1.1, Letter page with 0.8in margins, 17pt section headings,
13pt bold sub-headings, and tables with 1pt black borders.

Only needs re-running if the styles below change:
    python make_reference_docx.py
"""
import re
import subprocess
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'reference.docx'
FONT = 'Ubuntu'

default = subprocess.run(['pandoc', '--print-default-data-file', 'reference.docx'],
                         capture_output=True, check=True).stdout
tmp = HERE / '_default_reference.docx'
tmp.write_bytes(default)

FONTS = f'<w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}" w:cs="{FONT}" w:eastAsia="{FONT}" />'


def set_style(xml, style_id, new):
    pattern = r'<w:style\b[^>]*w:styleId="%s".*?</w:style>' % style_id
    assert re.search(pattern, xml, re.S), style_id
    return re.sub(pattern, lambda m: new, xml, count=1, flags=re.S)


def heading(level, size_half_pt, bold, before, after):
    return f'''<w:style w:type="paragraph" w:styleId="Heading{level}">
    <w:name w:val="heading {level}" /><w:basedOn w:val="Normal" /><w:next w:val="BodyText" />
    <w:uiPriority w:val="9" /><w:qFormat />
    <w:pPr><w:keepNext /><w:keepLines /><w:spacing w:before="{before}" w:after="{after}" /><w:outlineLvl w:val="{level - 1}" /></w:pPr>
    <w:rPr>{FONTS}{'<w:b />' if bold else '<w:b w:val="0" />'}<w:color w:val="000000" /><w:sz w:val="{size_half_pt}" /><w:szCs w:val="{size_half_pt}" /></w:rPr>
  </w:style>'''


def patch_styles(xml):
    # Document defaults: Ubuntu 11pt, 5pt after each paragraph, line spacing 1.1
    xml = re.sub(r'<w:docDefaults>.*?</w:docDefaults>', lambda m: f'''<w:docDefaults>
    <w:rPrDefault><w:rPr>{FONTS}<w:sz w:val="22" /><w:szCs w:val="22" /><w:lang w:val="en-US" /></w:rPr></w:rPrDefault>
    <w:pPrDefault><w:pPr><w:spacing w:after="100" w:line="264" w:lineRule="auto" /></w:pPr></w:pPrDefault>
  </w:docDefaults>''', xml, count=1, flags=re.S)

    xml = set_style(xml, 'BodyText', '''<w:style w:type="paragraph" w:styleId="BodyText">
    <w:name w:val="Body Text" /><w:basedOn w:val="Normal" /><w:link w:val="BodyTextChar" /><w:qFormat />
    <w:pPr><w:spacing w:before="0" w:after="100" /></w:pPr>
  </w:style>''')
    xml = set_style(xml, 'Compact', '''<w:style w:type="paragraph" w:customStyle="1" w:styleId="Compact">
    <w:name w:val="Compact" /><w:basedOn w:val="BodyText" /><w:qFormat />
    <w:pPr><w:spacing w:before="0" w:after="20" /></w:pPr>
  </w:style>''')

    # Headings: h2 = 17pt regular, h3 = 13pt bold, both black (as in the Google Doc)
    xml = set_style(xml, 'Heading1', heading(1, 34, False, 200, 60))
    xml = set_style(xml, 'Heading2', heading(2, 34, False, 200, 60))
    xml = set_style(xml, 'Heading3', heading(3, 26, True, 120, 40))
    for level in (1, 2, 3):  # drop the linked character styles that carry the default blue
        xml = re.sub(r'<w:style\b[^>]*w:styleId="Heading%dChar".*?</w:style>' % level, '', xml, count=1, flags=re.S)

    # Tables: 1pt black grid, bold header row
    border = ''.join(f'<w:{side} w:val="single" w:sz="8" w:space="0" w:color="000000" />'
                     for side in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'))
    xml = set_style(xml, 'Table', f'''<w:style w:type="table" w:default="1" w:styleId="Table">
    <w:name w:val="Table" /><w:basedOn w:val="TableNormal" /><w:qFormat />
    <w:tblPr>
      <w:tblInd w:w="0" w:type="dxa" />
      <w:tblBorders>{border}</w:tblBorders>
      <w:tblCellMar>
        <w:top w:w="60" w:type="dxa" /><w:left w:w="100" w:type="dxa" />
        <w:bottom w:w="60" w:type="dxa" /><w:right w:w="100" w:type="dxa" />
      </w:tblCellMar>
    </w:tblPr>
    <w:tblStylePr w:type="firstRow"><w:rPr><w:b /></w:rPr></w:tblStylePr>
  </w:style>''')

    # Captions: 10pt, upright; table captions left, figure captions centred
    xml = set_style(xml, 'Caption', '''<w:style w:type="paragraph" w:styleId="Caption">
    <w:name w:val="Caption" /><w:basedOn w:val="Normal" />
    <w:pPr><w:spacing w:before="40" w:after="60" /></w:pPr>
    <w:rPr><w:sz w:val="20" /><w:szCs w:val="20" /></w:rPr>
  </w:style>''')
    xml = set_style(xml, 'ImageCaption', '''<w:style w:type="paragraph" w:customStyle="1" w:styleId="ImageCaption">
    <w:name w:val="Image Caption" /><w:basedOn w:val="Caption" />
    <w:pPr><w:spacing w:before="20" w:after="100" /><w:jc w:val="center" /></w:pPr>
  </w:style>''')
    xml = set_style(xml, 'Figure', '''<w:style w:type="paragraph" w:customStyle="1" w:styleId="Figure">
    <w:name w:val="Figure" /><w:basedOn w:val="Normal" />
    <w:pPr><w:spacing w:before="60" w:after="0" /><w:jc w:val="center" /></w:pPr>
  </w:style>''')

    # Links: Google Docs blue, underlined
    xml = set_style(xml, 'Hyperlink', '''<w:style w:type="character" w:styleId="Hyperlink">
    <w:name w:val="Hyperlink" /><w:basedOn w:val="DefaultParagraphFont" />
    <w:rPr><w:color w:val="1155CC" /><w:u w:val="single" /></w:rPr>
  </w:style>''')

    # Custom styles referenced from the HTML through data-custom-style
    extra = '''<w:style w:type="paragraph" w:customStyle="1" w:styleId="TitleLine">
    <w:name w:val="TitleLine" /><w:basedOn w:val="Normal" /><w:qFormat />
    <w:pPr><w:spacing w:before="0" w:after="0" /><w:jc w:val="center" /></w:pPr>
    <w:rPr><w:sz w:val="28" /><w:szCs w:val="28" /></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:customStyle="1" w:styleId="TitleGap">
    <w:name w:val="TitleGap" /><w:basedOn w:val="TitleLine" /><w:qFormat />
    <w:pPr><w:spacing w:before="1600" w:after="0" /></w:pPr>
  </w:style>
  <w:style w:type="paragraph" w:customStyle="1" w:styleId="Note">
    <w:name w:val="Note" /><w:basedOn w:val="Normal" /><w:qFormat />
    <w:rPr><w:sz w:val="20" /><w:szCs w:val="20" /></w:rPr>
  </w:style>
</w:styles>'''
    assert xml.rstrip().endswith('</w:styles>')
    return xml.rstrip()[:-len('</w:styles>')] + extra


def patch_document(xml):
    # US Letter with 0.8in (57.6pt) margins, as in the Google Doc
    sect = ('<w:sectPr><w:pgSz w:w="12240" w:h="15840" />'
            '<w:pgMar w:top="1152" w:right="1152" w:bottom="1152" w:left="1152" w:header="720" w:footer="720" w:gutter="0" />'
            '</w:sectPr>')
    assert re.search(r'<w:sectPr\b.*?</w:sectPr>', xml, re.S)
    return re.sub(r'<w:sectPr\b.*?</w:sectPr>', lambda m: sect, xml, count=1, flags=re.S)


with zipfile.ZipFile(tmp) as zin, zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename == 'word/styles.xml':
            data = patch_styles(data.decode('utf-8')).encode('utf-8')
        elif item.filename == 'word/document.xml':
            data = patch_document(data.decode('utf-8')).encode('utf-8')
        zout.writestr(item, data)
tmp.unlink()
print(f'Wrote {OUT}')
