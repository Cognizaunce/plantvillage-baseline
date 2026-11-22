-- Pandoc filter: turn <div class="page-break"></div> in the HTML into a real page break in the .docx.
function Div(el)
  if el.classes:includes('page-break') then
    if FORMAT:match('docx') then
      return pandoc.RawBlock('openxml', '<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
    end
    return {}
  end
end

-- The HTML <title> is only for the browser tab; keep it out of the document body.
function Meta(meta)
  meta.title = nil
  return meta
end
