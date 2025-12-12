import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
import re
import tempfile
import os
from pylatex import Document, Section, Subsection, Command
from pylatex.utils import NoEscape

class PyLaTeXGenerator:
    def __init__(self, filename):
        # pylatex adds .pdf extension automatically, so remove it if present
        if filename.endswith('.pdf'):
            self.basename = filename[:-4]
        else:
            self.basename = filename
            
        self.doc = Document(self.basename)
        self.doc.preamble.append(Command('title', 'Generated Questions'))
        self.doc.preamble.append(Command('date', NoEscape(r'\today')))
        self.doc.append(NoEscape(r'\maketitle'))
    
    def add_header(self, text, level=1):
        # Clean text for LaTeX header
        # If text contains special chars, we might need to escape, but assuming headers are simple
        if level == 1:
            with self.doc.create(Section(text, numbering=False)):
                pass
        else:
             with self.doc.create(Subsection(text, numbering=False)):
                pass
                
    def add_text(self, text):
        # Add text as NoEscape to render LaTeX commands if present
        # We add a paragraph break
        self.doc.append(NoEscape(text))
        self.doc.append(NoEscape(r'\par'))
        self.doc.append(NoEscape(r'\vspace{0.5cm}'))

    def add_separator(self):
        self.doc.append(NoEscape(r'\hrule'))
        self.doc.append(NoEscape(r'\vspace{0.5cm}'))
        
    def save(self):
        try:
            # clean_tex=False keeps the .tex file which is useful if compilation fails
            self.doc.generate_pdf(clean_tex=False, compiler='pdflatex')
        except Exception as e:
            print(f"⚠️ PDF generation via PyLaTeX failed (likely missing pdflatex): {e}")
            print(f"📝 Generated .tex file at: {self.basename}.tex")

class PDFGenerator:
    def __init__(self, filename):
        self.filename = filename
        self.story = []
        
        self.use_pylatex = True
        if self.use_pylatex:
            self.pylatex_gen = PyLaTeXGenerator(filename)
        
        self.styles = getSampleStyleSheet()
        self.style_normal = self.styles['Normal']
        self.style_heading = self.styles['Heading1']
        self.style_heading2 = self.styles['Heading2']
        self.style_code = ParagraphStyle('Code', parent=self.styles['Normal'], fontName='Courier', fontSize=10)
        self.temp_files = []

    def __del__(self):
        # Attempt to clean up temp files
        for f in self.temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except:
                pass

    def add_header(self, text, level=1):
        if self.use_pylatex:
            self.pylatex_gen.add_header(text, level)
            return
            
        if level == 1:
            self.story.append(Paragraph(text, self.style_heading))
        else:
            self.story.append(Paragraph(text, self.style_heading2))
        self.story.append(Spacer(1, 12))

    def add_separator(self):
        if self.use_pylatex:
            self.pylatex_gen.add_separator()
            return
            
        self.story.append(Spacer(1, 12))
        self.story.append(Paragraph("_" * 60, self.style_normal))
        self.story.append(Spacer(1, 12))
    
    def add_text(self, text):
        if self.use_pylatex:
            self.pylatex_gen.add_text(text)
            return
        self.add_full_latex_response(text)

    def add_full_latex_response(self, text):
        # Split by double newlines to identify paragraphs or blocks
        blocks = text.split('\n')
        
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            
            latex_candidate = block.replace(" ", r"\ ")
            
            if (latex_candidate.startswith('$') and latex_candidate.endswith('$')) or \
               (latex_candidate.startswith(r'\[') and latex_candidate.endswith(r'\]')):
                   formula = latex_candidate
            else:
                formula = f"${latex_candidate}$"

            img_buf, w, h = self.render_latex_to_image(formula, fontsize=12)
            
            if img_buf:
                max_width = 6.5 * inch
                img_width = w * inch
                img_height = h * inch
                
                if img_width > max_width:
                    scale = max_width / img_width
                    img_width = max_width
                    img_height = img_height * scale
                
                img = RLImage(img_buf, width=img_width, height=img_height)
                self.story.append(img)
                self.story.append(Spacer(1, 6))
            else:
                self.add_text_with_latex(block)

    def save(self):
        if self.use_pylatex:
            self.pylatex_gen.save()
            return
            
        doc = SimpleDocTemplate(self.filename, pagesize=letter)
        doc.build(self.story)

    def render_latex_to_image(self, formula, fontsize=12):
        """Renders LaTeX formula to an image using matplotlib."""
        try:
            fig = plt.figure(figsize=(0.1, 0.1))  # Dummy size, will be resized
            
            render_text = formula.strip()
            if not render_text.startswith('$'):
                render_text = f"${render_text}$"
            
            text = fig.text(0, 0, render_text, fontsize=fontsize)
            
            # Save to buffer
            buf = io.BytesIO()
            
            # We need to draw the canvas to get the bbox
            fig.canvas.draw()
            
            # Get bounding box of the text
            bbox = text.get_window_extent()
            
            # Adjust figure size to fit text
            # Convert pixels to inches (matplotlib uses dpi=100 by default usually)
            dpi = fig.dpi
            width = bbox.width / dpi
            height = bbox.height / dpi
            
            # Add some padding
            fig.set_size_inches(width + 0.1, height + 0.1)
            
            # Reposition text
            text.set_position((0.05, 0.05))
            
            # Save
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05, dpi=300)
            plt.close(fig)
            
            buf.seek(0)
            return buf, width, height
        except Exception as e:
            print(f"Error rendering LaTeX '{formula}': {e}")
            plt.close(fig)
            return None, 0, 0

    def add_text_with_latex(self, text):
        """
        Parses text for LaTeX patterns and adds paragraphs or images.
        Supported patterns:
        - $$ ... $$ (Block math)
        - $ ... $ (Inline math)
        """
        
        # Regex for block math: $$...$$ or \[...\]
        # We split the text by these blocks.
        # using re.DOTALL to allow . to match newlines, equivalent to [\s\S]
        # Pattern for block math
        block_pattern = re.compile(r'(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\])')
        
        parts = block_pattern.split(text)
        
        for part in parts:
            if not part:
                continue
                
            # Check for block math markers
            # We check both raw string and escaped string just in case
            if part.startswith('$$') or part.startswith(r'\['):
                # It's a block math
                formula = part
                if formula.startswith('$$'):
                    formula = formula[2:-2]
                else:
                    # Remove \[ and \]
                    formula = formula[2:-2]
                
                img_buf, w, h = self.render_latex_to_image(formula, fontsize=14)
                if img_buf:
                    # Create ReportLab Image
                    img = RLImage(img_buf, width=w*inch, height=h*inch)
                    self.story.append(img)
                    self.story.append(Spacer(1, 6))
                else:
                    # Fallback to text
                    self.story.append(Paragraph(part, self.style_normal))
            else:
                # It's normal text (maybe with inline math)
                self._add_inline_latex_paragraph(part)

    def _add_inline_latex_paragraph(self, text):
        # Pattern for inline math: $...$ or \(...\) or fallback for \[...\] if missed
        # We use [\s\S] for dot to match newlines in case of multiline inline math
        inline_pattern = re.compile(r'(\$[^\$]+?\$|\\\(.+?\\\)|\\\[.+?\\\])', re.DOTALL)
        
        segments = inline_pattern.split(text)
        processed_text = ""
        
        for segment in segments:
            is_inline_math = False
            formula = ""
            fontsize = 10
            
            if segment.startswith('$') and segment.endswith('$') and len(segment) > 2:
                formula = segment[1:-1]
                is_inline_math = True
            elif segment.startswith(r'\(') and segment.endswith(r'\)') and len(segment) > 4:
                formula = segment[2:-2]
                is_inline_math = True
            elif segment.startswith(r'\[') and segment.endswith(r'\]') and len(segment) > 4:
                # Fallback for \[...\] found in inline text
                formula = segment[2:-2]
                is_inline_math = True
                fontsize = 12 # Slightly larger for display math used inline
                
            if is_inline_math:
                img_buf, w, h = self.render_latex_to_image(formula, fontsize=10)
                if img_buf:
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                        tmp.write(img_buf.getvalue())
                        tmp_path = tmp.name
                        self.temp_files.append(tmp_path)
                    
                    # Convert inches to points (1 inch = 72 points)
                    w_pts = w * 72
                    h_pts = h * 72
                    
                    # Use <img/> tag for inline image
                    # valign="middle" aligns the image vertically with text
                    processed_text += f'<img src="{tmp_path}" width="{w_pts}" height="{h_pts}" valign="middle"/>'
                else:
                    processed_text += segment
            else:
                # Escape XML characters for ReportLab
                segment = segment.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                # Replace newlines with <br/>
                segment = segment.replace('\n', '<br/>')
                processed_text += segment
        
        if processed_text.strip():
            self.story.append(Paragraph(processed_text, self.style_normal))
            self.story.append(Spacer(1, 6))

    def add_text(self, text):
        """Alias for add_text_with_latex to support simple text addition."""
        self.add_full_latex_response(text)

    def add_full_latex_response(self, text):
        """
        Attempts to render the entire text block as a sequence of LaTeX-rendered lines/paragraphs.
        This is aggressive and assumes the content is mathematical or LaTeX-compatible.
        It avoids searching for delimiters and tries to render lines directly.
        """
        # Split by double newlines to identify paragraphs or blocks
        blocks = text.split('\n')
        
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            
            # Try to render the whole block as LaTeX
            # We need to escape spaces to preserve them in math mode
            # And we wrap in $$ to allow for display math style which handles complex expressions better
            # NOTE: This is heuristic.
            
            # Escape text-mode spaces for mathtext
            latex_candidate = block.replace(" ", r"\ ")
            
            # Check if it already has delimiters (if so, use them, else add them)
            # But user said "don't search", implying "just render it".
            # However, if we wrap "$...$" around "$...$", it breaks.
            # So we strip existing delimiters if they look like they wrap the whole string.
            
            # Heuristic: if it starts/ends with $, don't add more.
            if (latex_candidate.startswith('$') and latex_candidate.endswith('$')) or \
               (latex_candidate.startswith(r'\[') and latex_candidate.endswith(r'\]')):
                   formula = latex_candidate
            else:
                formula = f"${latex_candidate}$"

            img_buf, w, h = self.render_latex_to_image(formula, fontsize=12)
            
            if img_buf:
                # Add as image
                # Ensure it fits on page? ReportLab Image scales if we tell it?
                # We just provide the size.
                # If width is > 6 inches, we might need to scale down.
                max_width = 6.5 * inch
                img_width = w * inch
                img_height = h * inch
                
                if img_width > max_width:
                    scale = max_width / img_width
                    img_width = max_width
                    img_height = img_height * scale
                
                img = RLImage(img_buf, width=img_width, height=img_height)
                self.story.append(img)
                self.story.append(Spacer(1, 6))
            else:
                # Fallback to normal text processing if LaTeX rendering fails
                # This handles cases where text has special chars like # or % that break LaTeX
                # But we still want to try inline math parsing just in case
                self.add_text_with_latex(block)

    def add_text_with_latex(self, text):
        """
        Parses text for LaTeX patterns and adds paragraphs or images.
        Supported patterns:
        - $$ ... $$ (Block math)
        - $ ... $ (Inline math) - Note: handling inline math in ReportLab is tricky.
          We might have to render the whole paragraph or split it.
          For simplicity, we'll treat $$...$$ as block images and $...$ as inline images if possible,
          or just keep $...$ as text if it's too hard to align.
          
          Actually, ReportLab doesn't support inline images easily within a Paragraph flow in a standard way 
          that aligns perfectly with text baseline without custom Flowables.
          
          Let's try to handle block math $$...$$ or \[...\] as separate Image flowables.
          For inline math $...$, we might just leave it as text or try to render it.
          Given the user request "if there is any latex output... render it", I should try my best.
        """
        
        # Regex for block math: $$...$$ or \[...\]
        # We split the text by these blocks.
        # using re.DOTALL to allow . to match newlines, equivalent to [\s\S]
        # Pattern for block math
        block_pattern = re.compile(r'(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\])')
        
        parts = block_pattern.split(text)
        
        for part in parts:
            if not part:
                continue
                
            # Check for block math markers
            # We check both raw string and escaped string just in case
            if part.startswith('$$') or part.startswith(r'\['):
                # It's a block math
                formula = part
                if formula.startswith('$$'):
                    formula = formula[2:-2]
                else:
                    # Remove \[ and \]
                    formula = formula[2:-2]
                
                img_buf, w, h = self.render_latex_to_image(formula, fontsize=14)
                if img_buf:
                    # Create ReportLab Image
                    img = RLImage(img_buf, width=w*inch, height=h*inch)
                    self.story.append(img)
                    self.story.append(Spacer(1, 6))
                else:
                    # Fallback to text
                    self.story.append(Paragraph(part, self.style_normal))
            else:
                # It's normal text (maybe with inline math)
                self._add_inline_latex_paragraph(part)

    def _add_inline_latex_paragraph(self, text):
        # Pattern for inline math: $...$ or \(...\) or fallback for \[...\] if missed
        # We use [\s\S] for dot to match newlines in case of multiline inline math
        inline_pattern = re.compile(r'(\$[^\$]+?\$|\\\(.+?\\\)|\\\[.+?\\\])', re.DOTALL)
        
        segments = inline_pattern.split(text)
        processed_text = ""
        
        for segment in segments:
            is_inline_math = False
            formula = ""
            fontsize = 10
            
            if segment.startswith('$') and segment.endswith('$') and len(segment) > 2:
                formula = segment[1:-1]
                is_inline_math = True
            elif segment.startswith(r'\(') and segment.endswith(r'\)') and len(segment) > 4:
                formula = segment[2:-2]
                is_inline_math = True
            elif segment.startswith(r'\[') and segment.endswith(r'\]') and len(segment) > 4:
                # Fallback for \[...\] found in inline text
                formula = segment[2:-2]
                is_inline_math = True
                fontsize = 12 # Slightly larger for display math used inline
                
            if is_inline_math:
                img_buf, w, h = self.render_latex_to_image(formula, fontsize=10)
                if img_buf:
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                        tmp.write(img_buf.getvalue())
                        tmp_path = tmp.name
                        self.temp_files.append(tmp_path)
                    
                    # Convert inches to points (1 inch = 72 points)
                    w_pts = w * 72
                    h_pts = h * 72
                    
                    # Use <img/> tag for inline image
                    # valign="middle" aligns the image vertically with text
                    processed_text += f'<img src="{tmp_path}" width="{w_pts}" height="{h_pts}" valign="middle"/>'
                else:
                    processed_text += segment
            else:
                # Escape XML characters for ReportLab
                segment = segment.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                # Replace newlines with <br/>
                segment = segment.replace('\n', '<br/>')
                processed_text += segment
        
        if processed_text.strip():
            self.story.append(Paragraph(processed_text, self.style_normal))
            self.story.append(Spacer(1, 6))

    def save(self):
        doc = SimpleDocTemplate(self.filename, pagesize=letter)
        doc.build(self.story)

