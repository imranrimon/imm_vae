import zipfile
import re
import sys
import os

def extract_text_from_pptx(path):
    text_content = []
    
    try:
        # PPTX is a zip file
        with zipfile.ZipFile(path, 'r') as z:
            # Find all slide xmls
            slides = [f for f in z.namelist() if f.startswith('ppt/slides/slide') and f.endswith('.xml')]
            # Sort them by number (slide1, slide2, ...)
            slides.sort(key=lambda x: int(re.search(r'slide(\d+)', x).group(1)))
            
            for slide in slides:
                xml_content = z.read(slide).decode('utf-8')
                # Extract text from <a:t> tags
                # This is a naive regex approach but works for mostly plain text extraction
                matches = re.findall(r'<a:t>(.*?)</a:t>', xml_content)
                slide_text = " ".join(matches)
                slide_num = re.search(r'slide(\d+)', slide).group(1)
                text_content.append(f"--- Slide {slide_num} ---\n{slide_text}\n")
                
        return "\n".join(text_content)
    except Exception as e:
        return f"Error reading PPTX: {e}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_pptx.py <path_to_pptx>")
        sys.exit(1)
        
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)
        
    print(extract_text_from_pptx(path))
