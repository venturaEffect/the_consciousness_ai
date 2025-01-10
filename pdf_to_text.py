import PyPDF2
import sys

def convert_pdf_to_txt(pdf_path, txt_path):
    try:
        # Open PDF file in binary mode
        with open(pdf_path, 'rb') as pdf_file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from all pages
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # Write text to output file
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(text)
                
        print(f"Successfully converted {pdf_path} to {txt_path}")
        
    except FileNotFoundError:
        print(f"Error: File {pdf_path} not found")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    # Example usage
    convert_pdf_to_txt("INVE_MEM_2008_124320.pdf", "INVE_MEM_2008_124320.txt")