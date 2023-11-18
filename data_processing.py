import pandas as pd
import fitz  # Import PyMuPDF

def extract_data_from_pdf(pdf_file):
    pdf_data = []
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        page_text = page.get_text("text")
        pdf_data.append(page_text)

    return pdf_data

def preprocess_and_categorize(pdf_data):
    data = pd.DataFrame({'Description': pdf_data})
    data['Amount'] = data['Description'].str.extract(r'(\d+\.\d+)').astype(float)
    
    data['Category'] = data['Description'].apply(categorize_expense)
    
    return data

def categorize_expense(description):
    if "rent" in description.lower():
        return "Housing"
    elif "Zomato" in description.lower():
        return "Food"
    elif "Online Payment" in description.lower():
        return "Food"
    else:
        return "Other"
