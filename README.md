# MedBot - An AI Study Buddy

MedBot is an AI-powered study buddy designed to help medical students navigate through massive textbooks without having to manually parse through 15,000+ pages of content. Currently processes Harrison's Principles of Internal Medicine (21st Edition).

## Features

- **PDF Processing**: Loads and processes large medical textbooks
- **Text Cleaning**: Removes page numbers, headers, and formatting artifacts
- **Content Normalization**: Fixes hyphenated words and standardizes spacing
- **Document Preparation**: Prepares clean text for AI processing

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MedBot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your medical textbook**
   - Place Harrison's Principles of Internal Medicine PDF in the `data/` folder
   - The script will automatically detect the file

## Usage

### Preprocessing Medical Textbooks

Run the preprocessing script to clean and prepare your medical textbook:

```bash
python preprocess_and_convert_to_chroma.py
```

This will:
- Load the Harrison's medical textbook PDF
- Remove index and citation pages (processes 13,796 useful pages)
- Clean formatting artifacts (page numbers, headers, dividers)
- Normalize text content (fix hyphenated words, spacing)

### Output

The script processes the textbook and creates clean Document objects ready for AI applications.

## Project Structure

```
MedBot/
├── data/                          # Medical textbook PDFs
├── preprocess_and_convert_to_chroma.py  # Main preprocessing script
├── requirements.txt               # Python dependencies
├── README.md                     # Project documentation
└── Phase2_Exploration_Preprocessing_Summary.txt  # Week 2 submission
```

## Technical Details

- **Input**: 15,000+ page medical textbook PDF
- **Processing**: Filters to 13,796 useful pages
- **Cleaning**: Multi-stage text normalization pipeline
- **Output**: Clean Document objects for AI processing

## Dependencies

- `langchain_community`: Document loading and processing
- `pypdf`: PDF parsing backend

## Current Status

✅ PDF loading and text preprocessing  
✅ Document cleaning pipeline  
⏳ Vector database conversion (planned)  
⏳ AI chatbot interface (planned)  

## Contributing

This project is part of an AI development course. Currently in Phase 2: Exploration & Preprocessing.

## License

Educational project - see course guidelines.