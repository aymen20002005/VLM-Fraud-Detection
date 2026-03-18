import sys
from pathlib import Path
from fraud_detector import detect_fraud

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # PDF file path provided as argument
        pdf_path = sys.argv[1]
        
        if not Path(pdf_path).exists():
            print(f"Error: PDF file not found at {pdf_path}")
            sys.exit(1)
        
        print(f"Processing PDF: {pdf_path}")
        result = detect_fraud(pdf_path)

        print("\n=== FRAUD DETECTION RESULT ===")
        for k, v in result.items():
            print(f"{k}: {v}")
    else:
        print("Usage: python main.py <pdf_path>")
