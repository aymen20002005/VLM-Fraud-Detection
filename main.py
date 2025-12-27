from fraud_detector import detect_fraud

if __name__ == "__main__":
    document_text = """
    Insurance Claim Document
    Policy Number: CA-984563
    Claim Date: 15/02/2024
    Incident Date: 20/02/2024
    Claim Amount: 15000 EUR
    Description: Vehicle accident with no third party.
    Signature: Jhon Doe
    """

    result = detect_fraud(document_text)

    print("\n=== FRAUD DETECTION RESULT ===")
    for k, v in result.items():
        print(f"{k}: {v}")
