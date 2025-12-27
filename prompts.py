FRAUD_DETECTION_PROMPT = """
You are an expert in document fraud detection for financial and insurance institutions.

Analyze the document content below and determine whether it shows signs of fraud.

Tasks:
1. Identify suspicious elements (inconsistencies, altered fields, unrealistic values).
2. Assess the likelihood of fraud.
3. Provide a clear explanation.

Respond ONLY in valid JSON with the following structure:
{
  "fraud_probability": float (0 to 1),
  "fraud_label": "fraud" or "legitimate",
  "suspicious_elements": [string],
  "explanation": string
}

Document content:
-----------------
{document_text}
"""
