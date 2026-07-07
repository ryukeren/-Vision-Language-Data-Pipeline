import json
from pydantic import ValidationError
from app.schemas.extraction import ComprehensiveInvoiceSchema

def mock_vlm_extract(simulate_error: bool = False) -> str:
    """
    Simulates the JSON string output from a Vision-Language Model.
    """
    if simulate_error:
        # Messy data: Missing 'total_amount' and malformed 'tax_amount' string
        return json.dumps({
            "vendor_name": "TechCorp Global",
            "invoice_number": "INV-2026-X",
            "invoice_date": "2026-06-30",
            "line_items": [
                {
                    "description": "GPU Cluster Provisioning", 
                    "quantity": 4, 
                    "unit_price": "3500.00", 
                    "total_amount": 14000.0
                }
            ],
            "tax_amount": "Tax roughly $700.00" 
            # Missing total_amount entirely
        })
    else:
        # Clean data that conforms perfectly
        return json.dumps({
            "vendor_name": "TechCorp Global",
            "invoice_number": "INV-2026-X",
            "invoice_date": "2026-06-30",
            "line_items": [
                {
                    "description": "GPU Cluster Provisioning", 
                    "quantity": 4, 
                    "unit_price": 3500.0, 
                    "total_amount": 14000.0
                }
            ],
            "tax_amount": 700.0,
            "total_amount": 14700.0
        })

def run_extraction_simulation(simulate_error: bool = False):
    """
    Attempts to parse the VLM output through our strict Pydantic wall.
    """
    raw_vlm_output = mock_vlm_extract(simulate_error)
    print(f"--- Raw VLM Output ---\n{raw_vlm_output}\n")
    
    try:
        parsed_data = ComprehensiveInvoiceSchema.model_validate_json(raw_vlm_output)
        print("--- Extraction Successful ---")
        print(parsed_data.model_dump_json(indent=2))
        return parsed_data
        
    except ValidationError as e:
        print("--- Extraction Failed (Caught by Pydantic) ---")
        print(e.json(indent=2))
        return None

if __name__ == "__main__":
    print("=== RUNNING CLEAN SIMULATION ===")
    run_extraction_simulation(simulate_error=False)
    
    print("\n=== RUNNING MESSY SIMULATION ===")
    run_extraction_simulation(simulate_error=True)
