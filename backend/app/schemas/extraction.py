from datetime import date
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
import re

class InvoiceLineItem(BaseModel):
    description: str = Field(..., description="The name or description of the product or service.")
    quantity: int = Field(..., description="The number of units purchased.")
    unit_price: float = Field(..., description="The price per single unit.")
    total_amount: float = Field(..., description="The total cost for this line item (qty * unit_price).")    

    @field_validator("unit_price", "total_amount", mode="before")    
    @classmethod
    def clean_floats(cls, value):
        if isinstance(value, str):
            cleaned = re.sub(r'[^\d\.]', '', value)
            return float(cleaned) if cleaned else 0.0
        return value

class ComprehensiveInvoiceSchema(BaseModel):
    vendor_name: str = Field(..., description="Clean, authorized name of the issuing company.")
    invoice_number: str = Field(..., description="Unique identification string of the invoice.")
    invoice_date: date = Field(..., description="Date of issuance in YYYY-MM-DD format.")
    line_items: List[InvoiceLineItem] = Field(..., description="List of individual line items extracted from the table.")
    tax_amount: float = Field(..., description="Total tax applied.")
    total_amount: float = Field(..., description="Grand total amount due.")    

    @field_validator("tax_amount", "total_amount", mode="before")    
    @classmethod
    def clean_currency(cls, value):
        if isinstance(value, str):
            cleaned = re.sub(r'[^\d\.]', '', value)
            return float(cleaned) if cleaned else 0.0
        return value
