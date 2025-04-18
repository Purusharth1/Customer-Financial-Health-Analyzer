# API Documentation

The **Customer Financial Health Analyzer** provides a FastAPI-based API for processing bank statements and retrieving financial insights.

## API Endpoints

### 1. Process PDFs
- **Endpoint**: `/process`
- **Method**: `GET`
- **Description**: Triggers the workflow to parse PDFs, categorize transactions, generate visualizations, and create a financial narrative.
- **Response**:
  ```json
  {
    "success": true,
    "data": {
      "customer_info": [
        {
          "name": "John Doe",
          "email": "john.doe@example.com",
          "account_number": "1234567890",
          "city": "Mumbai",
          "state": "Maharashtra",
          "pdf_files": ["sample.pdf"]
        }
      ],
      "transactions": [
        [
          {
            "Date": "04/01/16",
            "Narration": "POS PURCHASE",
            "Reference_Number": "",
            "Value_Date": "",
            "Withdrawal_INR": 1000.0,
            "Deposit_INR": 0.0,
            "Closing_Balance_INR": 10000.0,
            "Source_File": "sample.pdf"
          }
        ]
      ]
    }
  }
  ```

## Running the API
Start the FastAPI server:

```bash
just start-api
```

Access Swagger UI for interactive testing at http://localhost:8000/docs.

## Next Steps
* [Installation Guide](installation.md)
* [Experience and Learnings](learnings.md)