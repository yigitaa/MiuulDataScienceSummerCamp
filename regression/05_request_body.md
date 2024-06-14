## POST -> CREATE

## PUT -> UPDATE

## PATCH -> Partial Update

## DELETE -> DELETE

## GET -> SELECT

## Request Body
- When you need to send data from a client (let's say, a browser) to your API, you send it as a request body.

- A request body is the data that is sent by the client to your API. 
- A response body is the data your API sends to the client.

- To declare a request body, you use Pydantic models with all their power and benefits.

- Let's create new customer
main.py customers endpoint
```
from pydantic import BaseModel


class Customer(BaseModel):
    CustomerID: int
    Gender: Optional[str] = None
    Age: Optional[int] = None
    AnnualIncome: Optional[float] = None
    SpendingScore: Optional[int] = None


@app.post("/customers")
async def create_customer(customer: Customer):
    return {"data": f"Customer {customer.CustomerID} is created."}
```
- Browser: http://127.0.0.1:8002/docs
- POST/customers
- Request body
```
{
  "CustomerID":1001,
  "Gender": "string",
  "Age": 0,
  "AnnualIncome": 0,
  "SpendingScore": 0
}
```

- Response body
```
{
  "CustomerID": 1001
}
```

