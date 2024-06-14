## Query parameters
When you declare other function parameters that are not part of the path parameters, they are automatically interpreted as "query" parameters.
- main.py customers endpoint
```
from fastapi import FastAPI

app = FastAPI()

fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]

# Query parameters
@app.get("/items/")
async def read_item(skip: int = 0, limit: int = 10):
    return fake_items_db[skip : skip + limit]
```
## Test with docs
- Browser: http://localhost:8002/docs#/default/read_item_items__get

## Test with browser
The query is the set of key-value pairs that go after the ? in a URL, separated by & characters.
- Browser: http://localhost:8002/items/?skip=0&limit=1

` [{"item_name":"Foo"}] `

- Browser: http://localhost:8002/items/?skip=1&limit=3

` [{"item_name":"Bar"},{"item_name":"Baz"}] `

- browser: http://localhost:8002/items/

`[{"item_name":"Foo"},{"item_name":"Bar"},{"item_name":"Baz"}]`

Default values are used.

## Optional parameters
