## Path parameters without types
You can declare path "parameters" or "variables" with the same syntax used by Python format strings:
- main.py
```
from fastapi import FastAPI

app = FastAPI()


@app.get("/items/{item_id}")
async def read_item(item_id):
    return {"item_id": item_id}
```

- Browser: http://127.0.0.1:8002/items/12   

- Result: ` {"item_id":"12"} ` 

- **item_id** in decorator and function must be same.

- The order of the decorators matters.

- All the data validation is performed by **Pydantic** under the hood.

- Try str and int combination

http://localhost:8002/items/selam

` {"item_id":"selam"} `

http://localhost:8002/items/12+14

`{"item_id":"12+14"}`

## Path parameters with types
Just add int type to read_item function item_id parameter.
- main.py
```
from fastapi import FastAPI

app = FastAPI()


@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

- http://localhost:8002/items/hello

` {"detail":[{"loc":["path","item_id"],"msg":"value is not a valid integer","type":"type_error.integer"}]} `

## Documentation
And when you open your browser at [http://127.0.0.1:8002/docs](http://127.0.0.1:8002/docs), you will see an automatic, interactive, API documentation.

## Pydantic¶
All the data validation is performed under the hood by Pydantic, so you get all the benefits from it. And you know you are in good hands.

## Order matters
### The first one will always be used since the path matches first.
```commandline
@app.get("/users")
async def read_users():
    return ["Rick", "Morty"]


@app.get("/users")
async def read_users2():
    return ["Bean", "Elfo"]
```

