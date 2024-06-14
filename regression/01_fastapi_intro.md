# FastAPI Intro

FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints.

### Fast:
Very high performance, on par with NodeJS and Go (thanks to Starlette and Pydantic). One of the fastest Python frameworks available. 

### Fast to code:
Increase the speed to develop features by about 200% to 300%. *
### Fewer bugs: 
Reduce about 40% of human (developer) induced errors. *
### Intuitive: 
Great editor support. Completion everywhere. Less time debugging.
### Easy: 
Designed to be easy to use and learn. Less time reading docs.
### Short: 
Minimize code duplication. Multiple features from each parameter declaration. Fewer bugs.
### Robust: 
Get production-ready code. With automatic interactive documentation.
### Standards-based: 
Based on (and fully compatible with) the open standards for APIs: OpenAPI (previously known as Swagger) and JSON Schema.


## What is Typer?
Typer is FastAPI's little sibling. And it's intended to be the FastAPI of CLIs.

## Requirements of FastAPI
### 1. Python 3.7+
### 2. Starlette for the web parts.
### 3. Pydantic for the data parts.


## How to Install FastAPI and all dependencies?
` pip install "fastapi[all]==0.86.0" `

## What is the License of FastAPI?
This project is licensed under the terms of the **MIT** license.

## Features

### Based on open standards
- OpenAPI: for API creation, including declarations of path operations, parameters, body requests, security, etc.

### Automatic docs
#### Swagger UI
With interactive exploration, call and test your API directly from the browser.

####  ReDoc

### Just Modern Python

### Editor support
- Autocompletion works everywhere.

### Short
- It has sensible defaults for everything, with optional configurations everywhere. 

### Validation¶
- Validation for most (or all?) Python data types, including.

### Security and authentication
All the security schemes defined in OpenAPI, including:

- HTTP Basic.
- OAuth2 (also with JWT tokens). Check the tutorial on OAuth2 with JWT.
- API keys in:
  - Headers.
  - Query parameters.
  - Cookies, etc.

Plus all the security features from Starlette (including session cookies).

All built as reusable tools and components that are easy to integrate with your systems, data stores, relational and NoSQL databases, etc.

