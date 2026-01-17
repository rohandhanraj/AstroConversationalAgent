"""
This module is used as the entry point for running the application. 
It initializes and starts the application.

`Author` :: Rohan Dhanraj Yadav
`Email` :: rohan.dhanraj.y@gmail.com
"""


import uvicorn

if __name__ == '__main__':
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
        )