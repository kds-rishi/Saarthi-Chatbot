from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import scraper, user_query

app = FastAPI(
    title="KDS Chatbot",
    description="AI assistant for resolving user's queries on KDS's website",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


app.include_router(scraper.router, prefix="/api", tags=["Scraper"])
app.include_router(user_query.router, prefix="/api", tags=["Scraper"])

@app.get("/")
def root():
   return {"status": "Server Running..."}
