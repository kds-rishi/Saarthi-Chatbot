from fastapi import APIRouter, Query
import pandas as pd
from utils.text_scraper import text_scraper
from utils.embeddings import store_embeddings
router = APIRouter()

@router.get("/scrape")
def scrape_api():
    cleaned_data = text_scraper()
    print(cleaned_data)
    df = pd.DataFrame(cleaned_data)

    print("Head -> ", df.head())

    df.to_csv("cleaned_scraped_data.csv", index=False, encoding="utf-8")
    print(
            f"Saved {len(df)} pages to cleaned_scraped_data.csv\n"
        )

    print("\nNow creating Embeddings\n")
    result = store_embeddings() 

    if result["success"]:
        print(
            f"Saved {len(df)} pages to cleaned_scraped_data.csv and created vector embeddings\n"
            f"Total Count: {result['total_count']}\nStored: {result['stored_count']}"
        )

    return {
        "message": f"Scraped {len(df)} pages successfully",
        "embedding_result": result
    }
