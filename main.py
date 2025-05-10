from fastapi import FastAPI
from fastapi.responses import JSONResponse
from last_try import run_ipl_analysis  # Ensure last_try.py is in the root folder

app = FastAPI(
    title="IPL 2025 Prediction API",
    description="Predicts league standings and playoff outcomes based on historical data and simulation logic.",
    version="1.0.0"
)

@app.get("/ipl2025")
def get_ipl_prediction():
    try:
        result = run_ipl_analysis()
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
