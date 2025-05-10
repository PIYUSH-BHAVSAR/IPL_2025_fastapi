from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
from last_try import run_ipl_analysis  # Ensure last_try.py is in the same directory

app = FastAPI(
    title="IPL 2025 Prediction API",
    description="Upload a points table CSV to get playoff predictions.",
    version="1.0.0"
)

@app.post("/ipl2025")
async def upload_points_table(file: UploadFile = File(...)):
    try:
        # Read uploaded CSV into DataFrame
        df = pd.read_csv(file.file)

        # Call your updated analysis function
        result = run_ipl_analysis(df)

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
