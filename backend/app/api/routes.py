from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services.pipeline import TokenFlowPipeline
from app.services.history import history_store
import traceback

router = APIRouter()
pipeline = TokenFlowPipeline()

@router.get("/health")
def health():
    return {"status": "healthy", "service": "tokenflow"}

@router.post("/optimize")
async def optimize(
    file: UploadFile = File(...),
    query: str = Form(""),
    target_reduction: float = Form(0.45),
):
    if not file.filename:
        raise HTTPException(400, "A file is required.")
    if target_reduction < 0 or target_reduction > 0.9:
        raise HTTPException(400, "target_reduction must be between 0 and 0.9.")

    content = await file.read()
    try:
        result = pipeline.process(
            filename=file.filename,
            content=content,
            query=query,
            target_reduction=target_reduction,
        )
        history_store.add(result)
        return result
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        traceback.print_exc()  # TEMP: prints full traceback to the uvicorn console
        raise HTTPException(500, f"Optimization failed: {exc}")

@router.get("/history")
def history():
    return {"items": history_store.items()}
