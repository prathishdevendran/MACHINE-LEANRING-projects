from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from prediction_model import predictor
import base64
from PIL import Image
import io
import numpy as np

app = FastAPI(title="ML CATS VS DOGS")
templates = Jinja2Templates(directory="templates")


@app.get("/")
def welcome(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"request": request}
    )

@app.get("/upload/")
def redirect_to_home():
    return RedirectResponse(url="/", status_code=303)

@app.post("/upload/")
async def upload(request: Request, file: UploadFile = File(...)):
    try:
        # Read file into memory
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Resize to 160x160 (model input dimension)
        image = image.resize((160, 160))
        img_array = np.array(image, dtype=np.float32)
        img_batch = np.expand_dims(img_array, axis=0)

        result = predictor(img_batch)
        
        # Save back to memory for browser display
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        byte_im = buf.getvalue()

        # Encode base64
        encoded = base64.b64encode(byte_im).decode("utf-8")
        image_data = f"data:image/png;base64,{encoded}"
        
        return templates.TemplateResponse(
            request=request, 
            name="index.html", 
            context={"request": request, "image_data": image_data, "result": result}
        )
    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"request": request, "error": f"Error processing image: {str(e)}"}
        )
