from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from predictor import Model_OCR
from PIL import Image
from typing import Annotated
import io

app = FastAPI()
model = Model_OCR('config/predict_cpu.yml')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.post("/uploadfiles/")
async def create_upload_files(
    files: Annotated[
        list[UploadFile], File(description="Multiple files as UploadFile")
    ],
):
    return {"filenames": [file.filename for file in files]}

@app.post('/ocr/')
async def predict_image(files: Annotated[list[UploadFile], File(description="Multiple files as UploadFile")]):
    #content = await file.read()
    contents = [await file.read() for file in files]
    pil_images = [Image.open(io.BytesIO((content))) for content in contents]
    text = [model.predict(pil_image) for pil_image in pil_images]
    return {'predict':text}
@app.get("/")
async def main():
    content = """
<body>
<form action="/ocr/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
<form action="/uploadfiles/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)