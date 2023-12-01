from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
import face_recognition
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from fastapi import Form
from pydantic import BaseModel
from util import load_image_contents
import os

app = FastAPI()

# Milvus connection parameters
MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = int(os.getenv('MILVUS_PORT'))
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
VECTOR_DIMENSION = 128

# Connect to Milvus
connections.connect(uri=MILVUS_HOST, token=MILVUS_TOKEN)

# Create Milvus collection
collection_name = "faces"
embedding = FieldSchema(
    name="embedding", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION)
primary_key = FieldSchema(
    name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
name = FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=191)
schema = CollectionSchema(
    fields=[embedding, primary_key, name], collection_name=collection_name)
collection = Collection(name=collection_name, schema=schema)
collection.create_index(field_name="embedding", index_params={
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}
})

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def encode_and_store_face(contents, name):
    image = load_image_contents(contents=contents)
    encoding = face_recognition.face_encodings(image)

    if encoding:
        encoding = encoding[0]
        # Convert encoding to Milvus format (list of float values)
        embedding = list(encoding)
        try:
            entities = [{"embedding": embedding, "name": name}]
            collection.insert(entities)
        except Exception as e:
            print(e)
            return JSONResponse(content={"error": str(e)}, status_code=500)

        return JSONResponse(content={"message": f"Face encoding for {name} stored successfully"}, status_code=200)
    else:
        return JSONResponse(content={"error": "No face detected in the image"}, status_code=400)


def recognize_faces(contents):
    image = load_image_contents(contents=contents)
    encoding = face_recognition.face_encodings(image)
    if encoding:
        encoding = encoding[0]
        # Convert encoding to Milvus format (list of float values)
        embedding = list(encoding)
        print(embedding)
        # Search for the closest match in Milvus

        try:
            entities = [{"embedding": embedding}]
            collection.load()
            results = collection.search(data=[embedding], anns_field="embedding", param={
                                        "metric_type": "L2",
                                        "offset": 0,
                                        "ignore_growing": False,
                                        "params": {"nprobe": 10}}, limit=1, output_fields=['name'], )
            if results:
                recognized_name = results[0][0].name
                return JSONResponse(content={"recognized_name": recognized_name}, status_code=200)
            else:
                return JSONResponse(content={"recognized_name": "Unknown"}, status_code=200)
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)
    else:
        return JSONResponse(content={"error": "No face detected in the image"}, status_code=400)


@app.get("/")
async def read_index():
    return FileResponse('index.html')


@app.post("/encode-face")
async def encode_face(name: str = Form(...), file: UploadFile = File(...)):
    contents = await file.read()

    return encode_and_store_face(contents, name)


@app.post("/recognize-face")
async def recognize_face(file: UploadFile = File(...)):
    contents = await file.read()
    return recognize_faces(contents)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
