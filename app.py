import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from fastapi import FastAPI, UploadFile, File
import uvicorn
import os
import requests
import shutil
from huggingface_hub import hf_hub_download

app = FastAPI()
STORAGE_DIR = "storage"
os.makedirs(STORAGE_DIR, exist_ok=True)

# Tenta carregar o Whisper.cpp (Performance), senão vai de Faster-Whisper (Compatibilidade)
try:
    from whisper_cpp_python import Whisper
    MODEL_TYPE = "cpp"
    # Definições para o modelo Small
    REPO_ID = "ggerganov/whisper.cpp"
    FILENAME = "ggml-small.bin"
    MODEL_PATH = os.path.join(STORAGE_DIR, FILENAME)
    print(f"--- Modo: Whisper.cpp (Alta Performance ARM) - Modelo: {FILENAME} ---")
except ImportError:
    from faster_whisper import WhisperModel
    MODEL_TYPE = "faster"
    MODEL_PATH = "large-v3-turbo" # O faster-whisper baixa por nome automaticamente
    print("--- Modo: Faster-Whisper (Compatibilidade Windows) ---")

# Inicialização do Modelo
if MODEL_TYPE == "cpp":
    # hf_hub_download garante que o arquivo de ~480MB esteja íntegro
    print(f"Verificando/Baixando modelo {FILENAME}...")
    hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=STORAGE_DIR,
        local_dir_use_symlinks=False
    )
    # Inicializa com 4 threads para o modelo Small (instância ARM aguenta bem)
    model = Whisper(model_path=MODEL_PATH, n_threads=4)
else:
    # O faster-whisper gerencia o download sozinho
    model = WhisperModel(MODEL_PATH, device="cpu", compute_type="int8")

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    temp_path = "temp.wav"
    with open(temp_path, "wb") as f:
        f.write(await audio.read())
    
    if MODEL_TYPE == "cpp":
        result = model.transcribe(temp_path)
        text = result["text"]
    else:
        segments, _ = model.transcribe(temp_path, beam_size=5)
        text = "".join([s.text for s in segments])
    
    os.remove(temp_path)
    return {"text": text, "engine": MODEL_TYPE}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)