# Remove BG Streamlit

App de Streamlit para quitar el fondo de imágenes con IA (`rembg`) y colocar un color sólido (blanco por defecto). Soporta subida múltiple y descarga en ZIP.

## Requisitos

- Python 3.8+
- pip

## Uso local

```bash
# (opcional) crear venv
python -m venv .venv
# activar venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
