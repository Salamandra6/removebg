import io
import zipfile
from typing import List, Tuple

import streamlit as st
from PIL import Image
from rembg import remove, new_session


# ---------------------------
# Configuración de la página
# ---------------------------
st.set_page_config(
    page_title="Quitar fondo → Fondo blanco",
    page_icon="🖼️",
    layout="wide",
)

st.title("🖼️ Quitar fondo y poner fondo blanco")
st.caption("Sube tus imágenes, eliminamos el fondo con IA y lo reemplazamos por un color sólido (por defecto: blanco).")

# ---------------------------
# Recursos y caché
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_rembg_session():
    # Modelos posibles: "u2net", "u2netp", "isnet-general-use", "silueta"
    return new_session("u2net")

session = get_rembg_session()

@st.cache_data(show_spinner=False)
def compose_on_bg(file_bytes: bytes, bg_rgb: Tuple[int, int, int], max_width: int) -> bytes:
    """
    Quita el fondo con rembg y lo pega sobre un color sólido.
    Devuelve PNG en bytes.
    """
    # 1) Remover fondo → bytes RGBA con alfa
    no_bg_bytes = remove(file_bytes, session=session)

    # 2) Asegurar RGBA
    fg = Image.open(io.BytesIO(no_bg_bytes)).convert("RGBA")

    # 3) Redimensionar si corresponde (antes de componer)
    if isinstance(max_width, int) and max_width > 0:
        w, h = fg.size
        if w > max_width:
            new_h = int(h * (max_width / w))
            fg = fg.resize((max_width, new_h), Image.LANCZOS)

    # 4) Fondo sólido + composición con alfa
    bg = Image.new("RGB", fg.size, bg_rgb)
    bg.paste(fg, mask=fg.split()[3])

    # 5) Exportar a PNG en memoria
    out_buf = io.BytesIO()
    bg.save(out_buf, format="PNG")
    out_buf.seek(0)
    return out_buf.read()


# ---------------------------
# Sidebar (opciones)
# ---------------------------
st.sidebar.header("Opciones")
use_custom = st.sidebar.toggle("Usar color de fondo personalizado", value=False)
bg_color = (255, 255, 255)  # blanco por defecto
if use_custom:
    picked = st.sidebar.color_picker("Elige color de fondo", "#FFFFFF")
    bg_color = tuple(int(picked.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

max_width = st.sidebar.number_input(
    "Redimensionar ancho máximo (px, 0 = sin cambio)",
    min_value=0, max_value=8000, value=0, step=50,
    help="Si es mayor que 0, redimensiona manteniendo proporción."
)

st.sidebar.info("Sugerencia: para fotos de producto usa fondo **#FFFFFF** (blanco puro).")


# ---------------------------
# Carga de archivos
# ---------------------------
uploaded_files = st.file_uploader(
    "Arrastra o selecciona una o varias imágenes",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True
)

# ---------------------------
# Procesamiento
# ---------------------------
if uploaded_files:
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("Originales")
    with col_right:
        st.subheader(f"Procesadas (fondo {'personalizado' if use_custom else 'blanco'})")

    results_for_zip = []

    for file in uploaded_files:
        # Leer bytes de manera segura
        try:
            file_bytes = file.getvalue()  # más estable que .read()
        except Exception as e:
            st.error(f"No pude leer {file.name}: {e}")
            continue

        # Validar que realmente es imagen decodificable
        try:
            orig_pil = Image.open(io.BytesIO(file_bytes))
            orig_pil.load()  # fuerza la decodificación
        except Exception as e:
            st.error(f"Archivo inválido o no soportado ({file.name}): {e}")
            continue

        with st.spinner(f"Procesando {file.name}… (la primera imagen puede tardar por carga del modelo)"):
            out_bytes = compose_on_bg(file_bytes, bg_color, max_width)

        # Mostrar usando PIL (evita problemas de tipo)
        try:
            res_pil = Image.open(io.BytesIO(out_bytes))
            res_pil.load()
        except Exception as e:
            st.error(f"Error al abrir el resultado de {file.name}: {e}")
            continue

        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(orig_pil, caption=f"Original: {file.name}", use_column_width=True)
        with c2:
            st.image(res_pil, caption=f"Resultado: {file.name}", use_column_width=True)
            st.download_button(
                label="⬇️ Descargar PNG procesado",
                data=out_bytes,
                file_name=f"bg_{file.name.rsplit('.', 1)[0]}.png",
                mime="image/png",
                use_container_width=True
            )

        results_for_zip.append((f"bg_{file.name.rsplit('.', 1)[0]}.png", out_bytes))
        st.divider()

    if results_for_zip:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname, data in results_for_zip:
                zf.writestr(fname, data)
        zip_buf.seek(0)

        st.success(f"✅ Procesadas {len(results_for_zip)} imagen(es).")
        st.download_button(
            "⬇️ Descargar todas en .zip",
            data=zip_buf,
            file_name="imagenes_procesadas.zip",
            mime="application/zip",
            use_container_width=True
        )
else:
    st.info("Sube una o varias imágenes para comenzar. El modelo de IA se cargará cuando procese la primera imagen.")



