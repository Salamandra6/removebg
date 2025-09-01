import io
import zipfile
from typing import Dict, Tuple

import streamlit as st
from PIL import Image
from rembg import remove, new_session
from streamlit_drawable_canvas import st_canvas
import numpy as np


# ---------------------------
# Configuración de la página
# ---------------------------
st.set_page_config(page_title="Quitar fondo → Fondo blanco", page_icon="🖼️", layout="wide")
st.title("🖼️ Quitar fondo y refinar con pincel")
st.caption("IA para quitar fondo + refinado manual: pinta verde para CONSERVAR y rojo para ELIMINAR.")

# ---------------------------
# Estado
# ---------------------------
if "refine" not in st.session_state:
    # Guardamos pinceladas por archivo: { filename: {"keep": bool mask, "remove": bool mask} }
    st.session_state.refine: Dict[str, Dict[str, np.ndarray]] = {}

# ---------------------------
# Carga diferida del modelo
# ---------------------------
@st.cache_resource(show_spinner=False)
def _create_session(model_name: str = "u2net"):
    return new_session(model_name)

def get_session():
    return _create_session("u2net")

@st.cache_data(show_spinner=False)
def get_rgba_and_mask(file_bytes: bytes, model_name: str = "u2net") -> Tuple[Image.Image, Image.Image]:
    """
    Ejecuta rembg una sola vez y devuelve:
      - fg_rgba: foreground con alfa (PIL RGBA)
      - mask_L: máscara alfa (PIL L: 0-255)
    """
    session = _create_session(model_name)
    out_bytes = remove(file_bytes, session=session)  # RGBA con alfa
    fg_rgba = Image.open(io.BytesIO(out_bytes)).convert("RGBA")
    mask_L = fg_rgba.split()[3]  # canal alfa
    return fg_rgba, mask_L

def compose_on_background(orig_rgb: Image.Image, mask_L: Image.Image, bg_rgb: Tuple[int, int, int], max_width: int) -> bytes:
    """
    Compone la imagen original (RGB) sobre un color sólido usando la máscara dada.
    Devuelve PNG en bytes.
    """
    # Redimensionar si se pide (manteniendo proporción)
    if isinstance(max_width, int) and max_width > 0:
        w, h = orig_rgb.size
        if w > max_width:
            new_h = int(h * (max_width / w))
            orig_rgb = orig_rgb.resize((max_width, new_h), Image.LANCZOS)
            mask_L = mask_L.resize((max_width, new_h), Image.NEAREST)

    bg = Image.new("RGB", orig_rgb.size, bg_rgb)
    bg.paste(orig_rgb, mask=mask_L)

    out_buf = io.BytesIO()
    bg.save(out_buf, format="PNG")
    out_buf.seek(0)
    return out_buf.read()

def np_bool_mask(shape_hw):
    return np.zeros(shape_hw, dtype=bool)

# ---------------------------
# Sidebar (opciones)
# ---------------------------
st.sidebar.header("Opciones")
use_custom = st.sidebar.toggle("Usar color de fondo personalizado", value=False)
bg_color = (255, 255, 255)
if use_custom:
    picked = st.sidebar.color_picker("Color de fondo", "#FFFFFF")
    bg_color = tuple(int(picked.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

max_width = st.sidebar.number_input(
    "Redimensionar ancho máximo (px, 0 = sin cambio)",
    min_value=0, max_value=8000, value=0, step=50,
    help="Si es mayor que 0, redimensiona manteniendo proporción."
)

st.sidebar.info("Si la IA 'muerde' hombros/pelo, usa el refinado con pincel en la tarjeta de cada imagen.")

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
    for file in uploaded_files:
        # Leer bytes de manera segura
        try:
            file_bytes = file.getvalue()
        except Exception as e:
            st.error(f"No pude leer {file.name}: {e}")
            continue

        # Validar imagen
        try:
            orig_pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            orig_pil.load()
        except Exception as e:
            st.error(f"Archivo inválido o no soportado ({file.name}): {e}")
            continue

        # Ejecutar rembg una vez y obtener máscara
        with st.spinner(f"Procesando {file.name}… (la primera imagen puede tardar por carga del modelo)"):
            fg_rgba, mask_L = get_rgba_and_mask(file_bytes)  # cacheado

        # Composición inicial con la máscara de IA
        out_bytes = compose_on_background(orig_pil, mask_L, bg_color, max_width)
        res_pil = Image.open(io.BytesIO(out_bytes)); res_pil.load()

        # Mostrar tarjeta
        st.markdown(f"### 📷 {file.name}")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(orig_pil, caption="Original", use_column_width=True)
        with c2:
            st.image(res_pil, caption=f"Resultado (fondo {'personalizado' if use_custom else 'blanco'})", use_column_width=True)
            st.download_button(
                label="⬇️ Descargar PNG",
                data=out_bytes,
                file_name=f"bg_{file.name.rsplit('.', 1)[0]}.png",
                mime="image/png",
                use_container_width=True
            )

        # ---------------------------
        # Refinado con pincel
        # ---------------------------
        with st.expander("✍️ Refinar máscara (pincel: verde = CONSERVAR, rojo = ELIMINAR)"):
            # Inicializar buffers de refinado por archivo
            h, w = orig_pil.size[1], orig_pil.size[0]
            if file.name not in st.session_state.refine:
                st.session_state.refine[file.name] = {
                    "keep": np_bool_mask((h, w)),
                    "remove": np_bool_mask((h, w)),
                }

            br_col1, br_col2, br_col3 = st.columns([1, 1, 1])
            with br_col1:
                brush = st.slider("Tamaño pincel", 5, 120, 40, step=5)
            with br_col2:
                mode = st.radio("Modo de pincel", ["Conservar (verde)", "Eliminar (rojo)"], horizontal=True)
            with br_col3:
                clear = st.button("🧽 Borrar pinceladas", key=f"clear_{file.name}")

            # Reset de pinceladas
            if clear:
                st.session_state.refine[file.name]["keep"][:] = False
                st.session_state.refine[file.name]["remove"][:] = False
                st.experimental_rerun()

            draw_color = "#00FF00" if "Conservar" in mode else "#FF0000"

            st.write("Dibuja sobre la imagen (usa el mouse o dedo).")
            # Usamos tamaño ORIGINAL para no tener que reescalar la máscara
            canvas_result = st_canvas(
                fill_color="rgba(0,0,0,0)",
                stroke_width=brush,
                stroke_color=draw_color,
                background_image=orig_pil,
                height=orig_pil.height,
                width=orig_pil.width,
                drawing_mode="freedraw",
                key=f"canvas_{file.name}"
            )

            # Tomar las pinceladas y actualizarlas en máscaras keep/remove
            if canvas_result.image_data is not None:
                arr = canvas_result.image_data  # (H, W, 4) RGBA, solo strokes
                arr = arr.astype(np.uint8)

                alpha = arr[:, :, 3] > 0

                # Detectar color de trazo por canales (verde y rojo puros)
                is_green = (arr[:, :, 1] > 200) & (arr[:, :, 0] < 50) & (arr[:, :, 2] < 50) & alpha
                is_red   = (arr[:, :, 0] > 200) & (arr[:, :, 1] < 50) & (arr[:, :, 2] < 50) & alpha

                ref = st.session_state.refine[file.name]
                # Acumular (no se pierden trazos previos)
                ref["keep"] |= is_green
                ref["remove"] |= is_red

            # Botón para aplicar refinado
            if st.button("✅ Aplicar refinado a esta imagen", key=f"apply_{file.name}"):
                ref = st.session_state.refine[file.name]

                # Máscara base de IA → bool
                base = np.array(mask_L, dtype=np.uint8)
                fg_bool = base >= 128

                # Aplicar correcciones
                fg_bool[ref["keep"]] = True
                fg_bool[ref["remove"]] = False

                refined_mask = (fg_bool * 255).astype(np.uint8)
                refined_mask_L = Image.fromarray(refined_mask, mode="L")

                # Re-componer y mostrar
                refined_bytes = compose_on_background(orig_pil, refined_mask_L, bg_color, max_width)
                refined_pil = Image.open(io.BytesIO(refined_bytes)); refined_pil.load()

                st.image(refined_pil, caption="Resultado refinado", use_column_width=True)
                st.download_button(
                    "⬇️ Descargar PNG refinado",
                    data=refined_bytes,
                    file_name=f"bg_refined_{file.name.rsplit('.', 1)[0]}.png",
                    mime="image/png",
                    use_container_width=True
                )

        st.divider()

else:
    st.info("Sube una o varias imágenes para comenzar. El modelo de IA se cargará al procesar la primera.")
