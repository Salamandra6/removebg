import io
import zipfile
from typing import Dict, Tuple

import numpy as np
import streamlit as st
from PIL import Image
from rembg import remove, new_session
from streamlit_drawable_canvas import st_canvas


# ──────────────────────────────────────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Quitar fondo → Fondo blanco", page_icon="🖼️", layout="wide")
st.title("🖼️ Quitar fondo y refinar con pincel")
st.caption("IA para quitar fondo + refinado manual: pinta verde para CONSERVAR y rojo para ELIMINAR.")


# ──────────────────────────────────────────────────────────────────────────────
# Estado para refinado
# ──────────────────────────────────────────────────────────────────────────────
if "refine" not in st.session_state:
    st.session_state.refine: Dict[str, Dict[str, np.ndarray]] = {}


# ──────────────────────────────────────────────────────────────────────────────
# Modelo (carga diferida y cacheado)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _create_session(model_name: str = "u2net"):
    return new_session(model_name)

@st.cache_data(show_spinner=False)
def get_rgba_and_mask(file_bytes: bytes, model_name: str = "u2net") -> Tuple[Image.Image, Image.Image]:
    """
    Ejecuta rembg y devuelve:
      - fg_rgba: imagen RGBA con alfa (PIL)
      - mask_L: canal alfa como máscara (PIL mode 'L')
    """
    session = _create_session(model_name)
    out_bytes = remove(file_bytes, session=session)
    fg_rgba = Image.open(io.BytesIO(out_bytes)).convert("RGBA")
    mask_L = fg_rgba.split()[3]
    return fg_rgba, mask_L

def compose_on_background(orig_rgb: Image.Image, mask_L: Image.Image,
                          bg_rgb: Tuple[int, int, int], max_width: int) -> bytes:
    """
    Pega orig_rgb sobre un color sólido usando mask_L. Devuelve PNG en bytes.
    """
    if isinstance(max_width, int) and max_width > 0:
        w, h = orig_rgb.size
        if w > max_width:
            new_h = int(h * (max_width / w))
            orig_rgb = orig_rgb.resize((max_width, new_h), Image.LANCZOS)
            mask_L = mask_L.resize((max_width, new_h), Image.NEAREST)

    bg = Image.new("RGB", orig_rgb.size, bg_rgb)
    bg.paste(orig_rgb, mask=mask_L)

    buf = io.BytesIO()
    bg.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
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
st.sidebar.info("Si la IA 'muerde' hombros/pelo, usa el refinado con pincel en cada imagen.")


# ──────────────────────────────────────────────────────────────────────────────
# Carga de archivos
# ──────────────────────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Arrastra o selecciona una o varias imágenes",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True
)


# ──────────────────────────────────────────────────────────────────────────────
# Procesamiento
# ──────────────────────────────────────────────────────────────────────────────
if uploaded_files:
    for file in uploaded_files:
        # Leer bytes y validar
        try:
            file_bytes = file.getvalue()
            orig_pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            orig_pil.load()
        except Exception as e:
            st.error(f"Archivo inválido o no soportado ({file.name}): {e}")
            continue

        # Ejecutar rembg (cacheado)
        with st.spinner(f"Procesando {file.name}… (la primera imagen puede tardar por carga del modelo)"):
            fg_rgba, mask_L = get_rgba_and_mask(file_bytes)

        # Resultado inicial
        out_bytes = compose_on_background(orig_pil, mask_L, bg_color, max_width)
        res_pil = Image.open(io.BytesIO(out_bytes)); res_pil.load()

        st.markdown(f"### 📷 {file.name}")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(orig_pil, caption="Original", use_column_width=True)
        with col2:
            st.image(res_pil, caption=f"Resultado (fondo {'personalizado' if use_custom else 'blanco'})", use_column_width=True)
            st.download_button(
                "⬇️ Descargar PNG",
                data=out_bytes,
                file_name=f"bg_{file.name.rsplit('.', 1)[0]}.png",
                mime="image/png",
                use_container_width=True
            )

        # ──────────────────────────────────────────────────────────────────────
        # EXPANDER: Refinado con pincel
        # ──────────────────────────────────────────────────────────────────────
        with st.expander("✍️ Refinar máscara (pincel: verde = CONSERVAR, rojo = ELIMINAR)"):
            # Tamaños
            orig_w, orig_h = orig_pil.size
            CANVAS_MAX_W = 1024
            if orig_w > CANVAS_MAX_W:
                canvas_w = CANVAS_MAX_W
                canvas_h = int(orig_h * (CANVAS_MAX_W / orig_w))
            else:
                canvas_w, canvas_h = orig_w, orig_h

            # Imagen de fondo del canvas (PIL RGB a la escala del lienzo)
            canvas_bg = orig_pil if (orig_w == canvas_w) else orig_pil.resize((canvas_w, canvas_h), Image.LANCZOS)
            canvas_bg_rgb = canvas_bg.convert("RGB")
            bg_np_for_diff = np.array(canvas_bg_rgb)  # usado para detectar trazos por diferencia

            # Estado por archivo a resolución del lienzo
            def zeros_hw(h: int, w: int) -> np.ndarray:
                return np.zeros((h, w), dtype=bool)

            key_base = file.name
            state_key = f"refine_{key_base}"
            if state_key not in st.session_state:
                st.session_state[state_key] = {
                    "keep": zeros_hw(canvas_h, canvas_w),
                    "remove": zeros_hw(canvas_h, canvas_w),
                }

            a, b, c = st.columns([1, 1, 1])
            with a:
                brush = st.slider("Tamaño pincel", 5, 120, 30, step=5)
            with b:
                mode = st.radio("Modo de pincel", ["Conservar (verde)", "Eliminar (rojo)"],
                                horizontal=True, index=0)
            with c:
                if st.button("🧽 Borrar pinceladas", key=f"clear_{key_base}"):
                    st.session_state[state_key]["keep"][:] = False
                    st.session_state[state_key]["remove"][:] = False
                    st.rerun()

            draw_color = "#00FF00" if "Conservar" in mode else "#FF0000"
            st.write("Dibuja sobre la imagen. Si el lienzo no aparece, reduce el tamaño del pincel.")

            # Canvas principal — usa PIL como background_image (no NumPy)
            canvas_result = st_canvas(
                fill_color="rgba(0,0,0,0)",
                stroke_width=int(brush),
                stroke_color=draw_color,
                background_color="#00000000",
                background_image=canvas_bg_rgb,   # PIL Image
                height=canvas_h,
                width=canvas_w,
                drawing_mode="freedraw",
                update_streamlit=True,
                display_toolbar=True,
                key=f"canvas_{key_base}",
            )

            # Extraer pinceladas: comparamos contra el fondo (diferencia RGB)
            if canvas_result.image_data is not None:
                arr = canvas_result.image_data.astype("uint8")  # (H,W,4)
                rgb = arr[:, :, :3]
                # Dónde hay diferencia vs la imagen de fondo → son trazos
                diff = np.any(rgb != bg_np_for_diff, axis=2)

                # Clasificar por color (verde / rojo) con margen
                is_green = diff & (rgb[:, :, 1] > rgb[:, :, 0] + 50) & (rgb[:, :, 1] > rgb[:, :, 2] + 50) & (rgb[:, :, 1] > 150)
                is_red   = diff & (rgb[:, :, 0] > rgb[:, :, 1] + 50) & (rgb[:, :, 0] > rgb[:, :, 2] + 50) & (rgb[:, :, 0] > 150)

                st.session_state[state_key]["keep"] |= is_green
                st.session_state[state_key]["remove"] |= is_red

            # Aplicar refinado
            if st.button("✅ Aplicar refinado a esta imagen", key=f"apply_{key_base}"):
                ref = st.session_state[state_key]

                # Reescalar (lienzo → original)
                def to_orig(mask_small: np.ndarray) -> Image.Image:
                    m = (mask_small * 255).astype("uint8")
                    m_img = Image.fromarray(m, mode="L")
                    if (canvas_w, canvas_h) != (orig_w, orig_h):
                        m_img = m_img.resize((orig_w, orig_h), Image.NEAREST)
                    return m_img

                keep_L_orig   = to_orig(ref["keep"])
                remove_L_orig = to_orig(ref["remove"])

                # Máscara base de IA a tamaño original
                base = np.array(mask_L.resize((orig_w, orig_h), Image.NEAREST), dtype="uint8")
                fg_bool = base >= 128

                # Aplica correcciones
                keep_bool   = np.array(keep_L_orig, dtype="uint8") >= 128
                remove_bool = np.array(remove_L_orig, dtype="uint8") >= 128
                fg_bool = (fg_bool | keep_bool) & (~remove_bool)

                refined_mask_L = Image.fromarray((fg_bool * 255).astype("uint8"), mode="L")

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

        # Canvas de prueba (NO dentro de expander para evitar anidar)
        show_debug = st.checkbox("🧪 Mostrar canvas de prueba", key=f"debug_toggle_{file.name}")
        if show_debug:
            st_canvas(
                fill_color="rgba(255,0,0,0.2)",
                stroke_width=20,
                stroke_color="#00FF00",
                background_color="#FFFFFF",
                height=250,
                width=350,
                drawing_mode="freedraw",
                update_streamlit=True,
                display_toolbar=True,
                key=f"debug_canvas_{file.name}",
            )

        st.divider()

else:
    st.info("Sube una o varias imágenes para comenzar. El modelo de IA se cargará al procesar la primera.")
