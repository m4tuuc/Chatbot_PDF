# RAG PDF
Programa con arquitectura RAG integrada, donde el usuario ingresa un pdf y este le devuelve toda la informacion que se le pida acerca del documento.


Puedes probarlo aqui ->  [![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Space-blue?logo=huggingface)](https://huggingface.co/spaces/M4tuuc/cannmodel_es)    
---

Use el LLM *gemma-2-2b-it* de Google, un modelo simple y ligero que cumple con la funcion.


![imagen](https://i.imgur.com/TCwhKss.png)

Por el momento el modelo devuelve la respuesta en ingles.

---

## Como usar
   *Vamos a nuestra terminal y hacemos lo siguiente*

1. **Clonar el repositorio**
```bash
git clone <your-repo-url>
cd RAG_PDF
```

2. **Crear un entorno virtual**
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Corremos el programa**
```bash
streamlit run src/app.py
```

---



## 🤝 Contributing

Feel free to submit issues and enhancement requests.


