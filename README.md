
<div align="center">
   <h1><strong>Chatbot PDF</strong></h1>
</div>

Chatbot para hacer consultas a documentos PDF usando LLM y RAG.

---

Use el LLM *gemma-2-2b-it* de Google, un modelo simple y ligero que cumple con la funcion.


![imagen](https://i.imgur.com/TCwhKss.png)

*Por el momento el modelo devuelve la respuesta en ingles.*

---

<h2>VERSION DEMO</h2>  

La generación de respuestas puede tardar un poco ya que tuvo que ser adaptada a CPU.

Puedes probarlo aqui ->  [![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Space-blue?logo=huggingface)](https://huggingface.co/spaces/M4tuuc/RAG_PDF)    
_proximamente disponible en gpu_

---

## Como usarlo localmente (SOLO SI HAY GPU DISPONIBLE)
   _Presionamos WINDOWS+R para y ejecutamos el comando cmd_
   
   _Al abrir nuestra terminal navegamos hasta el directorio donde vamos a guardar nuestro programa._

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

**Usar Conda como entorno(opcional)**
```bash
conda create -n myenv python=3.11

conda activate myenv

```


3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Corremos el programa**
```bash
streamlit run src/app.py
```
Abrir navegador: http://localhost:8501

Para finalizar el programa basta con apretar CTRL+C dentro de la terminal

---





## 🤝 Contributing

Feel free to submit issues and enhancement requests.


