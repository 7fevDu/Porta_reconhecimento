import sys
from pathlib import Path

# Expõe o projeto inteiro ao path para que web/app.py resolva imports corretamente
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from web.app import app  # Vercel procura o objeto WSGI chamado 'app'
