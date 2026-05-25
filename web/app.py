from flask import Flask, render_template, request, jsonify
from pathlib import Path
from datetime import datetime
import base64, json, subprocess, threading, sys, yaml, os

BASE_DIR = Path(__file__).resolve().parent.parent

with open(BASE_DIR / "config" / "settings.yaml", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)

_web = _cfg.get("web", {})

app = Flask(__name__)

_training: dict[str, str] = {}

# No Vercel o filesystem é read-only; escrita vai para /tmp (efêmero)
_IS_VERCEL = bool(os.environ.get("VERCEL"))
_WRITE_ROOT = Path("/tmp") if _IS_VERCEL else BASE_DIR


def _users_dir() -> Path:
    d = _WRITE_ROOT / _cfg["paths"]["users_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


# ─── Handlers globais de erro → sempre retornam JSON ─────────────────────────

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"erro": str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"erro": str(e)}), 404


# ─── Páginas ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ─── API de usuários ──────────────────────────────────────────────────────────

@app.route("/api/usuario/criar", methods=["POST"])
def criar():
    body = request.get_json(force=True)
    nome = (body.get("nome") or "").strip()
    if not nome:
        return jsonify({"erro": "Nome é obrigatório"}), 400

    user_dir = _users_dir() / nome
    user_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "nome": nome,
        "setor": (body.get("setor") or "").strip(),
        "cargo": (body.get("cargo") or "").strip(),
        "matricula": (body.get("matricula") or "").strip(),
        "nivel_acesso": body.get("nivel_acesso") or "colaborador",
        "cadastrado_em": datetime.now().isoformat(),
    }
    (user_dir / "info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return jsonify({"ok": True})


@app.route("/api/usuario/foto", methods=["POST"])
def foto():
    body = request.get_json(force=True)
    nome = (body.get("nome") or "").strip()
    imagem = body.get("imagem") or ""

    user_dir = _users_dir() / nome
    if not user_dir.exists():
        return jsonify({"erro": "Usuário não encontrado"}), 404

    if "," in imagem:
        imagem = imagem.split(",", 1)[1]

    idx = len(sorted(user_dir.glob("foto_*.jpg"))) + 1
    (user_dir / f"foto_{idx:03d}.jpg").write_bytes(base64.b64decode(imagem))

    return jsonify({"ok": True, "total": idx})


@app.route("/api/usuario/treinar", methods=["POST"])
def treinar():
    body = request.get_json(force=True)
    nome = (body.get("nome") or "").strip()

    # No Vercel não há deepface/tensorflow — simula conclusão imediata
    if _IS_VERCEL:
        _training[nome] = "concluido"
        return jsonify({"ok": True, "aviso": "Modo demonstração: treinamento ML não roda no Vercel"})

    _training[nome] = "treinando"
    script = BASE_DIR / "scripts" / "modelo_treino.py"

    def _run():
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, cwd=str(BASE_DIR),
        )
        _training[nome] = "concluido" if result.returncode == 0 else "erro"

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/usuario/status-treino/<nome>")
def status_treino(nome):
    return jsonify({"status": _training.get(nome, "aguardando")})


@app.route("/api/usuarios")
def usuarios():
    result = []
    users_dir = _users_dir()
    if users_dir.exists():
        for d in sorted(users_dir.iterdir()):
            if d.is_dir():
                info_file = d / "info.json"
                if info_file.exists():
                    info = json.loads(info_file.read_text(encoding="utf-8"))
                    info["total_fotos"] = len(list(d.glob("foto_*.jpg")))
                    result.append(info)
    return jsonify(result)


if __name__ == "__main__":
    app.run(
        host=_web.get("host", "0.0.0.0"),
        port=_web.get("port", 5000),
        debug=_web.get("debug", False),
    )
