/* ─────────────────────────────────────────────────────────────────────────────
   ZanaflexAccess — Cadastro de Usuários
   Máquina de estados: etapa1 → etapa2 (câmera) → etapa3 (treino/resultado)
───────────────────────────────────────────────────────────────────────────── */

const MIN_PHOTOS    = 20;
const TARGET_PHOTOS = 30;

const state = {
    user:       {},
    photoCount: 0,
    stream:     null,
};

// ─── Utilitários ──────────────────────────────────────────────────────────────

const $ = (id) => document.getElementById(id);

async function api(path, body) {
    const res = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    return res.json();
}

// ─── Navegação entre etapas ───────────────────────────────────────────────────

function goToStep(n) {
    document.querySelectorAll(".step").forEach((el, i) => {
        el.classList.toggle("active", i + 1 === n);
    });

    [1, 2, 3].forEach((i) => {
        const nav  = $(`nav-${i}`);
        const line = i < 3 ? $(`line-${i}${i + 1}`) : null;
        nav.classList.remove("active", "done");
        if (i === n) nav.classList.add("active");
        if (i < n)  nav.classList.add("done");
        if (line) line.classList.toggle("done", i < n);
    });
}

// ─── ETAPA 1 — Formulário ─────────────────────────────────────────────────────

$("form-dados").addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = e.target;
    const errEl = $("form-error");
    errEl.classList.add("hidden");

    const nome  = f.nome.value.trim();
    const setor = f.setor.value.trim();

    if (!nome || !setor) {
        errEl.textContent = "Nome e Setor são obrigatórios.";
        errEl.classList.remove("hidden");
        return;
    }

    state.user = {
        nome,
        setor,
        cargo:         f.cargo.value.trim(),
        matricula:     f.matricula.value.trim(),
        nivel_acesso:  f.nivel_acesso.value,
    };

    try {
        const res = await api("/api/usuario/criar", state.user);
        if (res.erro) throw new Error(res.erro);
        await iniciarCaptura();
    } catch (err) {
        errEl.textContent = err.message || "Erro ao registrar usuário.";
        errEl.classList.remove("hidden");
    }
});

// ─── ETAPA 2 — Câmera ─────────────────────────────────────────────────────────

async function iniciarCaptura() {
    state.photoCount = 0;
    $("cap-nome").textContent   = state.user.nome;
    $("foto-count").textContent = "0";
    $("prog-fill").style.width  = "0%";
    $("prog-fill").closest(".track").classList.remove("done");
    $("thumbs").innerHTML  = "";
    $("cap-msg").textContent = "";
    $("cap-msg").className   = "cap-msg";
    $("btn-finalizar").disabled = true;

    goToStep(2);

    try {
        state.stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: "user" },
        });
        const video = $("video");
        video.srcObject = state.stream;
        await video.play();
        $("btn-capturar").disabled = false;
    } catch {
        mostrarMsg("Não foi possível acessar a câmera. Verifique as permissões do navegador.", "warn");
    }
}

function pararCamera() {
    if (state.stream) {
        state.stream.getTracks().forEach((t) => t.stop());
        state.stream = null;
    }
    $("btn-capturar").disabled = true;
}

// Captura uma foto e envia ao servidor
async function capturarFoto() {
    if (!state.stream) return;

    // Flash visual
    const flash = $("flash");
    flash.classList.add("on");
    setTimeout(() => flash.classList.remove("on"), 140);

    // Desenha frame no canvas (sem espelhar — imagem original da câmera)
    const video  = $("video");
    const canvas = $("cap-canvas");
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    const imageData = canvas.toDataURL("image/jpeg", 0.88);

    try {
        const res = await api("/api/usuario/foto", { nome: state.user.nome, imagem: imageData });
        if (res.erro) throw new Error(res.erro);

        state.photoCount = res.total;
        atualizarProgresso();
        adicionarThumbnail(imageData);

        if (state.photoCount >= MIN_PHOTOS) $("btn-finalizar").disabled = false;
        if (state.photoCount === TARGET_PHOTOS) {
            mostrarMsg(`${TARGET_PHOTOS} fotos capturadas! Você já pode finalizar o cadastro.`, "ok");
        }
    } catch (err) {
        mostrarMsg("Erro ao salvar foto: " + err.message, "warn");
    }
}

function atualizarProgresso() {
    const count = state.photoCount;
    const pct   = Math.min((count / TARGET_PHOTOS) * 100, 100);

    const numEl = $("foto-count");
    numEl.textContent = count;
    numEl.classList.add("bump");
    setTimeout(() => numEl.classList.remove("bump"), 150);

    $("prog-fill").style.width = pct + "%";
    if (count >= TARGET_PHOTOS) $("prog-fill").closest(".track").classList.add("done");
}

function adicionarThumbnail(src) {
    const container = $("thumbs");
    const img = document.createElement("img");
    img.src = src;
    img.className = "thumb";
    img.alt = "Foto capturada";
    container.appendChild(img);
    // Mantém só as últimas 8 miniaturas visíveis
    const all = container.querySelectorAll(".thumb");
    if (all.length > 8) all[0].remove();
}

function mostrarMsg(texto, tipo = "") {
    const el = $("cap-msg");
    el.textContent = texto;
    el.className   = "cap-msg " + tipo;
}

$("btn-capturar").addEventListener("click", capturarFoto);

$("btn-finalizar").addEventListener("click", async () => {
    pararCamera();
    await iniciarTreino();
});

$("btn-voltar").addEventListener("click", () => {
    pararCamera();
    goToStep(1);
});

// ─── ETAPA 3 — Treinamento ────────────────────────────────────────────────────

async function iniciarTreino() {
    goToStep(3);
    $("proc-nome").textContent = state.user.nome;
    mostrarTela("processing");

    try {
        const res = await api("/api/usuario/treinar", { nome: state.user.nome });
        if (res.erro) throw new Error(res.erro);
        aguardarTreino();
    } catch {
        mostrarTela("error");
    }
}

function aguardarTreino() {
    const interval = setInterval(async () => {
        try {
            const nome = encodeURIComponent(state.user.nome);
            const res  = await fetch(`/api/usuario/status-treino/${nome}`).then((r) => r.json());

            if (res.status === "concluido") {
                clearInterval(interval);
                mostrarSucesso();
            } else if (res.status === "erro") {
                clearInterval(interval);
                mostrarTela("error");
            }
        } catch {
            clearInterval(interval);
            mostrarTela("error");
        }
    }, 3000);
}

function mostrarTela(nome) {
    ["processing", "success", "error"].forEach((t) => {
        $(`scr-${t}`).classList.toggle("hidden", t !== nome);
    });
}

function mostrarSucesso() {
    const { nome, setor, cargo, matricula, nivel_acesso } = state.user;

    $("ok-msg").textContent =
        `${nome} foi cadastrado com sucesso com ${state.photoCount} foto${state.photoCount !== 1 ? "s" : ""}.`;

    const itens = [
        ["Nome",             nome],
        ["Setor",            setor      || "—"],
        ["Cargo",            cargo      || "—"],
        ["Matrícula",        matricula  || "—"],
        ["Nível de Acesso",  nivel_acesso],
        ["Fotos capturadas", state.photoCount],
    ];

    $("ok-details").innerHTML = itens
        .map(([label, val]) => `
            <div>
                <div class="d-label">${label}</div>
                <div class="d-value">${val}</div>
            </div>`)
        .join("");

    mostrarTela("success");
}

// Novo cadastro
$("btn-novo").addEventListener("click", () => {
    state.user       = {};
    state.photoCount = 0;
    $("form-dados").reset();
    $("form-error").classList.add("hidden");
    goToStep(1);
});

// Tentar treino novamente
$("btn-retry").addEventListener("click", async () => {
    await iniciarTreino();
});

// ─── Init ─────────────────────────────────────────────────────────────────────

goToStep(1);
