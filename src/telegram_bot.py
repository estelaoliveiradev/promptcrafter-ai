import os
import io
import sqlite3
from datetime import datetime
from dotenv import load_dotenv

import easyocr
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from promptcraft.finance.invoice_parser import InvoiceParser
from promptcraft.finance.models import InvoiceData

load_dotenv()

# Inicializa o leitor de OCR e o parser do promptcraft em memória
ocr_reader = easyocr.Reader(['pt', 'en'])
invoice_parser = InvoiceParser()

# --- Banco de Dados Local ---
def inicializar_banco(db_path: str = "sistema_notas.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS despesas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            telegram_user_id INTEGER,
            empresa TEXT,
            cnpj TEXT,
            data_emissao TEXT,
            valor_total REAL,
            chave_acesso TEXT,
            data_registro TEXT
        )
    """)
    conn.commit()
    conn.close()

def salvar_despesa(user_id: int, dados: InvoiceData, db_path: str = "sistema_notas.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO despesas (telegram_user_id, empresa, cnpj, data_emissao, valor_total, chave_acesso, data_registro)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        dados.empresa_emitente,
        dados.cnpj_emitente,
        dados.data_emissao,
        dados.valor_total,
        dados.chave_acesso,
        datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    ))
    conn.commit()
    conn.close()

# --- Handlers do Telegram ---
async def comando_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    nome = update.effective_user.first_name
    texto = (
        f"Olá, {nome}! 👋\n\n"
        "Envie ou tire uma foto de uma **nota fiscal** ou **cupom fiscal**.\n"
        "Eu farei a leitura automática e registrarei os dados no sistema."
    )
    await update.message.reply_text(texto, parse_mode="Markdown")

async def processar_foto_nota(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mensagem_status = await update.message.reply_text("📸 Imagem recebida! Processando OCR...")

    try:
        # 1. Baixar a imagem com maior resolução enviada pelo Telegram direto para memória
        foto = update.message.photo[-1]
        arquivo_telegram = await context.bot.get_file(foto.file_id)
        
        foto_bytes = io.BytesIO()
        await arquivo_telegram.download_to_memory(out=foto_bytes)
        foto_bytes.seek(0)

        # 2. Executar OCR direto nos bytes da imagem
        await mensagem_status.edit_text("🔍 Extraindo texto do documento...")
        linhas_encontradas = ocr_reader.readtext(foto_bytes.getvalue(), detail=0)
        texto_extraido = "\n".join(linhas_encontradas)

        if not texto_extraido.strip():
            await mensagem_status.edit_text("⚠️ Não consegui identificar textos legíveis na imagem. Tente uma foto mais aproximada e focada.")
            return

        # 3. Chamar o seu pacote promptcraft (Gemini)
        await mensagem_status.edit_text("🧠 Analisando dados fiscais com IA...")
        dados_estruturados: InvoiceData = invoice_parser.parse(texto_extraido)

        # 4. Gravar no banco de dados SQLite
        salvar_despesa(user_id=update.effective_user.id, dados=dados_estruturados)

        # 5. Formatar resposta amigável para o usuário
        valor_formatado = f"R$ {dados_estruturados.valor_total:.2f}".replace('.', ',') if dados_estruturados.valor_total else "Não identificado"
        
        resposta = (
            "✅ **Nota registrada com sucesso!**\n\n"
            f"🏢 **Estabelecimento:** {dados_estruturados.empresa_emitente or 'Não identificado'}\n"
            f"📑 **CNPJ:** {dados_estruturados.cnpj_emitente or 'Não identificado'}\n"
            f"📅 **Data:** {dados_estruturados.data_emissao or 'Não informada'}\n"
            f"💰 **Valor Total:** {valor_formatado}\n"
        )
        
        if dados_estruturados.chave_acesso:
            resposta += f"\n🔑 **Chave:** `{dados_estruturados.chave_acesso}`"

        await mensagem_status.edit_text(resposta, parse_mode="Markdown")

    except Exception as e:
        await mensagem_status.edit_text(f"❌ Ocorreu um erro ao processar a nota: {str(e)}")

# --- Ponto de Entrada ---
if __name__ == "__main__":
    inicializar_banco()
    
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN não configurado no arquivo .env!")

    app = ApplicationBuilder().token(token).build()

    # Registra comandos e listeners de fotos
    app.add_handler(CommandHandler("start", comando_start))
    app.add_handler(MessageHandler(filters.PHOTO, processar_foto_nota))

    print("🤖 Bot do Telegram em execução! Pressione Ctrl+C para parar.")
    app.run_polling()