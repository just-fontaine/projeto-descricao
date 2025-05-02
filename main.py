import os
import cv2
import json
import re
import sys
# import io # Não parece estar sendo usado diretamente
import threading
import tkinter
import tkinter.filedialog
import hashlib # <<< Adicionado para hashing
from thefuzz import fuzz
import tkinter.messagebox
import customtkinter as ctk
# from tqdm import tqdm # Comentado pois não é visível na GUI
from PIL import Image
import torch
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer,
    T5TokenizerFast, T5ForConditionalGeneration, pipeline
)
import subprocess # Para abrir arquivos

# --- Configurações Globais ---
# DEFAULT_OUTPUT_FILENAME = "resumos_multimidia.json" # <<< Removido, agora é dinâmico
RESULTS_SUBFOLDER = "_results" # <<< Nova constante para pasta de resultados
MAX_VIDEO_FRAMES = 10
MAX_SUMMARIZER_INPUT_CHARS = 1000
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VALID_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# --- Verificação de GPU e Carregamento de Modelos ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# --- Variáveis Globais para Modelos ---
blip_processor = None
blip_model = None
vit_processor = None
vit_tokenizer = None
vit_model = None
summarizer = None
models_loaded = False
models_loading_error = None

# --- Função para obter o caminho base da aplicação ---
def get_app_base_dir():
    """Retorna o diretório onde o script está localizado ou o diretório de trabalho."""
    try:
        # __file__ é mais confiável quando o script é executado diretamente
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Fallback para o diretório de trabalho atual se __file__ não estiver definido (ex: REPL)
        return os.getcwd()

# --- Função para Gerar Caminho do JSON de Saída ---
def get_output_json_path(input_folder_path):
    """Gera um caminho de arquivo JSON único baseado no caminho da pasta de entrada."""
    if not input_folder_path:
        return None
    try:
        # Normaliza o caminho para consistência entre OS e entradas do usuário
        normalized_path = os.path.normpath(os.path.abspath(input_folder_path))
        # Cria um hash MD5 do caminho normalizado
        # Usar lower() para consistência em sistemas case-insensitive como Windows
        path_bytes = normalized_path.lower().encode('utf-8')
        hash_id = hashlib.md5(path_bytes).hexdigest()
        filename = f"{hash_id}.json"

        app_dir = get_app_base_dir()
        results_dir = os.path.join(app_dir, RESULTS_SUBFOLDER)
        # Não criar a pasta aqui, será criada antes de salvar/processar

        return os.path.join(results_dir, filename)
    except Exception as e:
        print(f"Erro ao gerar nome do arquivo JSON para '{input_folder_path}': {e}")
        return None

# --- Carregamento de Modelos (Inalterado) ---
def load_models():
    """Carrega todos os modelos necessários."""
    global blip_processor, blip_model, vit_processor, vit_tokenizer, vit_model, summarizer, models_loaded, models_loading_error
    if models_loaded: return True
    models_loading_error = None
    print("Iniciando carregamento dos modelos...")
    try:
        print("  Carregando BLIP...")
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        print("  Modelo BLIP carregado.")

        print("  Carregando ViT-GPT2...")
        vit_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        # Usar use_fast=True para potencial otimização e silenciar warnings
        vit_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning", use_fast=True)
        vit_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
        vit_tokenizer.pad_token = vit_tokenizer.eos_token
        print("  Modelo ViT-GPT2 carregado.")

        print("  Carregando T5 (sumarizador)...")
        sum_tokenizer = T5TokenizerFast.from_pretrained("t5-small") # Já é 'fast'
        sum_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
        pipeline_device = 0 if device == "cuda" else -1
        summarizer = pipeline("summarization", model=sum_model, tokenizer=sum_tokenizer, device=pipeline_device)
        print("  Modelo T5 carregado.")

        models_loaded = True
        print("Todos os modelos foram carregados com sucesso.")
        return True

    except Exception as e:
        models_loading_error = f"Erro Crítico ao carregar modelos: {e}\nVerifique a conexão, dependências e memória."
        print(f"\n{models_loading_error}\n")
        models_loaded = False
        blip_processor = blip_model = vit_processor = vit_tokenizer = vit_model = summarizer = None
        return False

# --- Funções de Processamento (Inalteradas internamente, dependem do caminho de saída correto) ---
def limpar_repeticoes(texto):
    if not texto: return ""
    palavras = texto.lower().split()
    if not palavras: return ""
    resultado = [palavras[0]]
    for i in range(1, len(palavras)):
        if palavras[i] != palavras[i-1]:
            resultado.append(palavras[i])
    return ' '.join(resultado)

def gerar_descricao_imagem_blip(imagem_path):
    if not blip_processor or not blip_model:
        return "[Erro: Modelo BLIP não carregado]"
    try:
        imagem = Image.open(imagem_path).convert("RGB")
        inputs = blip_processor(images=imagem, return_tensors="pt").to(device)
        with torch.no_grad():
             output = blip_model.generate(**inputs, max_length=50)
        legenda = blip_processor.decode(output[0], skip_special_tokens=True)
        return limpar_repeticoes(legenda)
    except FileNotFoundError:
        print(f"  Erro: Arquivo de imagem não encontrado: {imagem_path}")
        return "[Erro: Arquivo não encontrado]"
    except Exception as e:
        print(f"  Erro ao processar imagem {os.path.basename(imagem_path)} com BLIP: {e}")
        return "[Erro ao processar imagem]"

def gerar_descricao_quadro_vit(pil_image):
    if not vit_processor or not vit_model or not vit_tokenizer:
        return None # Não retorna erro, deixa processar outros quadros
    try:
        pixel_values = vit_processor(images=pil_image, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            output_ids = vit_model.generate(pixel_values, max_length=30)
        descricao = vit_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return limpar_repeticoes(descricao)
    except Exception as e:
        print(f"  Erro ao gerar descrição de quadro com ViT-GPT2: {e}")
        return None

def extrair_quadros_distribuidos(video_path, max_quadros=MAX_VIDEO_FRAMES):
    quadros = []
    cap = None
    try:
        # Usar tratamento de erro robusto para caminhos de arquivo
        if not os.path.isfile(video_path):
             print(f"  Erro: Arquivo de vídeo não encontrado ou inválido: {video_path}")
             return []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  Erro ao abrir o vídeo: {os.path.basename(video_path)}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"  Aviso: Vídeo sem quadros ou inválido: {os.path.basename(video_path)}")
            return []

        intervalo = max(1, total_frames // max_quadros) if max_quadros > 0 and total_frames > max_quadros else 1
        indices_quadros = []
        for i in range(max_quadros):
            idx = min(i * intervalo, total_frames - 1)
            if not indices_quadros or idx > indices_quadros[-1]: # Garante índices crescentes e únicos
                 indices_quadros.append(idx)
            if len(indices_quadros) >= max_quadros: break

        if total_frames > 0 and indices_quadros[-1] < total_frames - 1 and len(indices_quadros) < max_quadros:
            indices_quadros.append(total_frames - 1) # Garante último quadro se houver espaço

        indices_unicos = sorted(list(set(indices_quadros))) # Segurança extra contra duplicatas

        for frame_index in indices_unicos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    quadros.append(Image.fromarray(frame_rgb))
                except Exception as cvt_err:
                     print(f"  Erro convertendo/lendo quadro {frame_index} de {os.path.basename(video_path)}: {cvt_err}")
            # else: # Log opcional
            #    print(f"  Aviso: Falha ao ler quadro {frame_index} de {os.path.basename(video_path)}")

    except Exception as e:
        print(f"  Erro inesperado durante extração de quadros de {os.path.basename(video_path)}: {e}")
        quadros = []
    finally:
        if cap is not None and cap.isOpened(): cap.release()
    return quadros

def sumarizar_texto(texto):
    if not summarizer:
        return "[Erro: Sumarizador T5 não carregado]"
    if not texto or not texto.strip():
        return "[Info: Nenhuma descrição de quadro para sumarizar]"
    texto_cortado = texto[:MAX_SUMMARIZER_INPUT_CHARS]
    try:
        resumo_modelo = summarizer(texto_cortado, max_length=80, min_length=15, do_sample=False)
        # Limpeza básica do resumo
        resumo = re.sub(r'\s+([.?!])', r'\1', resumo_modelo[0]['summary_text']).strip()
        return resumo if resumo else "[Info: Resumo gerado estava vazio]"
    except Exception as e:
        print(f"  Erro durante a sumarização T5: {e}")
        return "[Erro ao gerar resumo]"

def processar_video(video_path, progress_callback=None):
    quadros = extrair_quadros_distribuidos(video_path, MAX_VIDEO_FRAMES)
    if not quadros:
        # Mensagens de erro mais específicas da extração
        if not os.path.exists(video_path): return "[Erro: Arquivo de vídeo não encontrado]"
        # Verificar se é possível abrir
        cap_check = cv2.VideoCapture(video_path)
        is_openable = cap_check.isOpened()
        cap_check.release()
        if not is_openable: return "[Erro: Falha ao abrir o arquivo de vídeo (formato?)]"
        return "[Erro: Falha ao extrair quadros (vídeo vazio ou erro interno)]"

    descricoes_quadros = []
    for i, quadro in enumerate(quadros):
        descricao_quadro = gerar_descricao_quadro_vit(quadro)
        # Adiciona apenas descrições válidas e únicas
        if descricao_quadro and descricao_quadro not in descricoes_quadros:
            descricoes_quadros.append(descricao_quadro)

    if not descricoes_quadros: return "[Info: Nenhuma descrição única válida gerada para os quadros]"
    texto_completo = ". ".join(descricoes_quadros) + "."
    resumo = sumarizar_texto(texto_completo)
    return resumo

def processar_pasta(pasta_path, output_json_path, progress_callback=None, update_progress_bar=None):
    """Processa arquivos na pasta e salva no JSON especificado."""
    resultados = {}
    if progress_callback: progress_callback(f"Procurando arquivos em: {pasta_path}...")

    # --- Validação inicial ---
    if not pasta_path or not os.path.isdir(pasta_path):
        msg = f"ERRO: Pasta de entrada inválida ou inacessível: {pasta_path}"
        if progress_callback: progress_callback(msg)
        print(msg)
        return False # Falha, não pode continuar

    if not output_json_path:
        msg = "ERRO: Caminho do arquivo de saída não foi definido."
        if progress_callback: progress_callback(msg)
        print(msg)
        return False # Falha, não pode continuar

    # --- Coleta de Arquivos ---
    arquivos_encontrados = []
    try:
        for root, _, files in os.walk(pasta_path):
            for nome_arquivo in files:
                try:
                    full_path = os.path.join(root, nome_arquivo)
                    # Adiciona apenas se for um arquivo real (ignora links quebrados etc.)
                    if os.path.isfile(full_path):
                        arquivos_encontrados.append(full_path)
                except Exception as path_err:
                    print(f"Aviso: Ignorando erro ao processar caminho: {root}/{nome_arquivo} - {path_err}")
    except Exception as walk_err:
        msg = f"Erro crítico ao varrer a pasta '{pasta_path}': {walk_err}"
        print(msg)
        if progress_callback: progress_callback(msg)
        return False # Falha na varredura

    if not arquivos_encontrados:
        if progress_callback: progress_callback("Nenhum arquivo encontrado na pasta.")
        # Se não há arquivos, não há o que processar, mas não é um erro do processo em si.
        # Tentamos salvar um JSON vazio para indicar que a pasta foi "processada".
        try:
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump({}, f) # Salva um JSON vazio
            if progress_callback: progress_callback(f"Pasta '{os.path.basename(pasta_path)}' processada (sem arquivos encontrados). Arquivo de resultado vazio criado.")
            return True # Sucesso (sem processamento, mas concluiu)
        except Exception as save_err:
            msg = f"Erro ao salvar arquivo JSON vazio para pasta sem arquivos: {save_err}"
            print(msg)
            if progress_callback: progress_callback(msg)
            return False # Falha ao salvar o indicativo


    if progress_callback: progress_callback(f"Encontrados {len(arquivos_encontrados)} arquivos totais.")

    # --- Filtragem de Arquivos Suportados ---
    arquivos_a_processar = [
        arq for arq in arquivos_encontrados
        if os.path.splitext(arq)[1].lower() in VALID_IMAGE_EXTENSIONS or \
           os.path.splitext(arq)[1].lower() in VALID_VIDEO_EXTENSIONS
    ]

    if not arquivos_a_processar:
        if progress_callback: progress_callback("Nenhum arquivo com extensão de imagem ou vídeo suportada encontrado.")
        # Similar ao caso de não encontrar arquivos, salva JSON vazio.
        try:
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump({}, f)
            if progress_callback: progress_callback(f"Pasta '{os.path.basename(pasta_path)}' processada (sem arquivos suportados). Arquivo de resultado vazio criado.")
            return True
        except Exception as save_err:
            msg = f"Erro ao salvar arquivo JSON vazio para pasta sem arquivos suportados: {save_err}"
            print(msg)
            if progress_callback: progress_callback(msg)
            return False

    total_files = len(arquivos_a_processar)
    if progress_callback: progress_callback(f"Iniciando processamento de {total_files} arquivos suportados...")

    # --- Processamento dos Arquivos ---
    process_errors = 0
    for i, caminho_arquivo in enumerate(arquivos_a_processar):
        extensao = os.path.splitext(caminho_arquivo)[1].lower()
        resultado_final = None
        file_name = os.path.basename(caminho_arquivo)
        log_prefix = f"({i+1}/{total_files})"

        if update_progress_bar: update_progress_bar( (i + 1) / total_files )

        try:
            if extensao in VALID_IMAGE_EXTENSIONS:
                if progress_callback: progress_callback(f"{log_prefix} Imagem: {file_name}")
                resultado_final = gerar_descricao_imagem_blip(caminho_arquivo)
            elif extensao in VALID_VIDEO_EXTENSIONS:
                if progress_callback: progress_callback(f"{log_prefix} Vídeo: {file_name}")
                resultado_final = processar_video(caminho_arquivo, progress_callback=None)

            if resultado_final and isinstance(resultado_final, str):
                resultados[caminho_arquivo] = resultado_final
                if resultado_final.startswith("[Erro"):
                    process_errors += 1
                    if progress_callback: progress_callback(f"  -> {resultado_final}") # Log erro
                # elif resultado_final.startswith("[Info"): # Log Info (opcional)
                    # if progress_callback: progress_callback(f"  -> {resultado_final}")
            else:
                # Caso onde a função retorna None ou algo inesperado
                process_errors += 1
                resultados[caminho_arquivo] = "[Erro desconhecido durante processamento]"
                if progress_callback: progress_callback(f"  -> Aviso: Nenhum resultado válido para {file_name}")

        except MemoryError:
             process_errors += 1
             error_msg = f"ERRO DE MEMÓRIA processando {file_name}. Tente fechar outras aplicações."
             print(error_msg)
             if progress_callback: progress_callback(error_msg)
             resultados[caminho_arquivo] = "[Erro de Memória]"
             # Considerar abortar aqui? Por enquanto continua.
        except Exception as e:
            process_errors += 1
            error_msg = f"ERRO INESPERADO processando {file_name}: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc() # Imprime detalhes do erro no console
            if progress_callback: progress_callback(error_msg)
            resultados[caminho_arquivo] = "[Erro inesperado no processamento]"

    # --- Mensagem Final e Salvamento ---
    if progress_callback:
        status_msg = f"\nProcessamento de '{os.path.basename(pasta_path)}' concluído"
        if process_errors > 0:
            status_msg += f" com {process_errors} erro(s) em {total_files} arquivos."
        else:
             status_msg += f" com sucesso ({total_files} arquivos)."
        status_msg += " Salvando resultados..."
        progress_callback(status_msg)

    try:
        # Garante que a pasta de resultados exista ANTES de tentar salvar
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(resultados, f, indent=2, ensure_ascii=False)
        if progress_callback: progress_callback(f"✅ Resultados salvos com sucesso em:\n{output_json_path}")
        return True # Sucesso ao salvar
    except Exception as e:
        error_msg = f"ERRO CRÍTICO AO SALVAR O ARQUIVO JSON: {e}\nCaminho: {output_json_path}"
        print(error_msg)
        if progress_callback: progress_callback(error_msg)
        return False # Falha ao salvar

# --- Função de Busca (Inalterada, usa o json_path fornecido) ---
def search_descriptions(json_path, query, similarity_threshold=80):
    """
    Busca nos resultados JSON usando fuzzy matching e ranking.
    Retorna (lista_resultados_ordenados, mensagem_status).
    """
    # Adiciona verificação se o path é válido antes de checar existência
    if not json_path:
        return [], "Erro: Nenhum arquivo de resultados definido (Selecione uma pasta processada)."

    if not os.path.exists(json_path):
        # Tenta dar uma dica melhor sobre o que pode ter acontecido
        folder_name = os.path.basename(os.path.dirname(os.path.dirname(json_path))) # Tentativa de pegar nome da pasta original
        return [], f"Erro: Arquivo de resultados para a pasta selecionada não encontrado.\n'{os.path.basename(json_path)}'\nExecute o processamento primeiro."

    if not query or not query.strip():
        return [], "Info: Digite um termo para buscar."

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            # Verifica se o arquivo não está vazio antes de tentar carregar
            content = f.read()
            if not content:
                return [], f"Info: O arquivo de resultados '{os.path.basename(json_path)}' está vazio (nenhum arquivo processado ou erro anterior)."
            data = json.loads(content) # Usa o conteúdo lido
    except json.JSONDecodeError:
         return [], f"Erro: Arquivo JSON corrompido ou inválido:\n{json_path}"
    except Exception as e:
        return [], f"Erro ao ler o arquivo JSON: {e}"

    # Se data for vazio após carregar (caso raro, mas possível)
    if not data:
         return [], f"Info: O arquivo de resultados '{os.path.basename(json_path)}' não contém dados."

    query_lower = query.lower().strip()
    keywords = query_lower.split()

    if not keywords:
        return [], "Info: Consulta de busca inválida."

    match_scores = {}

    for file_path, description in data.items():
        # Pula entradas sem descrição ou que são erros
        if not description or not isinstance(description, str) or description.startswith("[Erro"):
            continue

        desc_lower = description.lower()
        total_score = 0
        matched_keyword_count = 0

        for keyword in keywords:
            score = fuzz.token_set_ratio(keyword, desc_lower)
            if score >= similarity_threshold:
                total_score += score
                matched_keyword_count += 1

        if matched_keyword_count > 0:
            final_score = total_score # Score simples por soma
            match_scores[file_path] = final_score

    if not match_scores:
        return [], f"Nenhum resultado similar encontrado para: '{query}' (limite: {similarity_threshold}%)"

    sorted_files = [item[0] for item in sorted(match_scores.items(), key=lambda item: item[1], reverse=True)]

    count = len(sorted_files)
    plural = "s" if count > 1 else ""
    return sorted_files, f"{count} resultado{plural} similar{plural} encontrado{plural} (ordenado por relevância)."

# --- Classe da Janela de Busca (Inalterada) ---
class SearchWindow(ctk.CTkToplevel):
    def __init__(self, parent, json_file_path):
        super().__init__(parent)
        self.parent_app = parent
        self.json_path = json_file_path # Recebe o caminho específico da pasta

        self.title(f"Buscar em: {os.path.basename(os.path.dirname(json_file_path))}") # Mostra pasta no título
        self.geometry("650x450")
        self.minsize(500, 300)
        self.lift()
        self.focus()

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.search_frame = ctk.CTkFrame(self)
        self.search_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.search_frame.grid_columnconfigure(0, weight=1)

        self.entry_search = ctk.CTkEntry(self.search_frame, placeholder_text="Digite palavras-chave (ex: cat, beach)...")
        self.entry_search.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="ew")
        self.entry_search.bind("<Return>", self.perform_search_event)

        self.button_search = ctk.CTkButton(self.search_frame, text="Buscar", width=100, command=self.perform_search)
        self.button_search.grid(row=0, column=1, padx=(5, 10), pady=10)

        self.results_frame = ctk.CTkScrollableFrame(self, label_text="Resultados da Busca")
        self.results_frame.grid(row=1, column=0, padx=10, pady=(0, 5), sticky="nsew")
        self.results_frame.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(self, text="Digite sua busca acima.", anchor="w")
        self.status_label.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="ew")

        self.result_widgets = []

    def perform_search_event(self, event):
        self.perform_search()

    def perform_search(self):
        query = self.entry_search.get()
        # Passa o self.json_path específico desta janela
        found_files, message = search_descriptions(self.json_path, query)

        for widget in self.result_widgets:
            widget.destroy()
        self.result_widgets = []

        self.status_label.configure(text=message)

        if found_files:
            for i, file_path in enumerate(found_files):
                try:
                    display_name = os.path.basename(file_path)
                except Exception:
                    display_name = "Erro no nome"

                btn = ctk.CTkButton(
                    self.results_frame,
                    text=display_name,
                    anchor="w",
                    fg_color="transparent",
                    hover_color=("gray70", "gray30"),
                    command=lambda p=file_path: self.open_file(p)
                )
                btn.grid(row=i, column=0, padx=5, pady=2, sticky="ew")
                self.result_widgets.append(btn)

    def open_file(self, file_path):
        """Tenta abrir o arquivo selecionado com o programa padrão."""
        normalized_path = os.path.normpath(file_path)
        if not os.path.exists(normalized_path):
             if not os.path.exists(file_path): # Tenta original
                tkinter.messagebox.showerror("Erro", f"Arquivo não encontrado:\n{file_path}")
                return
             else:
                 normalized_path = file_path # Usa original se existir

        try:
            if sys.platform == "win32":
                os.startfile(normalized_path)
            elif sys.platform == "darwin":
                subprocess.run(['open', normalized_path], check=True)
            else:
                subprocess.run(['xdg-open', normalized_path], check=True)
        except FileNotFoundError:
             tkinter.messagebox.showerror("Erro", f"Arquivo não encontrado:\n{normalized_path}")
        except subprocess.CalledProcessError as e:
             tkinter.messagebox.showerror("Erro", f"Falha ao abrir arquivo:\n{e}")
        except Exception as e:
            tkinter.messagebox.showerror("Erro", f"Não foi possível abrir o arquivo:\n{normalized_path}\n\nErro: {e}")
            print(f"Erro ao tentar abrir {normalized_path}: {e}")

# --- Classe da Interface Gráfica Principal (App - Modificada) ---
class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Resumidor de Mídia Local") # Título Atualizado
        self.geometry("700x550")
        self.minsize(600, 400)

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.input_folder_path = ""
        self.output_file_path = "" # <<< Agora guarda o caminho DINÂMICO do JSON
        self.processing_thread = None
        self.search_window = None

        # Layout Principal
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        # Frame de Seleção
        self.selection_frame = ctk.CTkFrame(self)
        self.selection_frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        self.selection_frame.grid_columnconfigure(1, weight=1)
        self.label_input = ctk.CTkLabel(self.selection_frame, text="Pasta de Entrada:")
        self.label_input.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="w")
        self.entry_input_path = ctk.CTkEntry(self.selection_frame, placeholder_text="Selecione a pasta com imagens/vídeos...")
        self.entry_input_path.grid(row=0, column=1, padx=5, pady=10, sticky="ew")
        self.entry_input_path.configure(state="disabled")
        self.button_select_folder = ctk.CTkButton(self.selection_frame, text="Selecionar Pasta", command=self.select_input_folder)
        self.button_select_folder.grid(row=0, column=2, padx=(5, 10), pady=10, sticky="e")

        # Frame de Ação
        self.action_frame = ctk.CTkFrame(self)
        self.action_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        self.action_frame.grid_columnconfigure(1, weight=1)
        self.button_search_window = ctk.CTkButton(self.action_frame, text="Abrir Busca", command=self.open_search_window, state="disabled")
        self.button_search_window.grid(row=0, column=0, padx=(10,5), pady=10, sticky="w")
        self.button_start = ctk.CTkButton(self.action_frame, text="Iniciar Processamento", command=self.start_processing_thread, state="disabled")
        self.button_start.grid(row=0, column=2, padx=(5, 10), pady=10, sticky="e")

        # Frame da Barra de Progresso
        self.progress_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.progress_frame.grid(row=2, column=0, padx=10, pady=0, sticky="ew")
        self.progress_frame.grid_columnconfigure(0, weight=1)
        self.progressbar = ctk.CTkProgressBar(self.progress_frame, orientation="horizontal")
        self.progressbar.grid(row=0, column=0, padx=0, pady=0, sticky="ew")
        self.progressbar.set(0)

        # Caixa de Texto para Status/Logs
        self.textbox_status = ctk.CTkTextbox(self, wrap="word")
        self.textbox_status.grid(row=3, column=0, padx=10, pady=(5, 10), sticky="nsew")
        self.textbox_status.configure(state="disabled", font=("Consolas", 11))

        # self.check_existing_results() # <<< Removido, a checagem é feita em select_input_folder

    # def check_existing_results(self): # <<< Função removida

    def update_status(self, message):
        """Atualiza a caixa de texto de status de forma segura."""
        if self.textbox_status.winfo_exists():
            try:
                self.textbox_status.configure(state="normal")
                self.textbox_status.insert("end", str(message) + "\n")
                self.textbox_status.see("end")
                self.textbox_status.configure(state="disabled")
            except Exception as e:
                print(f"Erro ao atualizar status GUI: {e}")

    def update_progress_bar(self, value):
        """Atualiza a barra de progresso de forma segura."""
        if self.progressbar.winfo_exists():
            try:
                # Garante que o valor está entre 0.0 e 1.0
                clamped_value = max(0.0, min(float(value), 1.0))
                self.progressbar.set(clamped_value)
            except Exception as e:
                print(f"Erro ao atualizar progresso GUI: {e}")

    def select_input_folder(self):
        """Seleciona a pasta, gera o path do JSON, verifica se existe e atualiza a UI."""
        folder_selected = tkinter.filedialog.askdirectory()
        if folder_selected:
            self.input_folder_path = folder_selected
            # Atualiza o campo de texto na UI
            self.entry_input_path.configure(state="normal")
            self.entry_input_path.delete(0, "end")
            self.entry_input_path.insert(0, self.input_folder_path)
            self.entry_input_path.configure(state="disabled")

            # Gera o caminho esperado para o arquivo JSON desta pasta
            self.output_file_path = get_output_json_path(self.input_folder_path)

            if not self.output_file_path:
                 self.update_status(f"ERRO: Não foi possível gerar o nome do arquivo de resultados para:\n{self.input_folder_path}")
                 self.button_start.configure(state="disabled")
                 self.button_search_window.configure(state="disabled")
                 return

            self.update_status(f"Pasta selecionada: {self.input_folder_path}")
            self.update_status(f"Arquivo de resultados esperado: ...\\{RESULTS_SUBFOLDER}\\{os.path.basename(self.output_file_path)}")

            # Verifica se o arquivo JSON já existe
            if os.path.exists(self.output_file_path):
                self.update_status("✅ Resultados encontrados para esta pasta. Busca disponível.")
                self.button_search_window.configure(state="normal") # Habilita busca
            else:
                self.update_status("ℹ️ Resultados não encontrados. É necessário processar esta pasta.")
                self.button_search_window.configure(state="disabled") # Desabilita busca

            # Habilita o botão de processar se os modelos carregaram (permite sobrescrever)
            if models_loaded or models_loading_error is None:
                 self.button_start.configure(state="normal")
            else:
                 self.button_start.configure(state="disabled")
                 self.update_status("AVISO: Modelos não carregados, processamento indisponível.")

        else:
            # Seleção cancelada, não muda estado dos botões se já tinha algo selecionado
            self.update_status("Seleção de pasta cancelada.")
            if not self.input_folder_path: # Se nada estava selecionado antes
                self.button_start.configure(state="disabled")
                self.button_search_window.configure(state="disabled")
                self.output_file_path = "" # Limpa o caminho de saída

    def set_ui_processing(self, is_processing):
        """Habilita/Desabilita controles durante o processamento."""
        select_state = "disabled" if is_processing else "normal"
        start_state = "disabled" # Start sempre desabilita durante

        # Lógica para habilitar Start após processamento (ou inicialmente)
        if not is_processing and self.input_folder_path and (models_loaded or models_loading_error is None):
            start_state = "normal"
            if models_loading_error is not None: # Mantém desabilitado se modelos falharam
                start_state = "disabled"

        # Lógica para habilitar Busca após processamento (ou inicialmente)
        search_state = "disabled"
        # Verifica a existência do ARQUIVO ATUALMENTE SELECIONADO
        if not is_processing and self.output_file_path and os.path.exists(self.output_file_path):
            search_state = "normal"

        # Aplica os estados com segurança
        try:
            if self.button_select_folder.winfo_exists(): self.button_select_folder.configure(state=select_state)
            if self.button_start.winfo_exists(): self.button_start.configure(state=start_state)
            if self.button_search_window.winfo_exists(): self.button_search_window.configure(state=search_state)
        except Exception as e:
            print(f"Erro ao definir estado da UI: {e}")

    def start_processing_thread(self):
        """Inicia a thread de processamento para a pasta e JSON selecionados."""
        # --- Validações Essenciais ---
        if not self.input_folder_path or not os.path.isdir(self.input_folder_path):
            self.update_status("ERRO: Selecione uma pasta de entrada válida antes de processar.")
            tkinter.messagebox.showerror("Erro", "Selecione uma pasta de entrada válida.")
            return
        if not self.output_file_path:
             self.update_status("ERRO: Caminho do arquivo de saída não definido. Selecione a pasta novamente.")
             tkinter.messagebox.showerror("Erro", "Caminho do arquivo de saída não definido.\nSelecione a pasta novamente.")
             return
        if self.processing_thread and self.processing_thread.is_alive():
            self.update_status("Aviso: Processamento já está em andamento.")
            tkinter.messagebox.showwarning("Aviso", "O processamento já está em andamento.")
            return
        if not models_loaded:
             error_msg = models_loading_error if models_loading_error else "Modelos não carregados."
             self.update_status(f"ERRO: {error_msg} Processamento não pode iniciar.")
             tkinter.messagebox.showerror("Erro de Modelo", f"Não é possível iniciar:\n{error_msg}")
             return

        # --- Preparação ---
        self.textbox_status.configure(state="normal")
        self.textbox_status.delete("1.0", "end") # Limpa status anterior
        self.textbox_status.configure(state="disabled")
        self.progressbar.set(0)
        self.update_status(f"Iniciando processamento para: {self.input_folder_path}")
        self.update_status(f"Salvando resultados em: {self.output_file_path}")

        # Garante que a pasta de resultados exista ANTES de iniciar a thread
        try:
            os.makedirs(os.path.dirname(self.output_file_path), exist_ok=True)
        except Exception as e:
            self.update_status(f"ERRO CRÍTICO: Não foi possível criar a pasta de resultados '{RESULTS_SUBFOLDER}': {e}")
            tkinter.messagebox.showerror("Erro de Pasta", f"Não foi possível criar a pasta '{RESULTS_SUBFOLDER}':\n{e}")
            return

        # --- Iniciar Thread ---
        self.set_ui_processing(True) # Desabilita controles
        self.processing_thread = threading.Thread(
            target=self.run_processing,
            # Passa os caminhos atuais para a thread
            args=(self.input_folder_path, self.output_file_path),
            daemon=True
        )
        self.processing_thread.start()

    def run_processing(self, folder_path, output_json_path):
        """Função executada na thread separada."""
        processing_successful = False
        try:
            # A verificação de modelos já foi feita antes de iniciar a thread
            processing_successful = processar_pasta(
                folder_path,
                output_json_path,
                progress_callback=self.safe_update_status,
                update_progress_bar=self.safe_update_progress_bar
            )
            # A função processar_pasta retorna True em sucesso (incluindo pasta vazia),
            # e False se houve erro crítico (leitura/escrita/varredura).

        except Exception as e:
            error_msg = f"ERRO FATAL na thread de processamento: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self.safe_update_status(error_msg)
            processing_successful = False # Marca como falha
        finally:
            # Reabilita a interface usando after(0) para rodar na thread principal
            # A função set_ui_processing verificará a existência do arquivo de saída
            # para habilitar a busca corretamente.
            self.safe_set_ui_processing(False)
            # Reseta a barra de progresso
            self.safe_update_progress_bar(0)
            # Se o processamento foi bem sucedido (criou/atualizou o JSON),
            # garante que o botão de busca esteja habilitado (caso não estivesse)
            # Isso é redundante se set_ui_processing funciona, mas é uma garantia extra.
            # if processing_successful:
            #     self.after(10, lambda: self.button_search_window.configure(state="normal") if self.output_file_path and os.path.exists(self.output_file_path) else None)


    # --- Funções Seguras para Chamar da Thread (Inalteradas) ---
    def safe_update_status(self, message):
        self.after(0, self.update_status, message)

    def safe_update_progress_bar(self, value):
        self.after(0, self.update_progress_bar, value)

    def safe_set_ui_processing(self, is_processing):
        self.after(0, self.set_ui_processing, is_processing)

    # --- Função para Abrir Janela de Busca (Modificada para verificar path) ---
    def open_search_window(self):
        """Abre a janela de busca Toplevel para o JSON da pasta atual."""
        # Verifica se um caminho de saída válido está definido E se o arquivo existe
        if not self.output_file_path:
             tkinter.messagebox.showwarning("Busca Indisponível",
                                           "Nenhuma pasta foi selecionada ou processada ainda.")
             return
        if not os.path.exists(self.output_file_path):
            tkinter.messagebox.showwarning("Busca Indisponível",
                                           f"O arquivo de resultados para a pasta selecionada não foi encontrado:\n"
                                           f"...\\{RESULTS_SUBFOLDER}\\{os.path.basename(self.output_file_path)}\n"
                                           "Execute o processamento primeiro.")
            # Tenta revalidar o estado (caso tenha sido criado mas a UI não pegou)
            self.set_ui_processing(False)
            return

        # Se a janela já existe, traz para frente. Senão, cria uma nova.
        if self.search_window is None or not self.search_window.winfo_exists():
            # Passa o caminho específico do JSON para a janela de busca
            self.search_window = SearchWindow(self, self.output_file_path)
            self.search_window.protocol("WM_DELETE_WINDOW", self._on_search_close)
        else:
            # Se a janela existe, mas é para um JSON diferente, fecha a antiga e abre a nova?
            # Ou apenas atualiza o título e força uma nova busca?
            # Por simplicidade, apenas traz para frente. O usuário pode fechar e reabrir se quiser buscar em outra pasta.
            # TODO: Considerar atualizar o json_path da janela existente se mudar a pasta selecionada na principal.
            self.search_window.lift()
            self.search_window.focus()

    def _on_search_close(self):
        """Handler para quando a janela de busca é fechada."""
        if self.search_window:
            self.search_window.destroy()
        self.search_window = None


# --- Execução Principal ---
if __name__ == "__main__":
    # Tenta carregar os modelos ANTES de iniciar a GUI
    print("Pré-carregando modelos... (Pode levar alguns minutos na primeira vez)")
    models_loaded_successfully = load_models()

    # Inicia a aplicação Tkinter
    app = App()

    # Informa o usuário na GUI se houve falha no carregamento inicial
    if not models_loaded_successfully:
        app.update_status(f"AVISO CRÍTICO: {models_loading_error}")
        app.update_status("O processamento de arquivos estará desabilitado.")
        # Desabilita explicitamente os botões que dependem dos modelos
        app.button_start.configure(state="disabled")
        tkinter.messagebox.showerror("Erro de Carregamento",
                                     f"Falha ao carregar modelos de IA:\n{models_loading_error}\n\n"
                                     "Verifique sua conexão com a internet, dependências (PyTorch, etc.) e memória.\n"
                                     "O processamento não funcionará.")
    else:
         app.update_status("Modelos carregados. Selecione uma pasta para iniciar.")

    app.mainloop()
