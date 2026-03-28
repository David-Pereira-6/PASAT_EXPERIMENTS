#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema Mestre de Sincronização - PASAT-C v2.0
Versão com controlo manual e timers automáticos
Atualizado: Mensagens opcionais para Biopac
Autor: Sistema de Integração Experimental
Data: 2026-02-03
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import time
import threading
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import ctypes
from ctypes import c_char_p, c_int, c_ulong, c_void_p, CFUNCTYPE
import subprocess
import sys
import os

# Imports para imagens
try:
    import imageio.v2 as imageio
except ImportError:
    import imageio
import cv2

app = Flask(__name__)
CORS(app)

# ============================================================================
# CONFIGURAÇÃO GLOBAL
# ============================================================================

BASE_DIR = Path(r"C:/Users/david/Desktop/PASAT_Experiments")
BASE_DIR.mkdir(parents=True, exist_ok=True)

XENICS_DLL = r"C:/Program Files/Common Files/XenICs/Runtime/xeneth64.dll"
GNURADIO_SCRIPT = BASE_DIR / "bioradar_recorder.py"

# Constantes Xenics
FT_16_BPP_GRAY = 2
XGF_Blocking = 1
XSMessage = 6

# Estado global
current_session = None
recording_status = {
    'xenics': False,
    'usrp': False,
    'biopac': False
}

# Variável global para armazenar resultados PASAT
pasat_session_results = {
    'test1': None,
    'test2': None,
    'test3': None
}

# ============================================================================
# XENICS RECORDER
# ============================================================================

class XenicsRecorder:
    def __init__(self, session_dir):
        self.session_dir = Path(session_dir)
        self.xenics_dir = self.session_dir / "xenics"
        
        self.tiff_dir = self.xenics_dir / "tiff"
        self.png_dir = self.xenics_dir / "png"
        self.npy_dir = self.xenics_dir / "npy"
        
        self.tiff_dir.mkdir(parents=True, exist_ok=True)
        self.png_dir.mkdir(parents=True, exist_ok=True)
        self.npy_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.lib = ctypes.WinDLL(XENICS_DLL)
            self._setup_api()
            print("[XENICS] ✅ DLL carregada")
        except Exception as e:
            print(f"[XENICS] ⚠️  Não foi possível carregar DLL: {e}")
            self.lib = None
        
        self.h_cam = None
        self.recording = False
        self.frame_count = 0
        self.markers = []
        self.t0 = None
        self.video_writer = None
        self.recording_thread = None
        self.error_count = 0
        self.success_count = 0
        
    def _setup_api(self):
        """Configurar protótipos da API Xenics"""
        XCHANDLE = c_void_p
        ErrCode = c_int
        StatusCallbackType = CFUNCTYPE(c_int, c_void_p, c_int, c_ulong, c_ulong)
        
        self.lib.XC_GetDeviceList.argtypes = [c_char_p, c_int]
        self.lib.XC_GetDeviceList.restype = ErrCode
        
        self.lib.XC_OpenCamera.argtypes = [c_char_p, StatusCallbackType, c_void_p]
        self.lib.XC_OpenCamera.restype = XCHANDLE
        
        self.lib.XC_CloseCamera.argtypes = [XCHANDLE]
        self.lib.XC_CloseCamera.restype = None
        
        self.lib.XC_GetWidth.argtypes = [XCHANDLE]
        self.lib.XC_GetWidth.restype = c_int
        
        self.lib.XC_GetHeight.argtypes = [XCHANDLE]
        self.lib.XC_GetHeight.restype = c_int
        
        self.lib.XC_GetFrameSize.argtypes = [XCHANDLE]
        self.lib.XC_GetFrameSize.restype = c_int
        
        self.lib.XC_GetMaxValue.argtypes = [XCHANDLE]
        self.lib.XC_GetMaxValue.restype = c_int
        
        self.lib.XC_StartCapture.argtypes = [XCHANDLE]
        self.lib.XC_StartCapture.restype = ErrCode
        
        self.lib.XC_StopCapture.argtypes = [XCHANDLE]
        self.lib.XC_StopCapture.restype = ErrCode
        
        self.lib.XC_GetFrame.argtypes = [XCHANDLE, c_int, c_int, c_void_p, c_int]
        self.lib.XC_GetFrame.restype = ErrCode
        
        def status_cb(user, iMsg, ulP, ulT):
            return 0
        
        self.status_cb = StatusCallbackType(status_cb)
    
    def _get_device_url(self):
        """Obter URL do primeiro dispositivo - VERSÃO CORRIGIDA"""
        if not self.lib:
            raise RuntimeError("DLL Xenics não carregada")
        
        buf = ctypes.create_string_buffer(4096)
        err = self.lib.XC_GetDeviceList(buf, ctypes.sizeof(buf))
        
        # ⚠️ Erro 1 é comum e não é fatal!
        if err != 0:
            print(f"[XENICS] ⚠️  XC_GetDeviceList retornou {err} (ignorando, é comum)")
        
        devices_str = buf.value.decode("ascii", errors="ignore")
        print(f"[XENICS] Dispositivos: '{devices_str}'")
        
        if not devices_str or devices_str.strip() == '':
            raise RuntimeError("❌ Nenhum dispositivo encontrado")
        
        # Parse: "cam://0|Gobi384BDCL50 (...)"
        parts = devices_str.split("|")
        
        if len(parts) < 1:
            raise RuntimeError(f"❌ Formato inválido: {devices_str}")
        
        base_url = parts[0].strip()
        
        if not base_url.startswith("cam://"):
            raise RuntimeError(f"❌ URL inválido: {base_url}")
        
        url_with_bitsize = base_url + "?bitsize=16"
        print(f"[XENICS] URL: {url_with_bitsize}")
        
        return url_with_bitsize
    
    def start_recording(self):
        """Inicia gravação contínua"""
        if not self.lib:
            print("[XENICS] ⚠️  DLL não disponível")
            return False
        
        print("\n[XENICS] 🎬 Iniciando...")
        print("="*60)
        
        try:
            # 1. Obter dispositivo
            print("[XENICS] 1️⃣ A procurar dispositivos...")
            url = self._get_device_url()
            
            # 2. Abrir câmara
            print("[XENICS] 2️⃣ A abrir câmara...")
            self.h_cam = self.lib.XC_OpenCamera(url.encode("ascii"), self.status_cb, None)
            
            if not self.h_cam:
                raise RuntimeError("❌ XC_OpenCamera retornou NULL")
            
            print(f"[XENICS] ✅ Câmara aberta (handle: {self.h_cam})")
            
            # 3. Obter parâmetros
            print("[XENICS] 3️⃣ A obter parâmetros...")
            self.width = self.lib.XC_GetWidth(self.h_cam)
            self.height = self.lib.XC_GetHeight(self.h_cam)
            self.frame_size = self.lib.XC_GetFrameSize(self.h_cam)
            self.max_value = self.lib.XC_GetMaxValue(self.h_cam)
            
            print(f"[XENICS] Resolução: {self.width}x{self.height}")
            print(f"[XENICS] Frame size: {self.frame_size} bytes")
            print(f"[XENICS] Max value: {self.max_value}")
            
            # 4. Iniciar captura
            print("[XENICS] 4️⃣ A iniciar captura...")
            err = self.lib.XC_StartCapture(self.h_cam)
            if err != 0:
                raise RuntimeError(f"❌ XC_StartCapture falhou: {err}")
            
            print("[XENICS] ✅ Captura iniciada")
            
            # 5. Criar VideoWriter
            print("[XENICS] 5️⃣ A criar VideoWriter...")
            video_path = self.xenics_dir / "sequence.avi"
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.video_writer = cv2.VideoWriter(
                str(video_path), fourcc, 13.0, (self.width, self.height), True
            )
            
            if not self.video_writer.isOpened():
                print("[XENICS] ⚠️  VideoWriter não abriu (não é crítico)")
            else:
                print("[XENICS] ✅ VideoWriter criado")
            
            # 6. Inicializar estado
            self.recording = True
            self.frame_count = 0
            self.error_count = 0
            self.success_count = 0
            self.t0 = time.time()
            self.markers = [{'name': 'RECORDING_START', 'time': 0.0, 'frame': 0}]
            
            # 7. Iniciar thread
            print("[XENICS] 6️⃣ A iniciar thread de gravação...")
            self.recording_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.recording_thread.start()
            
            print("[XENICS] ✅ Thread iniciada")
            print("="*60)
            print("[XENICS] ✅✅✅ GRAVAÇÃO INICIADA! ✅✅✅")
            print("="*60)
            
            # Aguardar primeiros frames
            time.sleep(2)
            
            if self.frame_count == 0:
                print(f"[XENICS] ⚠️  AVISO: Nenhum frame nos primeiros 2s!")
                print(f"[XENICS]    Erros: {self.error_count} | Sucessos: {self.success_count}")
            else:
                print(f"[XENICS] ✅ {self.frame_count} frames capturados")
            
            return True
            
        except Exception as e:
            print(f"[XENICS] ❌ ERRO: {e}")
            import traceback
            traceback.print_exc()
            
            if self.h_cam:
                try:
                    self.lib.XC_CloseCamera(self.h_cam)
                except:
                    pass
            
            return False
    
    def _capture_loop(self):
        """Loop de captura contínua"""
        print("[XENICS THREAD] 🎬 Loop iniciado")
        
        frame = np.zeros((self.height, self.width), dtype=np.uint16)
        buf_ptr = frame.ctypes.data_as(c_void_p)
        
        consecutive_errors = 0
        last_log_time = time.time()
        
        while self.recording:
            try:
                err = self.lib.XC_GetFrame(
                    self.h_cam, FT_16_BPP_GRAY, XGF_Blocking, buf_ptr, self.frame_size
                )
                
                if err != 0:
                    self.error_count += 1
                    consecutive_errors += 1
                    
                    if consecutive_errors == 1:
                        print(f"[XENICS THREAD] ⚠️  XC_GetFrame erro: {err}")
                    elif consecutive_errors == 100:
                        print(f"[XENICS THREAD] ❌ 100 erros consecutivos!")
                    
                    time.sleep(0.01)
                    continue
                
                # Sucesso!
                self.success_count += 1
                consecutive_errors = 0
                
                frame_clipped = np.minimum(frame, self.max_value).astype(np.uint16)
                
                # Guardar TIFF
                tiff_path = self.tiff_dir / f"frame_{self.frame_count:04d}.tiff"
                imageio.imwrite(tiff_path, frame_clipped)
                
                # Guardar NPY
                npy_path = self.npy_dir / f"frame_{self.frame_count:04d}.npy"
                np.save(npy_path, frame_clipped)
                
                # Guardar PNG
                png_8bit = (frame_clipped.astype(np.float32) / self.max_value * 255.0)
                png_8bit = np.clip(png_8bit, 0, 255).astype(np.uint8)
                png_path = self.png_dir / f"frame_{self.frame_count:04d}.png"
                imageio.imwrite(png_path, png_8bit)
                
                # Vídeo
                if self.video_writer and self.video_writer.isOpened():
                    frame_bgr = cv2.cvtColor(png_8bit, cv2.COLOR_GRAY2BGR)
                    self.video_writer.write(frame_bgr)
                
                self.frame_count += 1
                
                # Log a cada 10s
                now = time.time()
                if now - last_log_time > 10:
                    print(f"[XENICS THREAD] 📹 {self.frame_count} frames | "
                          f"Taxa: {self.success_count/(self.success_count+self.error_count)*100:.1f}%")
                    last_log_time = now
                
            except Exception as e:
                print(f"[XENICS THREAD] ❌ Exceção: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
        
        print(f"[XENICS THREAD] 🛑 Terminado ({self.frame_count} frames)")
    
    def mark_event(self, event_name, metadata=None):
        """Marca evento temporal"""
        if not self.recording:
            return
        
        t = time.time() - self.t0
        marker = {
            'name': event_name,
            'time': t,
            'frame': self.frame_count,
            'metadata': metadata or {}
        }
        self.markers.append(marker)
        print(f"[XENICS] 📍 {event_name} @ {t:.3f}s (frame {self.frame_count})")
    
    def stop_recording(self):
        """Para gravação"""
        if not self.recording:
            return
        
        print("\n[XENICS] ⏹️  Parando...")
        
        self.mark_event('RECORDING_STOP')
        self.recording = False
        
        if self.recording_thread:
            self.recording_thread.join(timeout=5)
        
        if self.lib and self.h_cam:
            self.lib.XC_StopCapture(self.h_cam)
            self.lib.XC_CloseCamera(self.h_cam)
        
        if self.video_writer:
            self.video_writer.release()
        
        markers_path = self.xenics_dir / "markers.json"
        with open(markers_path, 'w') as f:
            json.dump({
                'markers': self.markers,
                'total_frames': self.frame_count,
                'fps': 13.0,
                'resolution': [self.width, self.height],
                'statistics': {
                    'success_count': self.success_count,
                    'error_count': self.error_count
                }
            }, f, indent=2)
        
        print(f"[XENICS] ✅ {self.frame_count} frames salvos")

# ============================================================================
# USRP RECORDER
# ============================================================================

class USRPRecorder:
    def __init__(self, session_dir):
        self.session_dir = Path(session_dir)
        self.usrp_dir = self.session_dir / "usrp"
        self.usrp_dir.mkdir(parents=True, exist_ok=True)
        
        self.recording = False
        self.markers = []
        self.t0 = None
        self.process = None
        self.output_file = self.usrp_dir / "bioradar_data.dat"
    
    def start_recording(self):
        """Inicia gravação USRP"""
        print("\n[USRP] 🎬 Iniciando...")
        print(f"[USRP] Output file: {self.output_file}")
        
        self._create_gnuradio_script()
        
        if not GNURADIO_SCRIPT.exists():
            print(f"[USRP] ❌ Script não existe: {GNURADIO_SCRIPT}")
            return False
        
        try:
            # Flag -u para desabilitar buffering do Python
            cmd = [
                sys.executable,
                "-u",
                str(GNURADIO_SCRIPT),
                str(self.output_file)
            ]
            
            print(f"[USRP] Comando: {' '.join(cmd)}")
            print("[USRP] Iniciando processo GNU Radio...")
            print("="*60)
            
            # Processo sem captura - output vai direto para console
            self.process = subprocess.Popen(
                cmd, 
                cwd=str(BASE_DIR),
                stdout=None,
                stderr=None
            )
            
            print(f"[USRP] Processo iniciado (PID: {self.process.pid})")
            print("[USRP] Aguardando USRP inicializar (10 segundos)...")
            
            time.sleep(10)
            
            # Verificar se processo ainda está vivo
            poll_result = self.process.poll()
            if poll_result is not None:
                print(f"[USRP] ❌ Processo terminou prematuramente (exit code: {poll_result})")
                return False
            
            print(f"[USRP] ✅ Processo ativo")
            
            # Aguardar mais tempo para flowgraph completar inicialização
            print("[USRP] Aguardando flowgraph completar (mais 10 segundos)...")
            time.sleep(10)
            
            # DIAGNÓSTICO
            print("\n[USRP] === DIAGNÓSTICO ===")
            print(f"[USRP] Processo poll(): {self.process.poll()}")
            
            if self.output_file.exists():
                size = self.output_file.stat().st_size
                print(f"[USRP] Ficheiro após 20s: {size} bytes ({size/(1024*1024):.2f} MB)")
                
                if size > 0:
                    try:
                        with open(self.output_file, 'rb') as f:
                            first_bytes = f.read(100)
                            print(f"[USRP] Primeiros bytes lidos: {len(first_bytes)}")
                            print(f"[USRP] ✅✅✅ FICHEIRO CONTÉM DADOS! ✅✅✅")
                    except Exception as e:
                        print(f"[USRP] ⚠️  Erro ao ler: {e}")
                else:
                    print(f"[USRP] ⚠️  Ficheiro existe mas está vazio")
            else:
                print(f"[USRP] ⚠️  Ficheiro ainda não criado após 20s")
            
            # Verificar processo com psutil (se disponível)
            try:
                import psutil
                current_process = psutil.Process(self.process.pid)
                print(f"[USRP] Processo status: {current_process.status()}")
                print(f"[USRP] Processo nome: {current_process.name()}")
            except ImportError:
                pass
            except Exception as e:
                print(f"[USRP] Info processo: {e}")
            
            print("[USRP] === FIM DIAGNÓSTICO ===\n")
            
            # CRÍTICO: Marcar como recording SEMPRE
            self.recording = True
            self.t0 = time.time()
            self.markers = [{'name': 'RECORDING_START', 'time': 0.0}]
            
            print("[USRP] ✅ Gravação marcada como iniciada")
            print("[USRP] ⚠️  IMPORTANTE: O ficheiro pode demorar alguns segundos a aparecer")
            print("[USRP]    Isto é NORMAL - o GNU Radio escreve em buffer")
            print("="*60)
            return True
            
        except Exception as e:
            print(f"[USRP] ❌ Erro ao iniciar: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_gnuradio_script(self):
        """Cria script GNU Radio headless"""
        script_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BioRadar GNU Radio Script - Headless
Gerado automaticamente pelo pasat_c_sync_master.py
"""

import sys
import signal
import time

print("[GNU Radio] Iniciando BioRadar...")
print(f"[GNU Radio] Output: {sys.argv[1] if len(sys.argv) > 1 else 'N/A'}")
sys.stdout.flush()

try:
    from gnuradio import gr, blocks, analog, uhd
except ImportError as e:
    print(f"[GNU Radio] ❌ Erro ao importar: {e}")
    sys.stdout.flush()
    sys.exit(1)

class BioRadarHeadless(gr.top_block):
    def __init__(self, output_file):
        gr.top_block.__init__(self, "BioRadar Headless")
        
        samp_rate = 100000
        
        try:
            # USRP Source
            self.usrp_source = uhd.usrp_source(
                ",".join(("", '')),
                uhd.stream_args(cpu_format="fc32", args='', channels=list(range(0,1)))
            )
            self.usrp_source.set_samp_rate(samp_rate)
            self.usrp_source.set_center_freq(5800000000, 0)
            self.usrp_source.set_antenna("RX2", 0)
            self.usrp_source.set_gain(20, 0)
            
            print("[GNU Radio] USRP Source configurado")
            sys.stdout.flush()
            
            # USRP Sink
            self.usrp_sink = uhd.usrp_sink(
                ",".join(("", '')),
                uhd.stream_args(cpu_format="fc32", args='', channels=list(range(0,1))),
                ""
            )
            self.usrp_sink.set_samp_rate(samp_rate)
            self.usrp_sink.set_center_freq(5800000000, 0)
            self.usrp_sink.set_antenna("TX/RX", 0)
            self.usrp_sink.set_gain(80, 0)
            
            print("[GNU Radio] USRP Sink configurado")
            sys.stdout.flush()
            
            # Signal source
            self.sig_source = analog.sig_source_c(samp_rate, analog.GR_COS_WAVE, 10000, 1, 0, 0)
            
            # Mixer
            self.multiply = blocks.multiply_conjugate_cc(1)
            
            # File sink
            self.file_sink = blocks.file_sink(gr.sizeof_gr_complex, output_file, False)
            self.file_sink.set_unbuffered(False)
            
            print("[GNU Radio] File sink configurado")
            sys.stdout.flush()
            
            # Conexões
            self.connect((self.sig_source, 0), (self.multiply, 1))
            self.connect((self.sig_source, 0), (self.usrp_sink, 0))
            self.connect((self.usrp_source, 0), (self.multiply, 0))
            self.connect((self.multiply, 0), (self.file_sink, 0))
            
            print("[GNU Radio] ✅ Flowgraph conectado com sucesso")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"[GNU Radio] ❌ Erro na configuração: {e}")
            sys.stdout.flush()
            import traceback
            traceback.print_exc()
            raise

def signal_handler(sig, frame):
    print("\\n[GNU Radio] Parando...")
    sys.stdout.flush()
    sys.exit(0)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("[GNU Radio] ❌ Uso: python bioradar_recorder.py <output_file>")
        sys.stdout.flush()
        sys.exit(1)
    
    output_file = sys.argv[1]
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        tb = BioRadarHeadless(output_file)
        tb.start()
        
        print("[GNU Radio] ✅ BioRadar rodando...")
        print("[GNU Radio] (Ctrl+C para parar)")
        sys.stdout.flush()
        
        # Loop infinito
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\\n[GNU Radio] Interrompido")
        sys.stdout.flush()
        tb.stop()
        tb.wait()
    except Exception as e:
        print(f"[GNU Radio] ❌ Erro: {e}")
        sys.stdout.flush()
        import traceback
        traceback.print_exc()
        sys.exit(1)
'''
        
        with open(GNURADIO_SCRIPT, 'w', encoding='utf-8') as f:
            f.write(script_content)
    
    def mark_event(self, event_name, metadata=None):
        """Marca evento temporal"""
        if not self.recording:
            return
        
        t = time.time() - self.t0
        marker = {
            'name': event_name,
            'time': t,
            'metadata': metadata or {}
        }
        self.markers.append(marker)
        print(f"[USRP] 📍 {event_name} @ {t:.3f}s")
    
    def stop_recording(self):
        """Para gravação com múltiplas tentativas"""
        if not self.recording:
            return
        
        print("\n[USRP] ⏹️  Parando...")
        
        self.mark_event('RECORDING_STOP')
        self.recording = False
        
        if self.process:
            print(f"[USRP] Terminando processo (PID: {self.process.pid})...")
            
            # Terminate
            self.process.terminate()
            
            try:
                print("[USRP] Aguardando processo terminar (3s)...")
                self.process.wait(timeout=3)
                print("[USRP] ✅ Processo terminado gracefully")
            except subprocess.TimeoutExpired:
                print("[USRP] ⚠️  Timeout - tentando kill...")
                
                self.process.kill()
                
                try:
                    self.process.wait(timeout=2)
                    print("[USRP] ✅ Processo killed")
                except subprocess.TimeoutExpired:
                    print("[USRP] ⚠️  Processo não responde")
            
            # Aguardar flush
            print("[USRP] Aguardando flush de dados para disco (3 segundos)...")
            time.sleep(3)
        
        # Verificar tamanho final
        if self.output_file.exists():
            size = self.output_file.stat().st_size
            size_mb = size / (1024 * 1024)
            print(f"[USRP] Ficheiro final: {size} bytes ({size_mb:.2f} MB)")
            
            if size == 0:
                print("[USRP] ❌ ERRO: Ficheiro vazio!")
                print("[USRP]    Possíveis causas:")
                print("[USRP]    - USRP não conectado")
                print("[USRP]    - Processo GNU Radio falhou")
            elif size < 100000:
                print(f"[USRP] ⚠️  AVISO: Ficheiro pequeno ({size} bytes)")
            else:
                print("[USRP] ✅ Ficheiro com dados OK")
        else:
            print("[USRP] ❌ ERRO: Ficheiro não foi criado!")
        
        # Salvar markers
        markers_path = self.usrp_dir / "markers.json"
        try:
            with open(markers_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'markers': self.markers,
                    'sample_rate': 100000,
                    'center_freq': 5800000000,
                    'note': 'Gravação via GNU Radio'
                }, f, indent=2)
            print(f"[USRP] ✅ Markers salvos: {markers_path}")
        except Exception as e:
            print(f"[USRP] ❌ Erro ao salvar markers: {e}")
        
        print("[USRP] ✅ Paragem concluída")

# ============================================================================
# FLASK API
# ============================================================================

xenics_recorder = None
usrp_recorder = None

@app.route('/prepare_session', methods=['POST'])
def prepare_session():
    """Prepara nova sessão experimental"""
    global current_session, pasat_session_results
    
    data = request.json
    participant_id = data.get('participant_id', 'unknown')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_name = f"{participant_id}_{timestamp}"
    session_dir = BASE_DIR / "sessions" / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    
    current_session = {
        'id': session_name,
        'dir': str(session_dir),
        'participant': participant_id,
        'timestamp': timestamp
    }
    
    # Reset resultados PASAT
    pasat_session_results = {
        'test1': None,
        'test2': None,
        'test3': None
    }
    
    print(f"\n{'='*60}")
    print(f"🆕 NOVA SESSÃO: {session_name}")
    print(f"📁 Pasta: {session_dir}")
    print(f"{'='*60}\n")
    
    return jsonify({
        'status': 'session_prepared',
        'session_id': session_name,
        'directory': str(session_dir)
    })

@app.route('/start_devices', methods=['POST'])
def start_devices():
    """Inicia gravação de todos os dispositivos"""
    global xenics_recorder, usrp_recorder, recording_status
    
    if not current_session:
        return jsonify({'status': 'error', 'message': 'Sessão não preparada'}), 400
    
    print(f"\n{'='*60}")
    print("🎬 INICIANDO DISPOSITIVOS")
    print(f"{'='*60}")
    
    try:
        xenics_recorder = XenicsRecorder(current_session['dir'])
        recording_status['xenics'] = xenics_recorder.start_recording()
    except Exception as e:
        print(f"[XENICS] ⚠️  Erro: {e}")
        recording_status['xenics'] = False
    
    try:
        usrp_recorder = USRPRecorder(current_session['dir'])
        recording_status['usrp'] = usrp_recorder.start_recording()
    except Exception as e:
        print(f"[USRP] ⚠️  Erro: {e}")
        recording_status['usrp'] = False
    
    print("\n⚠️  ATENÇÃO: INICIE GRAVAÇÃO NO BIOPAC AGORA!")
    print("   Depois confirme na interface web\n")
    
    return jsonify({
        'status': 'devices_started',
        'recording_status': recording_status
    })

@app.route('/confirm_biopac', methods=['POST'])
def confirm_biopac():
    """Confirma que Biopac está a gravar"""
    recording_status['biopac'] = True
    print("✅ Biopac confirmado!")
    return jsonify({'status': 'biopac_confirmed'})

@app.route('/mark_event', methods=['POST'])
def mark_event():
    """Marca evento em todos os dispositivos"""
    data = request.json
    event_name = data.get('event')
    metadata = data.get('metadata', {})
    
    print(f"\n📍 EVENTO: {event_name}")
    if metadata:
        print(f"   Metadata: {metadata}")
    
    if xenics_recorder:
        xenics_recorder.mark_event(event_name, metadata)
    
    if usrp_recorder:
        usrp_recorder.mark_event(event_name, metadata)
    
    # Mensagem OPCIONAL para Biopac
    if event_name in ['GROUNDTRUTH_START', 'GROUNDTRUTH_END', 'GROUNDTRUTH_FINAL_START', 
                      'GROUNDTRUTH_FINAL_END', 'PASAT1_START', 'PASAT1_END', 
                      'PASAT2_START', 'PASAT2_END', 'PASAT3_START', 'PASAT3_END']:
        print(f"   💡 BIOPAC (opcional): Podes marcar '{event_name}' com Mouse Button 4")
        print(f"      ou deixar Python sincronizar automaticamente depois")
    
    return jsonify({'status': 'event_marked', 'event': event_name})

@app.route('/save_pasat_test_result', methods=['POST'])
def save_pasat_test_result():
    """Salva resultados de um teste PASAT específico"""
    global pasat_session_results
    
    data = request.json
    test_number = data.get('test_number')
    test_results = data.get('results')
    
    if test_number in [1, 2, 3]:
        pasat_session_results[f'test{test_number}'] = test_results
        print(f"[PASAT] ✅ Resultados do teste {test_number} salvos")
        print(f"   Corretas: {test_results.get('correct', 0)}")
        print(f"   Incorretas: {test_results.get('incorrect', 0)}")
        return jsonify({'status': 'saved'})
    else:
        return jsonify({'status': 'error', 'message': 'Número de teste inválido'}), 400

@app.route('/stop_devices', methods=['POST'])
def stop_devices():
    """Para todos os dispositivos"""
    global xenics_recorder, usrp_recorder, recording_status, pasat_session_results
    
    print(f"\n{'='*60}")
    print("⏹️  PARANDO DISPOSITIVOS")
    print(f"{'='*60}")
    
    if xenics_recorder:
        xenics_recorder.stop_recording()
    
    if usrp_recorder:
        usrp_recorder.stop_recording()
    
    if current_session:
        pasat_path = Path(current_session['dir']) / "pasat_results.json"
        with open(pasat_path, 'w') as f:
            json.dump(pasat_session_results, f, indent=2)
        print(f"\n💾 Resultados PASAT salvos: {pasat_path}")
        print(f"   Test 1: {'✅' if pasat_session_results['test1'] else '❌'}")
        print(f"   Test 2: {'✅' if pasat_session_results['test2'] else '❌'}")
        print(f"   Test 3: {'✅' if pasat_session_results['test3'] else '❌'}")
    
    print("\n⚠️  NÃO ESQUEÇA: Pare gravação no Biopac!")
    print(f"   Salve como: {current_session['id']}_biopac.acq")
    print(f"{'='*60}\n")
    
    recording_status = {'xenics': False, 'usrp': False, 'biopac': False}
    
    return jsonify({'status': 'stopped', 'session': current_session['id']})

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 SISTEMA PASAT-C v2.0 - CONTROLO MANUAL")
    print("="*60)
    print(f"📁 Pasta base: {BASE_DIR}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print("📡 Servidor: http://localhost:5000")
    print("🌐 Abra pasat_c_workflow.html no browser")
    print("="*60)
    print("\n💡 INSTRUÇÕES:")
    print("   1. Este script deve ser executado no Miniforge Prompt")
    print("   2. Com o ambiente 'gnuradio' ativo")
    print("   3. Comando: conda activate gnuradio && python pasat_c_sync_master.py")
    print("="*60 + "\n")
    
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)