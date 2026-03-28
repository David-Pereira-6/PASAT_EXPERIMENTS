#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Análise de Dados Sincronizados - PASAT-C v2.0 FINAL
Sincronização MANUAL: Biopac como ÂNCORA
- Descarta dados antes de EXPERIMENT_START
- Alinha USRP/Xenics com markers automáticos + offset fixo 15s
- Adiciona offset para marcar início dos TESTES (não padrões)
Autor: Sistema de Integração Experimental
Data: 2026-02-19
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from matplotlib.patches import Rectangle
from scipy import signal as scipy_signal
from scipy.ndimage import uniform_filter1d

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

BASE_DIR = Path(r"C:/Users/david/Desktop/PASAT_Experiments/sessions")
ANALYSIS_BASE_DIR = Path(r"C:/Users/david/Desktop/PASAT_Experiments/analysis")

# Duração do padrão de calibração (do código HTML)
# 3s countdown + 12s respirações (3×4s) + 5s sustenção + 3s "pode respirar"
PATTERN_DURATION = 23.0  # segundos

# Markers importantes (pares de início/fim)
PHASE_MARKERS = {
    'GROUNDTRUTH': ('GROUNDTRUTH_START', 'GROUNDTRUTH_END'),
    'PASAT1': ('PASAT1_START', 'PASAT1_END'),
    'PASAT2': ('PASAT2_START', 'PASAT2_END'),
    'PASAT3': ('PASAT3_START', 'PASAT3_END'),
    'GROUNDTRUTH_FINAL': ('GROUNDTRUTH_FINAL_START', 'GROUNDTRUTH_FINAL_END')
}

# Cores para fases
PHASE_COLORS = {
    'GROUNDTRUTH': '#3498db',
    'PASAT1': '#e74c3c',
    'PASAT2': '#f39c12',
    'PASAT3': '#9b59b6',
    'GROUNDTRUTH_FINAL': '#1abc9c'
}

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def create_analysis_structure(session_name):
    """Cria estrutura de pastas organizada"""
    session_analysis_dir = ANALYSIS_BASE_DIR / session_name
    
    xenics_dir = session_analysis_dir / "xenics"
    usrp_dir = session_analysis_dir / "usrp"
    biopac_dir = session_analysis_dir / "biopac"
    plots_dir = session_analysis_dir / "plots"
    plots_individual_dir = plots_dir / "individual"
    plots_phases_dir = plots_dir / "phases"
    
    xenics_dir.mkdir(parents=True, exist_ok=True)
    usrp_dir.mkdir(parents=True, exist_ok=True)
    biopac_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    plots_individual_dir.mkdir(parents=True, exist_ok=True)
    plots_phases_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'base': session_analysis_dir,
        'xenics': xenics_dir,
        'usrp': usrp_dir,
        'biopac': biopac_dir,
        'plots': plots_dir,
        'plots_individual': plots_individual_dir,
        'plots_phases': plots_phases_dir
    }

def list_sessions():
    """Lista todas as sessões disponíveis"""
    if not BASE_DIR.exists():
        print(f"❌ Pasta base não encontrada: {BASE_DIR}")
        return []
    
    sessions = [d for d in BASE_DIR.iterdir() if d.is_dir()]
    
    if not sessions:
        print("❌ Nenhuma sessão encontrada!")
        return []
    
    print("\n" + "="*60)
    print("SESSÕES DISPONÍVEIS")
    print("="*60)
    
    for i, session in enumerate(sessions):
        print(f"[{i}] {session.name}")
    
    print("="*60 + "\n")
    return sessions

def load_markers(device_dir):
    """Carrega markers de um dispositivo"""
    markers_file = device_dir / "markers.json"
    
    if not markers_file.exists():
        print(f"⚠️  Markers não encontrados: {markers_file}")
        return None
    
    try:
        with open(markers_file, encoding='utf-8') as f:
            data = json.load(f)
        return data.get('markers', [])
    except Exception as e:
        print(f"❌ Erro ao carregar markers: {e}")
        return None

def find_marker(markers, name):
    """Encontra um marker específico"""
    if not markers:
        return None
    
    for marker in markers:
        if marker['name'] == name:
            return marker
    return None

def find_all_markers(markers):
    """Retorna dicionário com todos os markers importantes"""
    marker_dict = {}
    for phase_name, (start_marker, end_marker) in PHASE_MARKERS.items():
        start = find_marker(markers, start_marker)
        end = find_marker(markers, end_marker)
        if start:
            marker_dict[start_marker] = start
        if end:
            marker_dict[end_marker] = end
    
    # Adicionar EXPERIMENT_START se existir
    experiment_start = find_marker(markers, 'EXPERIMENT_START')
    if experiment_start:
        marker_dict['EXPERIMENT_START'] = experiment_start
    
    return marker_dict

def load_pasat_results(session_dir):
    """Carrega resultados do PASAT"""
    pasat_file = session_dir / "pasat_results.json"
    
    if not pasat_file.exists():
        print(f"⚠️  Resultados PASAT não encontrados")
        return None
    
    try:
        with open(pasat_file, encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Erro ao carregar resultados PASAT: {e}")
        return None

# ============================================================================
# PROCESSAMENTO
# ============================================================================

def process_xenics(session_dir):
    """Processa dados da câmara térmica Xenics - COM DESCARTE DE LIXO"""
    print("\n📹 Processando Xenics...")
    
    xenics_dir = session_dir / "xenics"
    if not xenics_dir.exists():
        print("❌ Pasta Xenics não encontrada")
        return None
    
    markers = load_markers(xenics_dir)
    if not markers:
        return None
    
    marker_dict = find_all_markers(markers)
    
    if not marker_dict:
        print("❌ Nenhum marker importante encontrado")
        return None
    
    print(f"   ✅ {len(marker_dict)} markers encontrados")
    
    # ========================================================================
    # PASSO 1: Procurar EXPERIMENT_START (momento "Continuar para Experiência")
    # ========================================================================
    
    experiment_start_marker = find_marker(markers, 'EXPERIMENT_START')
    
    if experiment_start_marker:
        experiment_start_frame = experiment_start_marker['frame']
        print(f"   🗑️ EXPERIMENT_START encontrado no frame {experiment_start_frame}")
        print(f"   🗑️ A descartar frames 0-{experiment_start_frame} (lixo antes de 'Continuar')")
        
        # Ajustar todos os markers
        for marker in markers:
            marker['frame'] = marker['frame'] - experiment_start_frame
            marker['time'] = marker['time'] - experiment_start_marker['time']
        
        # Reconstruir marker_dict com valores ajustados
        marker_dict = find_all_markers(markers)
        
        frame_start = 0  # Agora começamos do frame 0 (que era EXPERIMENT_START)
    else:
        print("   ⚠️ EXPERIMENT_START não encontrado, usando RECORDING_START")
        experiment_start_frame = 0
        frame_start = 0
    
    # ========================================================================
    # PASSO 2: Determinar range de frames
    # ========================================================================
    
    t_end_marker = find_marker(markers, 'RECORDING_STOP')
    
    if not t_end_marker:
        print("❌ Marker RECORDING_STOP não encontrado")
        return None
    
    frame_end = t_end_marker['frame']
    
    print(f"   📊 Range de frames: {frame_start} → {frame_end} ({frame_end - frame_start} frames)")
    print(f"   ⏱️ Duração: {t_end_marker['time']:.2f}s")
    
    # ========================================================================
    # PASSO 3: Carregar frames
    # ========================================================================
    
    npy_dir = xenics_dir / "npy"
    
    if not npy_dir.exists():
        print(f"   ❌ Pasta npy/ não existe")
        return None
    
    frames_list = []
    
    print(f"   🔄 A carregar frames...")
    
    # Carregar frames a partir de experiment_start_frame (no disco)
    for frame_idx in range(experiment_start_frame, experiment_start_frame + frame_end + 1):
        npy_file = npy_dir / f"frame_{frame_idx:04d}.npy"
        if npy_file.exists():
            try:
                frame = np.load(npy_file)
                frames_list.append(frame)
                
                if len(frames_list) % 100 == 0:
                    print(f"      Carregados: {len(frames_list)} frames...")
            except Exception as e:
                print(f"   ❌ Erro ao carregar {npy_file.name}: {e}")
    
    if not frames_list:
        print("   ❌ Nenhum frame carregado")
        return None
    
    print(f"   ✅ {len(frames_list)} frames carregados")
    
    # ========================================================================
    # PASSO 4: Processar ROI
    # ========================================================================
    
    max_value = 65535
    
    h, w = frames_list[0].shape
    roi_h_start, roi_h_end = h//3, 2*h//3
    roi_w_start, roi_w_end = w//3, 2*w//3
    
    roi_means = []
    for frame in frames_list:
        roi = frame[roi_h_start:roi_h_end, roi_w_start:roi_w_end]
        roi_8bit = (roi.astype(np.float32) / max_value * 255.0)
        roi_8bit = np.clip(roi_8bit, 0, 255)
        roi_means.append(roi_8bit.mean())
    
    fps = 13.0
    time_axis = np.arange(len(roi_means)) / fps
    
    result = {
        'time': time_axis,
        'roi_mean': np.array(roi_means),
        'frames': np.array(frames_list),
        'markers': markers,
        'marker_dict': marker_dict,
        'fps': fps,
        't_start': 0.0,  # Agora sempre começa em 0 (EXPERIMENT_START)
        't_end': t_end_marker['time'],
        'experiment_start_frame': experiment_start_frame  # Guardar para referência
    }
    
    return result

def process_usrp(session_dir):
    """
    Processa dados do USRP (BioRadar)
    COM DESCARTE DE LIXO + Desmodulação
    """
    print("\n📡 Processando USRP...")
    
    usrp_dir = session_dir / "usrp"
    if not usrp_dir.exists():
        print("❌ Pasta USRP não encontrada")
        return None
    
    markers = load_markers(usrp_dir)
    if not markers:
        return None
    
    marker_dict = find_all_markers(markers)
    
    if not marker_dict:
        print("❌ Nenhum marker importante encontrado")
        return None
    
    print(f"   ✅ {len(marker_dict)} markers encontrados")
    
    data_file = usrp_dir / "bioradar_data.dat"
    if not data_file.exists():
        print("❌ Arquivo de dados não encontrado")
        return None
    
    # ========================================================================
    # PASSO 1: Carregar TODOS os dados
    # ========================================================================
    
    print("   📖 A ler dados raw...")
    raw_data = np.fromfile(str(data_file), dtype=np.complex64)
    
    sample_rate = 100000  # 100 kHz
    
    print(f"   ✅ {len(raw_data)} samples carregados")
    
    # ========================================================================
    # PASSO 2: Procurar EXPERIMENT_START e descartar dados antes
    # ========================================================================
    
    experiment_start_marker = find_marker(markers, 'EXPERIMENT_START')
    
    if experiment_start_marker:
        discard_time = experiment_start_marker['time']
        discard_samples = int(discard_time * sample_rate)
        
        print(f"   🗑️ EXPERIMENT_START encontrado em {discard_time:.2f}s")
        print(f"   🗑️ A descartar {discard_samples} samples (lixo antes de 'Continuar')")
        
        # Descartar!
        raw_data = raw_data[discard_samples:]
        
        # Ajustar todos os markers
        for marker in markers:
            marker['time'] = marker['time'] - discard_time
        
        # Reconstruir marker_dict
        marker_dict = find_all_markers(markers)
    else:
        print("   ⚠️ EXPERIMENT_START não encontrado")
        discard_time = 0.0
    
    # ========================================================================
    # PASSO 3: Determinar range de dados
    # ========================================================================
    
    t_start_marker = find_marker(markers, 'RECORDING_START')
    t_end_marker = find_marker(markers, 'RECORDING_STOP')
    
    if not t_start_marker or not t_end_marker:
        print("❌ Markers RECORDING_START/STOP não encontrados")
        return None
    
    sample_start = int(t_start_marker['time'] * sample_rate)
    sample_end = int(t_end_marker['time'] * sample_rate)
    
    sample_start = max(0, sample_start)
    sample_end = min(len(raw_data), sample_end)
    
    front_signal = raw_data[sample_start:sample_end]
    
    print(f"   📊 Range: {sample_start} → {sample_end} ({len(front_signal)} samples)")
    print(f"   ⏱️ Duração: {t_end_marker['time'] - t_start_marker['time']:.2f}s")
    
    # ========================================================================
    # PASSO 4: Desmodulação (atan demodulation - MATLAB)
    # ========================================================================
    
    print("   🔧 A desmodular sinal (atan demodulation)...")
    
    phase_dem_front = np.unwrap(np.angle(front_signal))
    
    l = len(phase_dem_front)
    time_axis = np.arange(l) / sample_rate
    
    print(f"   ✅ Desmodulação concluída ({l} samples)")
    
    magnitude = np.abs(front_signal)
    
    result = {
        'time': time_axis,
        'complex_data': front_signal,
        'magnitude': magnitude,
        'phase_demodulated': phase_dem_front,
        'markers': markers,
        'marker_dict': marker_dict,
        'sample_rate': sample_rate,
        't_start': t_start_marker['time'],
        't_end': t_end_marker['time'],
        'experiment_start_discard': discard_time
    }
    
    return result

def process_biopac(session_dir):
    """Processa dados do Biopac"""
    print("\n💓 Processando Biopac...")
    
    biopac_dir = session_dir / "biopac"
    if not biopac_dir.exists():
        print("⚠️  Pasta Biopac não encontrada")
        return None
    
    acq_files = list(biopac_dir.glob("*.acq"))
    
    if not acq_files:
        print("⚠️  Nenhum ficheiro .acq encontrado")
        return None
    
    acq_file = acq_files[0]
    print(f"   Ficheiro: {acq_file.name}")
    
    try:
        import bioread
        
        data = bioread.read(str(acq_file))
        
        print(f"   ✅ Ficheiro carregado!")
        print(f"   Duração: {data.time_index[-1]:.2f}s")
        
        ecg_channel = None
        resp_channel = None
        
        ecg_names = ['ecg', 'ekg', 'ch1', 'channel 1', 'heart', 'cardiac']
        resp_names = ['resp', 'respiration', 'breathing', 'ch2', 'channel 2', 'airflow']
        
        for channel in data.channels:
            channel_name_lower = channel.name.lower()
            
            if ecg_channel is None:
                if any(name in channel_name_lower for name in ecg_names):
                    ecg_channel = channel
            
            if resp_channel is None:
                if any(name in channel_name_lower for name in resp_names):
                    resp_channel = channel
        
        if ecg_channel is None and len(data.channels) >= 1:
            ecg_channel = data.channels[0]
        
        if resp_channel is None and len(data.channels) >= 2:
            resp_channel = data.channels[1]
        
        time_data = data.time_index
        
        result = {
            'time': time_data.copy(),
            'sample_rate': data.samples_per_second,
            'markers': [],
            'all_channels': [ch.name for ch in data.channels]
        }
        
        if ecg_channel:
            result['ecg'] = ecg_channel.data.copy()
            result['ecg_channel_name'] = ecg_channel.name
        
        if resp_channel:
            result['respiration'] = resp_channel.data.copy()
            result['resp_channel_name'] = resp_channel.name
        
        return result
        
    except ImportError:
        print("❌ bioread não instalado: pip install bioread")
        return None
    except Exception as e:
        print(f"❌ Erro: {e}")
        return None

# ============================================================================
# SINCRONIZAÇÃO MANUAL - 5 PADRÕES
# ============================================================================

def select_5_calibration_patterns(biopac_data, output_dirs):
    """
    Seleção MANUAL dos 5 padrões de calibração
    Ordem: GROUNDTRUTH_START, PASAT1, PASAT2, PASAT3, GROUNDTRUTH_FINAL
    """
    print("\n" + "="*60)
    print("🖱️  SELEÇÃO MANUAL DOS 5 PADRÕES DE CALIBRAÇÃO")
    print("="*60)
    print("\nOrdem de seleção:")
    print("  1. GROUNDTRUTH_START")
    print("  2. PASAT1_START")
    print("  3. PASAT2_START")
    print("  4. PASAT3_START")
    print("  5. GROUNDTRUTH_FINAL_START")
    print("\n⚠️  Clica no INÍCIO de cada padrão (primeira respiração profunda)")
    print("="*60)
    
    if 'respiration' not in biopac_data:
        print("❌ Sinal de respiração não disponível")
        return None
    
    biopac_resp = biopac_data['respiration']
    biopac_time = biopac_data['time']
    
    # Downsample para visualização
    if len(biopac_resp) > 100000:
        downsample = len(biopac_resp) // 50000
        resp_plot = biopac_resp[::downsample]
        time_plot = biopac_time[::downsample]
    else:
        resp_plot = biopac_resp
        time_plot = biopac_time
    
    # Nomes das fases
    phase_names = [
        'GROUNDTRUTH_START',
        'PASAT1_START',
        'PASAT2_START',
        'PASAT3_START',
        'GROUNDTRUTH_FINAL_START'
    ]
    
    phase_colors_list = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
    
    selected_times = []
    
    # Loop para selecionar cada padrão
    for i, phase_name in enumerate(phase_names):
        print(f"\n📍 Selecionando: {phase_name} ({i+1}/5)")
        
        fig, ax = plt.subplots(figsize=(20, 7))
        ax.plot(time_plot, resp_plot, 'b-', linewidth=0.8, alpha=0.6, label='Respiração Biopac')
        
        # Mostrar seleções anteriores
        for j, prev_time in enumerate(selected_times):
            ax.axvline(prev_time, color=phase_colors_list[j], linestyle='--', 
                      linewidth=2, alpha=0.8, label=f'{phase_names[j]} ({prev_time:.1f}s)')
            ax.axvspan(prev_time, prev_time + PATTERN_DURATION, alpha=0.1, color=phase_colors_list[j])
        
        ax.set_xlabel('Tempo (s)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Amplitude Respiração', fontsize=13, fontweight='bold')
        ax.set_title(f'Clica no INÍCIO do padrão: {phase_name}', 
                    fontsize=16, fontweight='bold', color=phase_colors_list[i])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='upper right')
        
        clicked_time = [None]
        
        def onclick(event):
            if event.xdata is not None:
                clicked_time[0] = event.xdata
                # Limpar linhas vermelhas anteriores
                for line in ax.lines[1:]:
                    if line.get_color() == 'red':
                        line.remove()
                
                ax.axvline(event.xdata, color='red', linewidth=3, 
                          label=f'Selecionado: {event.xdata:.2f}s', zorder=10)
                ax.axvspan(event.xdata, event.xdata + PATTERN_DURATION, alpha=0.15, color='red', zorder=5)
                ax.legend(fontsize=10, loc='upper right')
                plt.draw()
                print(f"   ✅ Tempo selecionado: {event.xdata:.2f}s")
                print("   Fecha a janela para confirmar...")
        
        fig.canvas.mpl_connect('button_press_event', onclick)
        
        plt.tight_layout()
        plt.show()
        
        if clicked_time[0] is None:
            print(f"\n❌ Nenhum padrão selecionado para {phase_name}")
            return None
        
        selected_times.append(clicked_time[0])
    
    print("\n" + "="*60)
    print("✅ TODOS OS 5 PADRÕES SELECIONADOS!")
    print("="*60)
    for i, (name, time) in enumerate(zip(phase_names, selected_times)):
        print(f"  {i+1}. {name}: {time:.2f}s")
    print("="*60)
    
    return {
        'GROUNDTRUTH_START': selected_times[0],
        'PASAT1_START': selected_times[1],
        'PASAT2_START': selected_times[2],
        'PASAT3_START': selected_times[3],
        'GROUNDTRUTH_FINAL_START': selected_times[4]
    }


def sync_all_devices_with_calibration_patterns(biopac_data, xenics_data, usrp_data, calibration_times, output_dirs):
    """
    Sincroniza TODOS os dispositivos:
    - BIOPAC é a ÂNCORA (timeline de referência)
    - USRP/Xenics ajustam-se USANDO MARKERS AUTOMÁTICOS + OFFSET FIXO DE 15s
    """
    print("\n" + "="*60)
    print("🔧 SINCRONIZAÇÃO: MARKERS + OFFSET FIXO DE 15s")
    print("="*60)
    
    # Offset fixo para compensar delay entre clique e marker
    FIXED_OFFSET = 13.9  # segundos
    
    # ========================================================================
    # PASSO 1: Ajustar BIOPAC (âncora)
    # ========================================================================
    
    print(f"\n🫁 Ajustando Biopac (ÂNCORA)...")
    
    biopac_t0 = calibration_times['GROUNDTRUTH_START']
    offset_biopac = -biopac_t0
    
    print(f"   Clique manual (início padrão): {biopac_t0:.3f}s")
    print(f"   Offset: {offset_biopac:+.3f}s")
    print(f"   → {biopac_t0:.3f}s vira 0.0s")
    
    biopac_data['time'] = biopac_data['time'] + offset_biopac
    
    biopac_data['calibration_patterns'] = {}
    print("\n   Padrões ajustados:")
    for marker_name, time_original in calibration_times.items():
        time_adjusted = time_original + offset_biopac
        biopac_data['calibration_patterns'][marker_name] = {
            'time_original': time_original,
            'time_adjusted': time_adjusted
        }
        print(f"      {marker_name}: {time_original:.2f}s → {time_adjusted:.2f}s")
    
    # ========================================================================
    # PASSO 2: Ajustar USRP usando marker automático + OFFSET FIXO
    # ========================================================================
    
    print(f"\n📡 Ajustando USRP...")
    print(f"   ⚙️  Aplicando OFFSET FIXO: {FIXED_OFFSET}s")
    
    usrp_marker = find_marker(usrp_data['markers'], 'GROUNDTRUTH_START')
    
    if usrp_marker:
        usrp_marker_time = usrp_marker['time']
        print(f"   ✅ Marker GROUNDTRUTH_START encontrado: {usrp_marker_time:.3f}s")
        
        # Compensar o offset fixo: marker está X segundos DEPOIS do início real do padrão
        usrp_pattern_time = usrp_marker_time - FIXED_OFFSET
        print(f"   📐 Início real do padrão (marker - {FIXED_OFFSET}s): {usrp_pattern_time:.3f}s")
        
        # Alinhar com Biopac (que está em t=0)
        offset_usrp = -usrp_pattern_time
        print(f"   Offset final: {offset_usrp:+.3f}s")
        print(f"   → {usrp_pattern_time:.3f}s vira 0.0s (alinhado com Biopac)")
        
        usrp_data['time'] = usrp_data['time'] + offset_usrp
        usrp_data['sync_offset'] = offset_usrp
        usrp_data['fixed_offset_applied'] = FIXED_OFFSET
        usrp_data['marker_original_time'] = usrp_marker_time
    else:
        print("   ⚠️ GROUNDTRUTH_START não encontrado no USRP")
        print("   Assumindo offset = 0")
        usrp_data['sync_offset'] = 0.0
    
    # ========================================================================
    # PASSO 3: Ajustar XENICS usando marker automático + OFFSET FIXO
    # ========================================================================
    
    if xenics_data:
        print(f"\n📹 Ajustando Xenics...")
        print(f"   ⚙️  Aplicando OFFSET FIXO: {FIXED_OFFSET}s")
        
        xenics_marker = find_marker(xenics_data['markers'], 'GROUNDTRUTH_START')
        
        if xenics_marker:
            xenics_marker_frame = xenics_marker['frame']
            xenics_marker_time = xenics_marker_frame / xenics_data['fps']
            print(f"   ✅ Marker GROUNDTRUTH_START encontrado: frame {xenics_marker_frame} ({xenics_marker_time:.3f}s)")
            
            # Compensar o offset fixo
            xenics_pattern_time = xenics_marker_time - FIXED_OFFSET
            print(f"   📐 Início real do padrão (marker - {FIXED_OFFSET}s): {xenics_pattern_time:.3f}s")
            
            # Alinhar com Biopac
            offset_xenics = -xenics_pattern_time
            print(f"   Offset final: {offset_xenics:+.3f}s")
            print(f"   → {xenics_pattern_time:.3f}s vira 0.0s (alinhado com Biopac)")
            
            xenics_data['time'] = xenics_data['time'] + offset_xenics
            xenics_data['sync_offset'] = offset_xenics
            xenics_data['fixed_offset_applied'] = FIXED_OFFSET
            xenics_data['marker_original_time'] = xenics_marker_time
        else:
            print("   ⚠️ GROUNDTRUTH_START não encontrado no Xenics")
            print("   Assumindo offset = 0")
            xenics_data['sync_offset'] = 0.0
    
    # ========================================================================
    # PASSO 4: Adicionar offset para marcar INÍCIO DOS TESTES
    # ========================================================================
    
    print(f"\n⏱️ Adicionando offset para início dos TESTES...")
    print(f"   Duração do padrão: {PATTERN_DURATION}s")
    print(f"   → t=0 moverá de 'início padrão' para 'início teste'")
    
    # Ajustar TODOS os dispositivos
    biopac_data['time'] = biopac_data['time'] - PATTERN_DURATION
    usrp_data['time'] = usrp_data['time'] - PATTERN_DURATION
    if xenics_data:
        xenics_data['time'] = xenics_data['time'] - PATTERN_DURATION
    
    # Atualizar calibration_patterns
    for marker_name in biopac_data['calibration_patterns']:
        biopac_data['calibration_patterns'][marker_name]['time_adjusted'] -= PATTERN_DURATION
    
    print(f"   ✅ Offset aplicado: t=0 agora = INÍCIO DO TESTE")
    
    # ========================================================================
    # PASSO 5: VERIFICAÇÃO
    # ========================================================================
    
    print("\n" + "="*60)
    print("📊 RESUMO DA SINCRONIZAÇÃO:")
    print("="*60)
    
    print(f"\n{'Dispositivo':<15} {'Marker Time':<15} {'Fixed Offset':<15} {'Pattern Start':<15} {'Final Offset':<15}")
    print("-"*75)
    print(f"{'Biopac':<15} {'-':<15} {'-':<15} {'0.0s':<15} {offset_biopac:<15.3f}")
    
    if usrp_data.get('marker_original_time'):
        pattern_start = usrp_data['marker_original_time'] - FIXED_OFFSET
        print(f"{'USRP':<15} {usrp_data['marker_original_time']:<15.3f} {FIXED_OFFSET:<15.1f} {pattern_start:<15.3f} {usrp_data['sync_offset']:<15.3f}")
    
    if xenics_data and xenics_data.get('marker_original_time'):
        pattern_start = xenics_data['marker_original_time'] - FIXED_OFFSET
        print(f"{'Xenics':<15} {xenics_data['marker_original_time']:<15.3f} {FIXED_OFFSET:<15.1f} {pattern_start:<15.3f} {xenics_data['sync_offset']:<15.3f}")
    
    print("-"*75)
    print(f"\n   💡 Interpretação:")
    print(f"      1. Marker está {FIXED_OFFSET}s DEPOIS do início do padrão")
    print(f"      2. Subtraímos {FIXED_OFFSET}s do marker para obter início real")
    print(f"      3. Alinhamos esse início com o clique manual do Biopac (t=0)")
    print(f"      4. Depois subtraímos {PATTERN_DURATION}s para marcar início do teste")
    
    print("="*60)
    
    biopac_data['sync_method'] = f'markers_with_fixed_offset_{FIXED_OFFSET}s'
    biopac_data['sync_offset'] = offset_biopac
    biopac_data['sync_confidence'] = 'Alta (markers + offset fixo manual)'
    biopac_data['pattern_duration'] = PATTERN_DURATION
    biopac_data['fixed_offset_seconds'] = FIXED_OFFSET
    
    # ========================================================================
    # PASSO 6: PLOT DE VERIFICAÇÃO
    # ========================================================================
    
    try:
        n_plots = 0
        if biopac_data: n_plots += 1
        if usrp_data: n_plots += 1
        if xenics_data: n_plots += 1
        
        if n_plots == 0:
            return biopac_data
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(20, 4*n_plots))
        
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Biopac
        if biopac_data:
            if len(biopac_data['time']) > 100000:
                downsample = len(biopac_data['time']) // 50000
                t_bio = biopac_data['time'][::downsample]
                resp_bio = biopac_data['respiration'][::downsample]
            else:
                t_bio = biopac_data['time']
                resp_bio = biopac_data['respiration']
            
            axes[plot_idx].plot(t_bio, resp_bio, 'b-', linewidth=0.6, alpha=0.7)
            axes[plot_idx].set_title('🫁 Biopac Respiração (ÂNCORA)', fontsize=12, fontweight='bold')
            axes[plot_idx].set_ylabel('Amplitude', fontsize=10)
            plot_idx += 1
        
        # USRP
        if usrp_data:
            downsample_usrp = 100
            t_usrp = usrp_data['time'][::downsample_usrp]
            phase_usrp = usrp_data['phase_demodulated'][::downsample_usrp]
            axes[plot_idx].plot(t_usrp, phase_usrp, 'g-', linewidth=0.6, alpha=0.7)
            axes[plot_idx].set_title(f'📡 USRP BioRadar (Offset fixo: {FIXED_OFFSET}s)', fontsize=12, fontweight='bold')
            axes[plot_idx].set_ylabel('Phase (Radians)', fontsize=10)
            plot_idx += 1
        
        # Xenics
        if xenics_data:
            axes[plot_idx].plot(xenics_data['time'], xenics_data['roi_mean'], 'r-', linewidth=0.8, alpha=0.7)
            axes[plot_idx].set_title(f'📹 Xenics Thermal (Offset fixo: {FIXED_OFFSET}s)', fontsize=12, fontweight='bold')
            axes[plot_idx].set_ylabel('Intensidade', fontsize=10)
            axes[plot_idx].set_xlabel('Tempo (s) - Timeline Sincronizada', fontsize=11, fontweight='bold')
            plot_idx += 1
        
        # Marcar t=0 (início do TESTE)
        for ax in axes:
            ax.axvline(0, color='black', linestyle='-', linewidth=3, alpha=0.4, 
                      label='t=0 (INÍCIO TESTE)')
        
        # Marcar início do PADRÃO (-23s)
        for ax in axes:
            ax.axvline(-PATTERN_DURATION, color='gray', linestyle='--', linewidth=2, alpha=0.5,
                      label=f'Início padrão (-{PATTERN_DURATION}s)')
        
        # Marcar os 5 testes
        colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
        marker_names = ['GROUNDTRUTH_START', 'PASAT1_START', 'PASAT2_START', 
                       'PASAT3_START', 'GROUNDTRUTH_FINAL_START']
        
        for i, (marker_name, color) in enumerate(zip(marker_names, colors)):
            if marker_name in biopac_data['calibration_patterns']:
                time_test = biopac_data['calibration_patterns'][marker_name]['time_adjusted']
                
                for j, ax in enumerate(axes):
                    if j == 0:
                        ax.axvline(time_test, color=color, linestyle='--', linewidth=2, alpha=0.7,
                                 label=marker_name)
                    else:
                        ax.axvline(time_test, color=color, linestyle='--', linewidth=2, alpha=0.7)
        
        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='upper right')
        
        plt.suptitle(f'Sincronização: Markers + Offset Fixo {FIXED_OFFSET}s', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        debug_file = output_dirs['plots'] / "sync_verification_fixed_offset.png"
        plt.savefig(debug_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n📊 Plot salvo: {debug_file.name}")
        
    except Exception as e:
        print(f"⚠️  Erro ao criar plot: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    
    return biopac_data


def auto_sync_biopac_intelligent(biopac_data, xenics_data, usrp_data, output_dirs):
    """
    Sincronização usando seleção MANUAL dos 5 padrões
    """
    print("\n🔄 Sincronização por Seleção Manual de Padrões...")
    
    if not biopac_data:
        print("❌ Biopac não disponível")
        return biopac_data
    
    if 'respiration' not in biopac_data:
        print("❌ Sinal de respiração não disponível no Biopac")
        return biopac_data
    
    # Selecionar os 5 padrões manualmente
    calibration_times = select_5_calibration_patterns(biopac_data, output_dirs)
    
    if calibration_times is None:
        print("\n❌ Seleção manual cancelada")
        return biopac_data
    
    # Sincronizar todos os dispositivos
    biopac_data = sync_all_devices_with_calibration_patterns(
        biopac_data, xenics_data, usrp_data, calibration_times, output_dirs
    )
    
    return biopac_data

# ============================================================================
# VISUALIZAÇÃO - PERFORMANCE PASAT
# ============================================================================

def plot_pasat_performance(pasat_results, output_dirs):
    """
    Visualiza performance nos 3 testes PASAT de forma clara
    - Gráfico de barras com corretas/incorretas/omissões
    - Gráfico de precisão (%)
    - Detalhes de cada teste
    """
    if not pasat_results:
        print("⚠️  Resultados PASAT não disponíveis")
        return
    
    print("\n📊 Gerando visualização de performance PASAT...")
    
    tests = ['test1', 'test2', 'test3']
    labels = ['PASAT 1\n(3.5s)', 'PASAT 2\n(2.5s)', 'PASAT 3\n(1.5s)']
    
    corretas = []
    incorretas = []
    omissoes = []
    accuracies = []
    tempos_resposta = []
    
    for test_key in tests:
        if test_key in pasat_results:
            test_data = pasat_results[test_key]
            correct = test_data.get('correct', 0)
            incorrect = test_data.get('incorrect', 0)
            omissions = test_data.get('omissions', 0)
            mean_rt = test_data.get('meanRT', 0)
            
            corretas.append(correct)
            incorretas.append(incorrect)
            omissoes.append(omissions)
            tempos_resposta.append(mean_rt)
            
            total = correct + incorrect
            acc = (correct / total * 100) if total > 0 else 0
            accuracies.append(acc)
    
    # Criar figura com 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # ========================================================================
    # PLOT 1: Barras empilhadas (Corretas/Incorretas/Omissões)
    # ========================================================================
    x = np.arange(len(labels))
    width = 0.6
    
    p1 = ax1.bar(x, corretas, width, label='✓ Corretas', color='#27ae60', edgecolor='white', linewidth=2)
    p2 = ax1.bar(x, incorretas, width, bottom=corretas, label='✗ Incorretas', 
                 color='#e74c3c', edgecolor='white', linewidth=2)
    bottom = np.array(corretas) + np.array(incorretas)
    p3 = ax1.bar(x, omissoes, width, bottom=bottom, label='⊘ Omissões', 
                 color='#95a5a6', edgecolor='white', linewidth=2)
    
    # Adicionar valores nas barras
    for i, (c, inc, om) in enumerate(zip(corretas, incorretas, omissoes)):
        # Corretas
        if c > 0:
            ax1.text(i, c/2, str(c), ha='center', va='center', 
                    fontweight='bold', fontsize=12, color='white')
        # Incorretas
        if inc > 0:
            ax1.text(i, c + inc/2, str(inc), ha='center', va='center',
                    fontweight='bold', fontsize=12, color='white')
        # Omissões
        if om > 0:
            ax1.text(i, c + inc + om/2, str(om), ha='center', va='center',
                    fontweight='bold', fontsize=12, color='white')
    
    ax1.set_ylabel('Número de Respostas', fontsize=13, fontweight='bold')
    ax1.set_title('Distribuição de Respostas', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # PLOT 2: Precisão (%)
    # ========================================================================
    colors = ['#3498db', '#f39c12', '#9b59b6']
    bars = ax2.bar(x, accuracies, width, color=colors, alpha=0.8, 
                   edgecolor='white', linewidth=2)
    
    # Adicionar valores nas barras
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{acc:.1f}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=13)
    
    ax2.set_ylabel('Precisão (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Precisão por Teste', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_ylim(0, 110)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, 
               linewidth=2, label='Chance (50%)')
    ax2.axhline(y=75, color='orange', linestyle='--', alpha=0.5,
               linewidth=2, label='Bom (75%)')
    ax2.legend(fontsize=10)
    
    # ========================================================================
    # PLOT 3: Tempo médio de resposta
    # ========================================================================
    bars = ax3.bar(x, tempos_resposta, width, color=colors, alpha=0.8,
                   edgecolor='white', linewidth=2)
    
    # Adicionar valores nas barras
    for bar, rt in zip(bars, tempos_resposta):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{rt:.2f}s', ha='center', va='bottom',
                fontweight='bold', fontsize=13)
    
    ax3.set_ylabel('Tempo Médio (s)', fontsize=13, fontweight='bold')
    ax3.set_title('Tempo de Resposta', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Linhas de referência do intervalo
    ax3.axhline(y=3.5, color='#3498db', linestyle=':', alpha=0.4, linewidth=2)
    ax3.axhline(y=2.5, color='#f39c12', linestyle=':', alpha=0.4, linewidth=2)
    ax3.axhline(y=1.5, color='#9b59b6', linestyle=':', alpha=0.4, linewidth=2)
    
    plt.suptitle('📝 Performance PASAT-C', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_file = output_dirs['plots'] / "pasat_performance.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ pasat_performance.png")
    
    # ========================================================================
    # Imprimir resumo no terminal
    # ========================================================================
    print("\n" + "="*60)
    print("📊 RESUMO DE PERFORMANCE PASAT")
    print("="*60)
    
    for i, test_key in enumerate(tests):
        if test_key in pasat_results:
            test_data = pasat_results[test_key]
            print(f"\n{labels[i].replace(chr(10), ' ')}:")
            print(f"  ✓ Corretas:   {corretas[i]:3d}")
            print(f"  ✗ Incorretas: {incorretas[i]:3d}")
            print(f"  ⊘ Omissões:   {omissoes[i]:3d}")
            print(f"  📊 Precisão:  {accuracies[i]:5.1f}%")
            print(f"  ⏱️  Tempo RT:   {tempos_resposta[i]:5.2f}s")
    
    print("="*60 + "\n")

# ============================================================================
# VISUALIZAÇÃO - INDIVIDUAIS
# ============================================================================

def plot_individual_devices(xenics_data, usrp_data, biopac_data, session_name, output_dirs):
    """Gráficos individuais de cada dispositivo"""
    print("\n📊 Gerando gráficos individuais...")
    
    # USRP - Fase desmodulada
    if usrp_data and 'phase_demodulated' in usrp_data:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        downsample = 100
        t_down = usrp_data['time'][::downsample]
        phase_down = usrp_data['phase_demodulated'][::downsample]
        
        ax.plot(t_down, phase_down, linewidth=0.8, color='#16a085', label='Demodulated')
        ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Phase (Radians)', fontsize=12, fontweight='bold')
        ax.set_title('📡 USRP BioRadar - Atan Demodulation', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        ax.axvline(0, color='black', linestyle='-', linewidth=2, alpha=0.3, label='t=0 (início teste)')
        
        if biopac_data and 'calibration_patterns' in biopac_data:
            colors_list = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
            for i, (marker_name, times) in enumerate(biopac_data['calibration_patterns'].items()):
                time_adjusted = times['time_adjusted']
                color = colors_list[i % len(colors_list)]
                ax.axvline(time_adjusted, color=color, linestyle='--', alpha=0.5, linewidth=1.5, label=marker_name)
        
        ax.legend(fontsize=9)
        plt.tight_layout()
        output_file = output_dirs['plots_individual'] / "usrp_phase_demodulated.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   ✅ usrp_phase_demodulated.png")
        plt.close()
    
    # Xenics
    if xenics_data:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(xenics_data['time'], xenics_data['roi_mean'], linewidth=1, color='#2c3e50')
        ax.set_xlabel('Tempo (s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Intensidade Pixel', fontsize=12, fontweight='bold')
        ax.set_title('📹 Xenics - Câmara Térmica', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        ax.axvline(0, color='black', linestyle='-', linewidth=2, alpha=0.3, label='t=0')
        
        if biopac_data and 'calibration_patterns' in biopac_data:
            colors_list = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
            for i, (marker_name, times) in enumerate(biopac_data['calibration_patterns'].items()):
                time_adjusted = times['time_adjusted']
                color = colors_list[i % len(colors_list)]
                ax.axvline(time_adjusted, color=color, linestyle='--', alpha=0.5, linewidth=1.5)
        
        ax.legend()
        plt.tight_layout()
        output_file = output_dirs['plots_individual'] / "xenics.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   ✅ xenics.png")
        plt.close()
    
    # Biopac Respiração
    if biopac_data and 'respiration' in biopac_data:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        if len(biopac_data['time']) > 100000:
            downsample = len(biopac_data['time']) // 50000
            t_down = biopac_data['time'][::downsample]
            resp_down = biopac_data['respiration'][::downsample]
        else:
            t_down = biopac_data['time']
            resp_down = biopac_data['respiration']
        
        ax.plot(t_down, resp_down, linewidth=0.8, color='#27ae60')
        ax.set_xlabel('Tempo (s)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Amplitude', fontsize=13, fontweight='bold')
        ax.set_title(f'🫁 Biopac - Respiração (ÂNCORA TEMPORAL)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        ax.axvline(0, color='black', linestyle='-', linewidth=3, alpha=0.4, label='t=0 (início teste)')
        
        if 'calibration_patterns' in biopac_data:
            colors_list = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
            for i, (marker_name, times) in enumerate(biopac_data['calibration_patterns'].items()):
                time_adjusted = times['time_adjusted']
                color = colors_list[i % len(colors_list)]
                ax.axvline(time_adjusted, color=color, linestyle='--', alpha=0.7, 
                          linewidth=2, label=marker_name)
        
        ax.legend(fontsize=9)
        plt.tight_layout()
        output_file = output_dirs['plots_individual'] / "biopac_respiration.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   ✅ biopac_respiration.png")
        plt.close()
    
    # Biopac ECG
    if biopac_data and 'ecg' in biopac_data:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        if len(biopac_data['time']) > 100000:
            downsample = len(biopac_data['time']) // 50000
            t_down = biopac_data['time'][::downsample]
            ecg_down = biopac_data['ecg'][::downsample]
        else:
            t_down = biopac_data['time']
            ecg_down = biopac_data['ecg']
        
        ax.plot(t_down, ecg_down, linewidth=0.6, color='#c0392b')
        ax.set_xlabel('Tempo (s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('ECG (mV)', fontsize=12, fontweight='bold')
        ax.set_title(f'💓 Biopac - ECG', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        ax.axvline(0, color='black', linestyle='-', linewidth=2, alpha=0.3, label='t=0')
        
        if 'calibration_patterns' in biopac_data:
            colors_list = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
            for i, (marker_name, times) in enumerate(biopac_data['calibration_patterns'].items()):
                time_adjusted = times['time_adjusted']
                color = colors_list[i % len(colors_list)]
                ax.axvline(time_adjusted, color=color, linestyle='--', alpha=0.5, linewidth=1.5)
        
        ax.legend()
        plt.tight_layout()
        output_file = output_dirs['plots_individual'] / "biopac_ecg.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   ✅ biopac_ecg.png")
        plt.close()

# ============================================================================
# VISUALIZAÇÃO - TODOS SINCRONIZADOS
# ============================================================================

def plot_all_synchronized(xenics_data, usrp_data, biopac_data, session_name, output_dirs):
    """Gráfico com TODOS os sinais sincronizados"""
    print("\n📊 Gerando gráfico completo sincronizado...")
    
    n_plots = 0
    if xenics_data: n_plots += 1
    if usrp_data: n_plots += 1
    if biopac_data:
        if 'ecg' in biopac_data: n_plots += 1
        if 'respiration' in biopac_data: n_plots += 1
    
    if n_plots == 0:
        return
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(18, 4*n_plots))
    
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Xenics
    if xenics_data:
        ax = axes[plot_idx]
        ax.plot(xenics_data['time'], xenics_data['roi_mean'], linewidth=1, color='#2c3e50')
        ax.set_ylabel('Intensidade Pixel', fontsize=11, fontweight='bold')
        ax.set_title('📹 Xenics', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        ax.axvline(0, color='black', linestyle='-', linewidth=2, alpha=0.3)
        
        if biopac_data and 'calibration_patterns' in biopac_data:
            for marker_name, times in biopac_data['calibration_patterns'].items():
                time = times['time_adjusted']
                color = PHASE_COLORS.get(marker_name.replace('_START', ''), '#95a5a6')
                ax.axvline(time, color=color, linestyle='--', alpha=0.6, linewidth=1.5)
        
        plot_idx += 1
    
    # USRP
    if usrp_data and 'phase_demodulated' in usrp_data:
        ax = axes[plot_idx]
        
        downsample = 100
        t_down = usrp_data['time'][::downsample]
        phase_down = usrp_data['phase_demodulated'][::downsample]
        
        ax.plot(t_down, phase_down, linewidth=0.7, color='#16a085')
        ax.set_ylabel('Phase (Radians)', fontsize=11, fontweight='bold')
        ax.set_title('📡 USRP BioRadar (Desmodulado)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        ax.axvline(0, color='black', linestyle='-', linewidth=2, alpha=0.3)
        
        if biopac_data and 'calibration_patterns' in biopac_data:
            for marker_name, times in biopac_data['calibration_patterns'].items():
                time = times['time_adjusted']
                color = PHASE_COLORS.get(marker_name.replace('_START', ''), '#95a5a6')
                ax.axvline(time, color=color, linestyle='--', alpha=0.6, linewidth=1.5)
        
        plot_idx += 1
    
    # Biopac ECG
    if biopac_data and 'ecg' in biopac_data:
        ax = axes[plot_idx]
        
        if len(biopac_data['time']) > 100000:
            downsample = len(biopac_data['time']) // 50000
            t_down = biopac_data['time'][::downsample]
            ecg_down = biopac_data['ecg'][::downsample]
        else:
            t_down = biopac_data['time']
            ecg_down = biopac_data['ecg']
        
        ax.plot(t_down, ecg_down, linewidth=0.6, color='#c0392b')
        ax.set_ylabel('ECG (mV)', fontsize=11, fontweight='bold')
        ax.set_title(f'💓 Biopac ECG', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        ax.axvline(0, color='black', linestyle='-', linewidth=2, alpha=0.3)
        
        if 'calibration_patterns' in biopac_data:
            for marker_name, times in biopac_data['calibration_patterns'].items():
                time = times['time_adjusted']
                color = PHASE_COLORS.get(marker_name.replace('_START', ''), '#95a5a6')
                ax.axvline(time, color=color, linestyle='--', alpha=0.6, linewidth=1.5)
        
        plot_idx += 1
    
    # Biopac Respiração
    if biopac_data and 'respiration' in biopac_data:
        ax = axes[plot_idx]
        
        if len(biopac_data['time']) > 100000:
            downsample = len(biopac_data['time']) // 50000
            t_down = biopac_data['time'][::downsample]
            resp_down = biopac_data['respiration'][::downsample]
        else:
            t_down = biopac_data['time']
            resp_down = biopac_data['respiration']
        
        ax.plot(t_down, resp_down, linewidth=0.8, color='#27ae60')
        ax.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
        ax.set_title(f'🫁 Biopac Respiração (ÂNCORA)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        ax.axvline(0, color='black', linestyle='-', linewidth=3, alpha=0.4)
        
        if 'calibration_patterns' in biopac_data:
            colors_list = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
            for i, (marker_name, times) in enumerate(biopac_data['calibration_patterns'].items()):
                time = times['time_adjusted']
                color = colors_list[i % len(colors_list)]
                ax.axvline(time, color=color, linestyle='--', alpha=0.7, linewidth=2)
        
        plot_idx += 1
    
    axes[-1].set_xlabel('Tempo (segundos) - t=0 = Início do Teste', fontsize=12, fontweight='bold')
    
    sync_text = f"Sessão: {session_name}\n"
    if biopac_data and 'sync_method' in biopac_data:
        sync_text += f"Sincronização: Biopac como Âncora + Offset {PATTERN_DURATION}s"
    
    plt.suptitle(sync_text, fontsize=13, fontweight='bold', y=0.998)
    plt.tight_layout()
    
    output_file = output_dirs['plots'] / "all_synchronized.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✅ all_synchronized.png")
    
    plt.show()

# ============================================================================
# PLOTS POR FASE
# ============================================================================

def plot_individual_phases(xenics_data, usrp_data, biopac_data, session_name, output_dirs):
    """
    Cria plots individuais para cada fase de teste
    Cada fase tem o seu próprio t=0 (início do teste após padrão)
    """
    print("\n📊 Gerando plots por fase (apenas zona relevante, SEM padrão)...")
    
    if not biopac_data or 'calibration_patterns' not in biopac_data:
        print("   ⚠️ Padrões de calibração não disponíveis")
        return
    
    # Detectar modo
    t_min = biopac_data['time'].min()
    t_max = biopac_data['time'].max()
    duracao_total = t_max - t_min
    
    print(f"\n   Duração total: {duracao_total:.1f}s ({duracao_total/60:.1f} min)")
    
    if duracao_total > 300:
        modo = 'REAL'
        duracao_groundtruth = 300
        duracao_pasat = 150
        print(f"   ✅ Modo: REAL (GT: 5min, PASAT: 2.5min)")
    else:
        modo = 'TESTE'
        duracao_groundtruth = 20
        duracao_pasat = 20
        print(f"   ✅ Modo: TESTE (GT: 20s, PASAT: 20s)")
    
    fases = [
        {
            'name': 'Groundtruth Inicial',
            'marker_start': 'GROUNDTRUTH_START',
            'duracao': duracao_groundtruth,
            'color': '#3498db',
            'filename': '01_groundtruth_initial.png'
        },
        {
            'name': 'PASAT 1 (3.5s)',
            'marker_start': 'PASAT1_START',
            'duracao': duracao_pasat,
            'color': '#e74c3c',
            'filename': '02_pasat1.png'
        },
        {
            'name': 'PASAT 2 (2.5s)',
            'marker_start': 'PASAT2_START',
            'duracao': duracao_pasat,
            'color': '#f39c12',
            'filename': '03_pasat2.png'
        },
        {
            'name': 'PASAT 3 (1.5s)',
            'marker_start': 'PASAT3_START',
            'duracao': duracao_pasat,
            'color': '#9b59b6',
            'filename': '04_pasat3.png'
        },
        {
            'name': 'Groundtruth Final',
            'marker_start': 'GROUNDTRUTH_FINAL_START',
            'duracao': duracao_groundtruth,
            'color': '#1abc9c',
            'filename': '05_groundtruth_final.png'
        }
    ]
    
    for fase in fases:
        marker_start_name = fase['marker_start']
        
        if marker_start_name not in biopac_data['calibration_patterns']:
            print(f"   ⚠️ {fase['name']}: Marker não encontrado")
            continue
        
        # Tempo do marker (início do PADRÃO)
        t_marker_global = biopac_data['calibration_patterns'][marker_start_name]['time_adjusted']
        
        # Adicionar 23s para pular o padrão
        t_start_global = t_marker_global + PATTERN_DURATION
        t_end_global = t_start_global + fase['duracao']
        
        print(f"\n   📌 {fase['name']}:")
        print(f"      Marker: {t_marker_global:.1f}s")
        print(f"      Teste: {t_start_global:.1f}s → {t_end_global:.1f}s")
        
        n_plots = 0
        if xenics_data: n_plots += 1
        if usrp_data: n_plots += 1
        if biopac_data:
            if 'ecg' in biopac_data: n_plots += 1
            if 'respiration' in biopac_data: n_plots += 1
        
        if n_plots == 0:
            continue
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(16, 3*n_plots))
        
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Xenics
        if xenics_data:
            ax = axes[plot_idx]
            mask = (xenics_data['time'] >= t_start_global) & (xenics_data['time'] <= t_end_global)
            t_global = xenics_data['time'][mask]
            data_plot = xenics_data['roi_mean'][mask]
            t_local = t_global - t_start_global
            
            if len(t_local) > 0:
                ax.plot(t_local, data_plot, linewidth=1, color='#2c3e50')
                ax.set_ylabel('Intensidade', fontsize=10, fontweight='bold')
                ax.set_title('📹 Xenics', fontsize=11, fontweight='bold')
                ax.axvline(0, color=fase['color'], linestyle='--', linewidth=2, alpha=0.7, label='t=0')
                ax.axvline(fase['duracao'], color='red', linestyle='--', linewidth=2, alpha=0.5, label='Fim')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                ax.set_xlim(-1, fase['duracao'] + 1)
            plot_idx += 1
        
        # USRP
        if usrp_data:
            ax = axes[plot_idx]
            mask = (usrp_data['time'] >= t_start_global) & (usrp_data['time'] <= t_end_global)
            t_global = usrp_data['time'][mask]
            data_plot = usrp_data['phase_demodulated'][mask]
            t_local = t_global - t_start_global
            
            if len(t_local) > 0:
                if len(t_local) > 50000:
                    downsample = len(t_local) // 25000
                    t_local = t_local[::downsample]
                    data_plot = data_plot[::downsample]
                
                ax.plot(t_local, data_plot, linewidth=0.7, color='#16a085')
                ax.set_ylabel('Phase', fontsize=10, fontweight='bold')
                ax.set_title('📡 USRP', fontsize=11, fontweight='bold')
                ax.axvline(0, color=fase['color'], linestyle='--', linewidth=2, alpha=0.7, label='t=0')
                ax.axvline(fase['duracao'], color='red', linestyle='--', linewidth=2, alpha=0.5, label='Fim')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                ax.set_xlim(-1, fase['duracao'] + 1)
            plot_idx += 1
        
        # Biopac ECG
        if biopac_data and 'ecg' in biopac_data:
            ax = axes[plot_idx]
            mask = (biopac_data['time'] >= t_start_global) & (biopac_data['time'] <= t_end_global)
            t_global = biopac_data['time'][mask]
            data_plot = biopac_data['ecg'][mask]
            t_local = t_global - t_start_global
            
            if len(t_local) > 0:
                if len(t_local) > 50000:
                    downsample = len(t_local) // 25000
                    t_local = t_local[::downsample]
                    data_plot = data_plot[::downsample]
                
                ax.plot(t_local, data_plot, linewidth=0.6, color='#c0392b')
                ax.set_ylabel('ECG', fontsize=10, fontweight='bold')
                ax.set_title('💓 Biopac ECG', fontsize=11, fontweight='bold')
                ax.axvline(0, color=fase['color'], linestyle='--', linewidth=2, alpha=0.7, label='t=0')
                ax.axvline(fase['duracao'], color='red', linestyle='--', linewidth=2, alpha=0.5, label='Fim')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                ax.set_xlim(-1, fase['duracao'] + 1)
            plot_idx += 1
        
        # Biopac Respiração
        if biopac_data and 'respiration' in biopac_data:
            ax = axes[plot_idx]
            mask = (biopac_data['time'] >= t_start_global) & (biopac_data['time'] <= t_end_global)
            t_global = biopac_data['time'][mask]
            data_plot = biopac_data['respiration'][mask]
            t_local = t_global - t_start_global
            
            if len(t_local) > 0:
                if len(t_local) > 50000:
                    downsample = len(t_local) // 25000
                    t_local = t_local[::downsample]
                    data_plot = data_plot[::downsample]
                
                ax.plot(t_local, data_plot, linewidth=0.8, color='#27ae60')
                ax.set_ylabel('Respiração', fontsize=10, fontweight='bold')
                ax.set_title('🫁 Biopac Respiração', fontsize=11, fontweight='bold')
                ax.axvline(0, color=fase['color'], linestyle='--', linewidth=2, alpha=0.7, label='t=0')
                ax.axvline(fase['duracao'], color='red', linestyle='--', linewidth=2, alpha=0.5, label='Fim')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                ax.set_xlabel('Tempo (s) - t=0 = Início do Teste', fontsize=10, fontweight='bold')
                ax.set_xlim(-1, fase['duracao'] + 1)
            plot_idx += 1
        
        titulo = f'{fase["name"]} - {fase["duracao"]}s - {modo}'
        plt.suptitle(titulo, fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        output_file = output_dirs['plots_phases'] / fase['filename']
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      ✅ {fase['filename']}")
    
    print(f"\n   ✅ Plots salvos em: {output_dirs['plots_phases']}")

# ============================================================================
# RELATÓRIO E EXPORT
# ============================================================================

def create_sync_report(biopac_data, output_dirs):
    """Cria relatório de sincronização"""
    print("\n📄 Gerando relatório de sincronização...")
    
    report_lines = [
        "="*60,
        "RELATÓRIO DE SINCRONIZAÇÃO",
        "="*60,
        ""
    ]
    
    if biopac_data and 'sync_method' in biopac_data:
        report_lines.append(f"Método utilizado: {biopac_data['sync_method']}")
        report_lines.append(f"Confiança: {biopac_data.get('sync_confidence', 'N/A')}")
        
        if 'calibration_patterns' in biopac_data:
            report_lines.append("\nPadrões de calibração selecionados manualmente:")
            report_lines.append("-"*60)
            for marker_name, times in biopac_data['calibration_patterns'].items():
                report_lines.append(f"  {marker_name}:")
                report_lines.append(f"    Tempo original: {times['time_original']:.3f}s")
                report_lines.append(f"    Tempo ajustado: {times['time_adjusted']:.3f}s")
            
            report_lines.append("\nMétodo: Biopac como Âncora Temporal")
            report_lines.append("Workflow:")
            report_lines.append("  1. USRP/Xenics iniciam gravação")
            report_lines.append("  2. Biopac inicia gravação manualmente")
            report_lines.append("  3. Clique em 'Continuar para Experiência'")
            report_lines.append("     → Marca EXPERIMENT_START (descarta lixo antes)")
            report_lines.append("  4. Padrão de calibração realizado")
            report_lines.append("  5. Padrões selecionados manualmente no Biopac (5×)")
            report_lines.append("  6. USRP/Xenics AJUSTAM-SE ao Biopac (âncora)")
            report_lines.append(f"  7. Offset de {biopac_data.get('pattern_duration', 23)}s adicionado")
            report_lines.append("     → t=0 = INÍCIO DO TESTE (não do padrão)")
            
            report_lines.append("\nDesmodulação USRP:")
            report_lines.append("  - Método: Atan demodulation (unwrap(angle(signal)))")
            report_lines.append("  - Equivalente ao código MATLAB original")
    else:
        report_lines.append("Sincronização não realizada")
    
    report_lines.append("")
    report_lines.append("="*60)
    
    report_text = "\n".join(report_lines)
    
    output_file = output_dirs['base'] / "sync_report.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"   ✅ sync_report.txt")

def export_synchronized_data(session_dir, session_name, xenics_data, usrp_data, biopac_data, output_dirs):
    """Exporta dados para CSV"""
    print("\n💾 Exportando CSVs...")
    
    if xenics_data:
        df = pd.DataFrame({
            'time_seconds': xenics_data['time'],
            'roi_mean_intensity': xenics_data['roi_mean']
        })
        output_file = output_dirs['xenics'] / "data.csv"
        df.to_csv(output_file, index=False)
        print(f"   ✅ Xenics: data.csv")
    
    if usrp_data:
        downsample = 100
        
        df = pd.DataFrame({
            'time_seconds': usrp_data['time'][::downsample],
            'phase_demodulated_radians': usrp_data['phase_demodulated'][::downsample],
            'magnitude': usrp_data['magnitude'][::downsample]
        })
        output_file = output_dirs['usrp'] / "data.csv"
        df.to_csv(output_file, index=False)
        print(f"   ✅ USRP: data.csv (fase desmodulada incluída)")
    
    if biopac_data:
        if 'ecg' in biopac_data:
            df = pd.DataFrame({
                'time_seconds': biopac_data['time'],
                'ecg_mv': biopac_data['ecg']
            })
            output_file = output_dirs['biopac'] / "ecg.csv"
            df.to_csv(output_file, index=False)
            print(f"   ✅ Biopac: ecg.csv")
        
        if 'respiration' in biopac_data:
            df = pd.DataFrame({
                'time_seconds': biopac_data['time'],
                'respiration': biopac_data['respiration']
            })
            output_file = output_dirs['biopac'] / "respiration.csv"
            df.to_csv(output_file, index=False)
            print(f"   ✅ Biopac: respiration.csv")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*60)
    print("ANÁLISE DE DADOS - PASAT-C v2.0 FINAL")
    print("Sincronização: Biopac como Âncora + Offset Fixo 15s")
    print("="*60)
    
    sessions = list_sessions()
    
    if not sessions:
        return
    
    try:
        idx = int(input("Seleciona a sessão (número): "))
        session_dir = sessions[idx]
    except (ValueError, IndexError):
        print("❌ Seleção inválida")
        return
    
    session_name = session_dir.name
    
    print(f"\n{'='*60}")
    print(f"PROCESSANDO: {session_name}")
    print(f"{'='*60}")
    
    output_dirs = create_analysis_structure(session_name)
    print(f"\n📁 Análise: {output_dirs['base']}")
    
    # Carregar PASAT
    pasat_results = load_pasat_results(session_dir)
    if pasat_results:
        print("\n📝 Resultados PASAT:")
        for test_key in ['test1', 'test2', 'test3']:
            if pasat_results.get(test_key):
                test_data = pasat_results[test_key]
                correct = test_data.get('correct', 0)
                incorrect = test_data.get('incorrect', 0)
                total = correct + incorrect
                accuracy = (correct / total * 100) if total > 0 else 0
                print(f"   {test_key.upper()}: {correct}/{total} ({accuracy:.1f}%)")
    
    # Processar dispositivos
    xenics_data = process_xenics(session_dir)
    usrp_data = process_usrp(session_dir)
    biopac_data = process_biopac(session_dir)
    
    # Sincronizar
    if biopac_data and usrp_data:
        biopac_data = auto_sync_biopac_intelligent(biopac_data, xenics_data, usrp_data, output_dirs)
    
    # VISUALIZAÇÃO DE PERFORMANCE PASAT
    if pasat_results:
        plot_pasat_performance(pasat_results, output_dirs)
    
    # VISUALIZAÇÕES
    plot_individual_devices(xenics_data, usrp_data, biopac_data, session_name, output_dirs)
    plot_all_synchronized(xenics_data, usrp_data, biopac_data, session_name, output_dirs)
    plot_individual_phases(xenics_data, usrp_data, biopac_data, session_name, output_dirs)
    
    # Relatório e export
    create_sync_report(biopac_data, output_dirs)
    export_synchronized_data(session_dir, session_name, xenics_data, usrp_data, biopac_data, output_dirs)
    
    print("\n" + "="*60)
    print("✅ ANÁLISE CONCLUÍDA!")
    print(f"📁 {output_dirs['base']}")
    print("   ├── plots/")
    print("   │   ├── pasat_performance.png  ← NOVO!")
    print("   │   ├── sync_verification_fixed_offset.png")
    print("   │   ├── individual/")
    print("   │   ├── phases/ ← 5 plots por teste")
    print("   │   └── all_synchronized.png")
    print("   ├── usrp/data.csv")
    print("   ├── biopac/ecg.csv")
    print("   ├── biopac/respiration.csv")
    print("   └── sync_report.txt")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()