#!/usr/bin/env python3
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
    print("\n[GNU Radio] Parando...")
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
        print("\n[GNU Radio] Interrompido")
        sys.stdout.flush()
        tb.stop()
        tb.wait()
    except Exception as e:
        print(f"[GNU Radio] ❌ Erro: {e}")
        sys.stdout.flush()
        import traceback
        traceback.print_exc()
        sys.exit(1)
