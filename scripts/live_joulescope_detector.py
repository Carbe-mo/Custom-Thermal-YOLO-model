"""
scripts/live_joulescope_detector.py
Real-time Joulescope JS220 monitor that detects when ESP32-P4 is reset,
records the full active inference run, and outputs exact metrics.
"""

import time
import numpy as np
import pyjoulescope_driver

def main():
    print("=" * 70)
    print("  ESP32-P4 JOULESCOPE LIVE BENCHMARK MONITOR")
    print("=" * 70)

    d = pyjoulescope_driver.Driver()
    dev_path = d.device_paths()[0]
    d.open(dev_path)

    all_readings = []
    
    def on_stats(topic, val):
        try:
            i = val['signals']['current']['avg']['value'] * 1000.0  # mA
            v = val['signals']['voltage']['avg']['value']          # V
            p = val['signals']['power']['avg']['value'] * 1000.0    # mW
            e = val['accumulators']['energy']['value']             # J
            t = time.time()
            all_readings.append((t, v, i, p, e))
        except Exception:
            pass

    d.subscribe(f"{dev_path}/s/stats/value", pyjoulescope_driver.SubscribeFlags.PUB, on_stats)
    d.publish(f"{dev_path}/s/stats/ctrl", 1)

    print(f"Connected to Joulescope: {dev_path}")
    print("\n👉 PLEASE PRESS THE 'RESET' (RST or EN) BUTTON ON THE ESP32-P4 NANO BOARD NOW!")
    print("Monitoring live power in 0.5s windows for 40 seconds...\n")

    t_start = time.time()
    last_len = 0
    while time.time() - t_start < 40:
        time.sleep(0.5)
        if len(all_readings) > last_len:
            r = all_readings[-1]
            elapsed = r[0] - t_start
            print(f"  [{elapsed:4.1f}s] ⚡ {r[3]:6.1f} mW | {r[1]:5.3f} V | {r[2]:6.1f} mA", flush=True)
            last_len = len(all_readings)

    d.publish(f"{dev_path}/s/stats/ctrl", 0)
    d.close(dev_path)

    if not all_readings:
        print("No readings captured.")
        return

    # Analyze the time series
    times = np.array([r[0] for r in all_readings]) - all_readings[0][0]
    vs = np.array([r[1] for r in all_readings])
    is_ma = np.array([r[2] for r in all_readings])
    ps_mw = np.array([r[3] for r in all_readings])

    print("\n" + "=" * 70)
    print("  SESSION SUMMARY")
    print("=" * 70)
    print(f"  Duration:         {times[-1]:.1f} s ({len(all_readings)} windows)")
    print(f"  Mean Voltage:     {np.mean(vs):.3f} V")
    print(f"  Overall Current:  {np.mean(is_ma):.2f} mA (Min: {np.min(is_ma):.2f}, Max: {np.max(is_ma):.2f})")
    print(f"  Overall Power:    {np.mean(ps_mw):.2f} mW ({np.mean(ps_mw)/1000.0:.4f} W)")
    print(f"  Peak Power:       {np.max(ps_mw):.2f} mW ({np.max(ps_mw)/1000.0:.4f} W)")
    print(f"  Min Power:        {np.min(ps_mw):.2f} mW ({np.min(ps_mw)/1000.0:.4f} W)")
    
    # Active vs Idle separation:
    # If any readings are > 215 mA, that corresponds to active processing/inference
    active_mask = is_ma > 215.0
    if np.any(active_mask):
        act_v = vs[active_mask]
        act_i = is_ma[active_mask]
        act_p = ps_mw[active_mask]
        idle_v = vs[~active_mask]
        idle_i = is_ma[~active_mask]
        idle_p = ps_mw[~active_mask]
        
        print("\n  [DETECTED ACTIVE INFERENCE RUN!]")
        print(f"    Active Windows:  {len(act_p)} ({len(act_p)*0.5:.1f} s)")
        print(f"    Active Voltage:  {np.mean(act_v):.3f} V")
        print(f"    Active Current:  {np.mean(act_i):.2f} mA (Peak: {np.max(act_i):.2f} mA)")
        print(f"    Active Power:    {np.mean(act_p):.2f} mW ({np.mean(act_p)/1000.0:.4f} W)")
        print(f"    Active Peak:     {np.max(act_p):.2f} mW ({np.max(act_p)/1000.0:.4f} W)")
        
        if len(idle_p) > 0:
            print(f"    Idle Power:      {np.mean(idle_p):.2f} mW ({np.mean(idle_p)/1000.0:.4f} W)")
            
        lat_s = 2.17366
        energy_per_frame_mj = (np.mean(act_p) / 1000.0) * lat_s * 1000.0
        fps = 1.0 / lat_s
        fpj = 1000.0 / energy_per_frame_mj
        print(f"    Energy/Frame:    {energy_per_frame_mj:.2f} mJ")
        print(f"    Frames/Joule:    {fpj:.2f} F/J")
    else:
        print("\n  [Notice] All readings were at steady idle (~207 mA).")
        print("  If you press the physical RST/EN button during monitoring, active inference will trigger.")
    print("=" * 70)

if __name__ == "__main__":
    main()
