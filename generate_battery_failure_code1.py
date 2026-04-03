import pybamm
import pandas as pd
import numpy as np

# 1. Khởi tạo mô hình và bộ giải (Dùng CasadiSolver để tránh lỗi JAX)
model = pybamm.lithium_ion.DFN(options={"thermal": "lumped"})
parameter_values = pybamm.ParameterValues("Chen2020")
safe_solver = pybamm.CasadiSolver(mode="safe")

all_cycles_data = []
num_cycles = 5 # Chạy 5 chu kỳ

print("Bắt đầu mô phỏng kịch bản lỗi Quá nạp (Overcharge Fault)...")

for i in range(1, num_cycles + 1):
    # --- MÔ PHỎNG LỖI QUÁ NẠP ---
    if i <= 2:
        # Chu kỳ 1-2: Bình thường (Sạc cắt ở 4.2V)
        v_max = 4.2
        ambient_temp = 24.0
    else:
        # Chu kỳ 3-5: BMS hỏng, sạc nhồi lên 4.45V (Quá nạp nghiêm trọng)
        # Lưu ý: Không nên đặt quá cao (như 5V) vì mô hình vật lý sẽ báo lỗi do nồng độ Li vượt mức 100%
        v_max = 4.45
        ambient_temp = 24.0 + (i - 2) # Nhiệt độ môi trường cũng bị ảnh hưởng nhẹ

    # Giữ nguyên hệ số tản nhiệt ở mức bình thường (10.0)
    parameter_values.update({
        "Total heat transfer coefficient [W.m-2.K-1]": 10.0,
        "Ambient temperature [K]": ambient_temp + 273.15,
    })

    # Thiết lập kịch bản sạc/xả tự động thay đổi theo v_max
    experiment = pybamm.Experiment([
        f"Charge at 1.5 A until {v_max} V",
        f"Hold at {v_max} V until 20 mA",
        "Rest for 5 minutes",
        "Discharge at 2.0 A until 2.7 V",
        "Rest for 5 minutes"
    ])

    print(f" Đang chạy chu kỳ {i} (Điện áp cắt: {v_max}V)...")

    try:
        sim = pybamm.Simulation(
            model,
            experiment=experiment,
            parameter_values=parameter_values,
            solver=safe_solver
        )
        sol = sim.solve()

        # Trích xuất dữ liệu thô
        time = sol["Time [s]"].entries
        voltage = sol["Terminal voltage [V]"].entries
        current_pybamm = sol["Current [A]"].entries
        temp = sol["Volume-averaged cell temperature [K]"].entries - 273.15

        # Đảo dấu dòng điện (xả âm, sạc dương) cho khớp với file mẫu của bạn
        current_measured = current_pybamm * -1.0

        # Lấy dung lượng của chu kỳ
        capacity = sol["Discharge capacity [A.h]"].entries[-1]

        # Thêm nhiễu (noise) vào điện áp ở các chu kỳ quá nạp do pin bị stress
        if i > 2:
            noise_v = np.random.normal(0, 0.015, len(voltage))
            voltage = voltage + noise_v

        # Phân loại trạng thái (charge, discharge, rest)
        types = []
        for c in current_pybamm:
            if c > 0.01:
                types.append("discharge")
            elif c < -0.01:
                types.append("charge")
            else:
                types.append("rest")

        # Giả lập Load (Trong lúc xả, Current_load dương, lúc sạc/nghỉ bằng 0)
        current_load = np.where(np.array(types) == "discharge", 2.0, 0.0)
        voltage_load = np.where(np.array(types) == "discharge", voltage, 0.0)

        # Đưa vào DataFrame với cấu trúc chuẩn
        df_cycle = pd.DataFrame({
            "Battery_ID": "B0005_Overcharge",
            "Cycle_Index": i,
            "Type": types,
            "Empty_column": "",
            "Ambient_Temperature": ambient_temp,
            "Time_step": time,
            "Voltage_measured": voltage,
            "Current_measured": current_measured,
            "Temperature_measured": temp,
            "Current_load": current_load,
            "Voltage_load": voltage_load,
            "Capacity": capacity
        })

        all_cycles_data.append(df_cycle)

    except Exception as e:
        print(f" Lỗi toán học tại chu kỳ {i} (Do ép pin quá mức): {e}")

# Xuất file CSV
if all_cycles_data:
    final_df = pd.concat(all_cycles_data, ignore_index=True)
    final_df.to_csv("overcharge_fault_5_cycles.csv", index=False)
    print("\nHoàn tất! File 'overcharge_fault_5_cycles.csv' đã được lưu thành công.")