import matplotlib.pyplot as plt
import numpy as np
import os

# --- Constants & Configuration ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Target: LLM_Research/visualization/plots
OUTPUT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../LLM_Research/visualization/plots"))

# ISOBUS / CAN Bus Physics
BAUD_RATE = 250000  # 250 kbps
# J1939 Transport Protocol (TP.DT) splits data into 7-byte payloads per frame
PAYLOAD_PER_FRAME = 7 
# Frame size estimation (Extended 29-bit ID):
# ~128 bits logical + ~20% bit stuffing = ~150 bits
BITS_PER_FRAME = 150 

# ECU Processing Overhead (Latency)
# The prompt mentions "real C++ firmware needs a few milliseconds to process packets".
# Since we process every frame (interrupt), even a small delay adds up massively.
# 0.0005 s = 0.5 ms per frame
ECU_LATENCY_PER_FRAME = 0.0005 

# Input Data
DATA_SIZES_BYTES = {
    "Markdown": 310053,
    "XML": 331454,
    "JSON": 539570
}

# Image Data: 200 KB PNG
IMAGE_SIZE_BYTES = 200 * 1024 
# ALL_IMAGES_BYTE = 12558285 
# IMAGE_SIZE_BYTES = ALL_IMAGES_BYTE
def calculate_transfer_stats(size_bytes):
    """
    Calculates detailed transfer statistics over ISOBUS.
    Returns a dict with wire_time, latency_time, total_time.
    """
    num_frames = np.ceil(size_bytes / PAYLOAD_PER_FRAME)
    
    # Physics of the wire
    total_bits = num_frames * BITS_PER_FRAME
    wire_time = total_bits / BAUD_RATE
    
    # ECU Overhead
    latency_time = num_frames * ECU_LATENCY_PER_FRAME
    
    return {
        "num_frames": int(num_frames),
        "wire_time": wire_time,
        "latency_time": latency_time,
        "total_time": wire_time + latency_time
    }

def plot_transfer_times():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Categories
    text_formats = ["Markdown", "XML", "JSON"]
    
    # Calculate times
    manual_stats = [calculate_transfer_stats(DATA_SIZES_BYTES[fmt]) for fmt in text_formats]
    image_stat = calculate_transfer_stats(IMAGE_SIZE_BYTES)
    
    # Prepare Plot Data
    labels = text_formats + ["1x Image\n(200kb PNG)"]
    values = [s["total_time"] for s in manual_stats] + [image_stat["total_time"]]
    
    # Colors: distinct for Text vs Image
    colors = ['#4c72b0', '#4c72b0', '#4c72b0', '#dd8452']

    # Console Output for Verification
    print(f"{'Item':<15} | {'Size (B)':<10} | {'Total Time(s)':<10}")
    print("-" * 45)
    for lbl, val in zip(labels, values):
        clean_lbl = lbl.replace('\n', ' ')
        print(f"{clean_lbl:<15} | {val:<10.2f}")

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(labels, values, color=colors, width=0.6)
    
    # Direct Labeling (No Legend)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}s', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Labels and Formatting
    ax.set_ylabel('Estimated Transfer Time (seconds)', fontsize=12)
    ax.set_title('Data Transfer Time over ISOBUS (250 kbps)', fontsize=14)
    
    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True) # Ensure grid is behind bars
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "bar_transfer_time_format.png")
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved to: {output_path}")

if __name__ == "__main__":
    plot_transfer_times()
